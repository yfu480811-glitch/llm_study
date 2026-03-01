#!/usr/bin/env python3
"""Fine-tune Qwen2.5-1.5B-Instruct on GSM8K (or another HF dataset).

Example:
    python train_qwen25_gsm8k.py \
        --output_dir ./qwen25-gsm8k-sft \
        --num_train_epochs 2 \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8
"""

from __future__ import annotations

import argparse
import os
from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_PROMPT_TEMPLATE = (
    "You are a helpful math tutor. Solve the following problem step by step, "
    "then provide the final answer clearly.\n\n"
    "Problem: {question}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-1.5B-Instruct on GSM8K.")

    # Model / data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument(
        "--question_field",
        type=str,
        default="question",
        help="Field name for question text in dataset.",
    )
    parser.add_argument(
        "--answer_field",
        type=str,
        default="answer",
        help="Field name for answer text in dataset.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Python format template with {question} placeholder.",
    )

    # Sequence / tokenization
    parser.add_argument("--max_length", type=int, default=1024)

    # Training params
    parser.add_argument("--output_dir", type=str, default="./qwen25-gsm8k-sft")
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable bf16 training (recommended on Ampere+ GPUs).",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable fp16 training.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory.",
    )

    return parser.parse_args()


def build_chat_sample(question: str, answer: str, tokenizer: AutoTokenizer, prompt_template: str) -> str:
    user_prompt = prompt_template.format(question=question)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": answer},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def preprocess_dataset(
    ds: DatasetDict,
    tokenizer: AutoTokenizer,
    question_field: str,
    answer_field: str,
    prompt_template: str,
    max_length: int,
) -> DatasetDict:
    def _to_model_inputs(examples: dict[str, list[Any]]) -> dict[str, Any]:
        texts = []
        for q, a in zip(examples[question_field], examples[answer_field]):
            texts.append(build_chat_sample(str(q), str(a), tokenizer, prompt_template))

        tokenized = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    remove_columns = ds[next(iter(ds.keys()))].column_names

    return ds.map(
        _to_model_inputs,
        batched=True,
        remove_columns=remove_columns,
        desc="Tokenizing dataset",
    )


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    raw_ds = load_dataset(args.dataset_name, args.dataset_config)

    dataset = DatasetDict(
        {
            "train": raw_ds[args.train_split],
            "eval": raw_ds[args.eval_split],
        }
    )

    tokenized_ds = preprocess_dataset(
        dataset,
        tokenizer,
        question_field=args.question_field,
        answer_field=args.answer_field,
        prompt_template=args.prompt_template,
        max_length=args.max_length,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
