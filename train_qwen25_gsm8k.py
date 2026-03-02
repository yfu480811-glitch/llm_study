#!/usr/bin/env python3
"""LoRA SFT training script for Qwen2.5-1.5B-Instruct on GSM8K."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)

DEFAULT_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA SFT for Qwen2.5-1.5B-Instruct on GSM8K")

    # Model / data
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--local_model_path",
        type=str,
        default="./Qwen2.5-1.5B-Instruct",
        help="Local model directory. If exists, it will be preferred over model_name.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load model/tokenizer using local files only (offline mode).",
    )
    parser.add_argument("--dataset_name", type=str, default="gsm8k")
    parser.add_argument("--dataset_config", type=str, default="main")
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default="./gsm8k_main",
        help="Local dataset saved by datasets.save_to_disk. If exists, it will be preferred.",
    )
    parser.add_argument(
        "--dataset_local_files_only",
        action="store_true",
        help="When loading from hub, only use local cache (offline).",
    )
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="test")
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--answer_field", type=str, default="answer")

    # Sequence / precision / memory
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # LoRA
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default=DEFAULT_TARGET_MODULES,
        help="Comma-separated target module names for LoRA",
    )

    # Training args
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen25-gsm8k-lora")
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
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
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "tensorboard", "wandb"],
    )
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Saving options
    parser.add_argument(
        "--merge_and_save",
        action="store_true",
        help="Merge LoRA into base model and save merged model",
    )
    parser.add_argument("--merged_output_dir", type=str, default=None)

    return parser.parse_args()


def build_chat_sample(question: str, answer: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful math tutor."},
        {"role": "user", "content": question.strip()},
        {"role": "assistant", "content": answer.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def preprocess_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    question_field: str,
    answer_field: str,
) -> Dataset:
    if question_field not in dataset.column_names or answer_field not in dataset.column_names:
        raise ValueError(
            f"Dataset missing field(s). Available fields: {dataset.column_names}. "
            f"Got question_field={question_field}, answer_field={answer_field}."
        )

    def _valid(example: dict[str, Any]) -> bool:
        q = example.get(question_field)
        a = example.get(answer_field)
        return q is not None and a is not None and str(q).strip() != "" and str(a).strip() != ""

    filtered = dataset.filter(_valid, desc="Filtering empty samples")

    def _format(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        texts = [
            build_chat_sample(str(q), str(a), tokenizer)
            for q, a in zip(batch[question_field], batch[answer_field])
        ]
        return {"text": texts}

    text_ds = filtered.map(
        _format,
        batched=True,
        remove_columns=filtered.column_names,
        desc="Formatting chat data",
    )
    return text_ds


def tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names, desc="Tokenizing")


def print_length_stats(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int, tag: str) -> None:
    if len(dataset) == 0:
        print(f"[WARN] {tag} dataset is empty after filtering.")
        return

    sample_size = min(128, len(dataset))
    subset = dataset.select(range(sample_size))
    lengths = [len(tokenizer(x["text"], truncation=True, max_length=max_length)["input_ids"]) for x in subset]
    avg_len = sum(lengths) / len(lengths)
    print(
        f"[INFO] {tag} token length stats over {sample_size} samples: "
        f"min={min(lengths)}, p50={sorted(lengths)[len(lengths)//2]}, "
        f"avg={avg_len:.1f}, max={max(lengths)}"
    )


def build_model_and_lora(args: argparse.Namespace, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id

    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    try:
        model = get_peft_model(model, lora_cfg)
    except ValueError as e:
        raise ValueError(
            "LoRA target_modules may not match model modules. "
            f"Got target_modules={target_modules}. Original error: {e}"
        ) from e

    trainable, total = 0, 0
    for _, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    ratio = 100.0 * trainable / total if total > 0 else 0.0
    print(f"[INFO] Trainable params: {trainable:,} / {total:,} ({ratio:.4f}%)")

    return model


def print_env_summary() -> None:
    cuda_ok = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count() if cuda_ok else 0
    device = "cuda" if cuda_ok else "cpu"
    print("[INFO] ===== Environment Summary =====")
    print(f"[INFO] torch: {torch.__version__}")
    print(f"[INFO] cuda available: {cuda_ok}")
    print(f"[INFO] num gpus: {n_gpu}")
    print(f"[INFO] device: {device}")
    if cuda_ok:
        print(f"[INFO] gpu[0]: {torch.cuda.get_device_name(0)}")


def main() -> None:
    args = parse_args()
    print_env_summary()

    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 cannot both be enabled.")

    if args.resume_from_checkpoint and not os.path.exists(args.resume_from_checkpoint):
        raise FileNotFoundError(
            f"resume_from_checkpoint path does not exist: {args.resume_from_checkpoint}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    model_source = args.model_name
    if args.local_model_path and os.path.isdir(args.local_model_path):
        model_source = args.local_model_path
        print(f"[INFO] Using local model path: {model_source}")
    else:
        print(f"[INFO] Using remote model id/path: {model_source}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.local_dataset_path and os.path.isdir(args.local_dataset_path):
        print(f"[INFO] Using local dataset path: {args.local_dataset_path}")
        raw_ds = load_from_disk(args.local_dataset_path)
    else:
        print(f"[INFO] Using remote dataset id: {args.dataset_name}/{args.dataset_config}")
        download_config = DownloadConfig(local_files_only=args.dataset_local_files_only)
        raw_ds = load_dataset(
            args.dataset_name,
            args.dataset_config,
            download_config=download_config,
        )
    if args.train_split not in raw_ds or args.eval_split not in raw_ds:
        raise ValueError(
            f"Invalid split name. Available splits: {list(raw_ds.keys())}. "
            f"Got train_split={args.train_split}, eval_split={args.eval_split}."
        )

    train_text = preprocess_dataset(
        raw_ds[args.train_split], tokenizer, args.question_field, args.answer_field
    )
    eval_text = preprocess_dataset(
        raw_ds[args.eval_split], tokenizer, args.question_field, args.answer_field)

    print_length_stats(train_text, tokenizer, args.max_length, tag="train")
    print_length_stats(eval_text, tokenizer, args.max_length, tag="eval")

    train_ds = tokenize_dataset(train_text, tokenizer, args.max_length)
    eval_ds = tokenize_dataset(eval_text, tokenizer, args.max_length)

    args.model_name = model_source
    model = build_model_and_lora(args, tokenizer)

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
        seed=args.seed,
        report_to=args.report_to,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        logging_first_step=True,
    )

    # Let collator build padded tensors + labels safely to avoid nested labels errors.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    try:
        train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        print(f"[INFO] Training finished. global_step={train_result.global_step}")
    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            "CUDA OOM encountered. Try reducing --max_length / batch size, "
            "increasing --gradient_accumulation_steps, enabling --gradient_checkpointing, "
            "or using LoRA with smaller rank."
        ) from e

    eval_metrics = trainer.evaluate()
    print(f"[INFO] Eval metrics: {eval_metrics}")

    # Save LoRA adapter + tokenizer (required)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[INFO] Saved LoRA adapter and tokenizer to: {args.output_dir}")

    if args.merge_and_save:
        merged_dir = args.merged_output_dir or os.path.join(args.output_dir, "merged")
        os.makedirs(merged_dir, exist_ok=True)
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"[INFO] Saved merged model to: {merged_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[WARN] Interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
