## Qwen2.5-1.5B-Instruct 微调示例（GSM8K）

这个仓库包含一个可直接运行的脚本，用 `transformers + datasets` 对
`Qwen/Qwen2.5-1.5B-Instruct` 做监督微调（SFT）。默认数据集是 `gsm8k`，
你也可以替换为你能获取到的任何 Hugging Face 数据集。

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 运行微调（GSM8K 默认配置）

```bash
python train_qwen25_gsm8k.py \
  --output_dir ./qwen25-gsm8k-sft \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16
```

> 如果你的 GPU 不支持 bf16，请改成 `--fp16` 或不加混精参数。

### 3) 可替换为其他数据集

脚本支持通过参数替换数据集和字段名，例如：

```bash
python train_qwen25_gsm8k.py \
  --dataset_name your_dataset_name \
  --dataset_config your_config \
  --train_split train \
  --eval_split validation \
  --question_field instruction \
  --answer_field output
```

只要数据里有“问题字段 + 答案字段”，就可以复用这套流程。

### 4) 说明

- 脚本会把样本转换为聊天格式（system / user / assistant），并调用 Qwen tokenizer 的 chat template。
- 训练后模型和 tokenizer 会保存到 `--output_dir`。
- 默认使用 full fine-tuning（不是 LoRA），显存压力较大；若你需要，我可以继续给你补一版 LoRA/QLoRA 脚本。
