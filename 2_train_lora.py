from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

import os
import torch
from datasets import load_dataset

from trl import SFTTrainer, SFTConfig

# Model setting
max_seq_length = 2048
dtype = None
load_in_4bit = True

print("베이스 AI 모델(Qwen2.5-Coder-7B) 로딩 중...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-7B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# LoRA
print("LoRA 어댑터 칩셋 부착 중...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Load training data
print("데이터셋(dataset.jsonl) 불러오는 중...")
dataset_path = "dataset.jsonl"
if not os.path.exists(dataset_path):
    print("❌ dataset.jsonl 파일이 없습니다. 1_data_preprocessor.py를 먼저 실행하세요!")
    exit(1)

dataset = load_dataset("json", data_files=dataset_path, split="train")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input (C++ Code):
{}

### Response (trace.json):
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

mapped_dataset = dataset.map(
    formatting_prompts_func,
    batched = True,
    remove_columns = dataset.column_names
)

# Train
print("훈련 시작")
trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = mapped_dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        remove_unused_columns = True,
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # 전체 데이터 모두 학습
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
print("훈련 완료")

# save
output_model_name = "lora_model_visualalgo"
model.save_pretrained(output_model_name)
print(f"'{output_model_name}' 폴더에 저장됨")
