import torch
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer
)
import pandas as pd
from peft import LoraConfig, get_peft_model, TaskType
import os
from fsdp import FSDPConfig
from torch.utils.data import Dataset

# GPU 설정
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6"


# Custom Dataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        # OpenAI 포맷으로 프롬프트 구성
        messages = [
            {"role": "system", "content": item['system']},
            {"role": "user", "content": item['question']},
            {"role": "assistant", "content": item['answer']}
        ]
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"<|im_start|>system\n{message['content']}<|im_end|>\n"
            elif message["role"] == "user":
                prompt += f"<|im_start|>user\n{message['content']}<|im_end|>\n"
            elif message["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{message['content']}<|im_end|>\n"

        # 토크나이징
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        # labels 생성 (input_ids와 동일하게 설정)
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()

        return encoding


def lora_wrap_policy(module):
    return any(hasattr(module, "lora_A") or hasattr(module, "lora_B") for name, module in module.named_modules())


def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="./tokenizer/llama-3.2-Korean-Bllossom-3B")
    # OpenAI 포맷 토큰 추가
    tokenizer.add_special_tokens({
        "pad_token": "<|im_end|>",
        "eos_token": "<|im_end|>",
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
    })
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir="./model/llama-3.2-Korean-Bllossom-3B",
    )
    # 토크나이저에 추가된 토큰에 대한 임베딩 크기 조정
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model


def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    return model


def prepare_dataset(tokenizer, excel_path, train_ratio=0.9):
    df = pd.read_excel(excel_path)
    train_size = int(len(df) * train_ratio)
    train_df = df[:train_size]
    eval_df = df[train_size:]

    train_dataset = CustomDataset(train_df, tokenizer)
    eval_dataset = CustomDataset(eval_df, tokenizer)

    return train_dataset, eval_dataset


def train_model(model, tokenizer, train_dataset, eval_dataset, fsdp_config, output_dir="./llama_lora_finetuned_hal",
                resume_from_checkpoint=None):
    training_args = TrainingArguments(
        output_dir=output_dir,
        gradient_accumulation_steps=16,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        logging_steps=10,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=15,
        weight_decay=0.01,
        save_total_limit=3,
        fp16=fsdp_config.use_fp16,
        bf16=fsdp_config.pure_bf16,
        report_to="none",
        load_best_model_at_end=True,
        dataloader_num_workers=4,
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "sharding_strategy": fsdp_config.sharding_strategy,
            "cpu_offload": fsdp_config.fsdp_cpu_offload,
            "mixed_precision": fsdp_config.mixed_precision,
            "auto_wrap_policy": lora_wrap_policy,
            "activation_checkpointing": True,
            "limit_all_gathers": True
        }
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    MODEL_NAME = "Bllossom/llama-3.2-Korean-Bllossom-3B"
    EXCEL_PATH = "train1.xlsx"
    fsdp_config = FSDPConfig()
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
    model = apply_lora(model)
    train_dataset, eval_dataset = prepare_dataset(tokenizer, EXCEL_PATH)
    train_model(model, tokenizer, train_dataset, eval_dataset, fsdp_config)