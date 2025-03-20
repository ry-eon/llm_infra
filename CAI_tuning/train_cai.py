from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTConfig, SFTTrainer
import torch
import os 
from datasets import load_dataset, concatenate_datasets
from fsdp import FSDPConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import argparse
import torch.nn as nn

class TrainerCAI:
    def __init__(self, model_id):
        self.model_id = model_id


        # 경로 설정
        self.log_path, self.model_cache_path, self.tokenizer_cache_path, self.checkpoint_path = self.make_path(model_id)
        
        # 모델 및 토크나이저 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=self.model_cache_path,
        )
        self.fsdp_layer = self.get_fsdp_layer_to_wrap(self.model)
        print(f"fsdp_transformer_layer_cls_to_wrap: {self.fsdp_layer}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=self.tokenizer_cache_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

    def make_path(self, model_id):
        base_path = model_id.split("/")[-1]
        log_path = os.path.join("./log", base_path)
        model_cache_path = os.path.join("./model", base_path)
        tokenizer_cache_path = os.path.join("./tokenizer", base_path)
        checkpoint_path = os.path.join("./checkpoint_sl", base_path)
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        return log_path, model_cache_path, tokenizer_cache_path, checkpoint_path
    
    def get_fsdp_layer_to_wrap(self, model):
        for layer in model.modules():
            if isinstance(layer, nn.Module) and layer.__class__.__name__ == "LlamaDecoderLayer":
                return layer.__class__.__name__
        return None

    def formatting_prompts_func_chosen(self, row):
        chosens = row["chosen"]
        instructions = row["instruction"]
        contexts = row["context"]
        texts = []
        for chosen, instruction, context in zip(chosens, instructions, contexts):
            chosen_text = []
            context_text = []
            for c in context:
                if c["from"] == "human":
                    context_text.append({"role": "user", "content": c["value"]})
                else:
                    context_text.append({"role": "assistant", "content": c["value"]}) 
            
            chosen_text += context_text
            chosen_text.append({"role": "user", "content": instruction["value"]})
            chosen_text.append({"role": "assistant", "content": chosen["value"]})
            texts.append(self.tokenizer.apply_chat_template(chosen_text, tokenize=False))
        return {"text": texts}

    def formatting_prompts_func_rejected(self, row):
        rejecteds = row["rejected"]
        instructions = row["instruction"]
        contexts = row["context"]
        texts = []
        for rejected, instruction, context in zip(rejecteds, instructions, contexts):
            rejected_text = []
            context_text = []
            for c in context:
                if c["from"] == "human":
                    context_text.append({"role": "user", "content": c["value"]})
                else:
                    context_text.append({"role": "assistant", "content": c["value"]}) 
            
            rejected_text += context_text
            rejected_text.append({"role": "user", "content": instruction["value"]})
            rejected_text.append({"role": "assistant", "content": rejected["value"]})
            texts.append(self.tokenizer.apply_chat_template(rejected_text, tokenize=False))
        return {"text": texts}

    def train(self):
        # 데이터셋 로드 및 처리
        dataset = load_dataset("heegyu/hh-rlhf-ko", split="train")
        dataset_chosen = dataset.map(self.formatting_prompts_func_chosen, 
                                     batched=True)
        dataset_rejected = dataset.map(self.formatting_prompts_func_rejected, 
                                       batched=True)
        dataset_final = concatenate_datasets([dataset_chosen, dataset_rejected])
        dataset_final = dataset_final.shuffle(seed=65)

        # FSDP 설정
        fsdp_config = FSDPConfig()
        
        # TrainingArguments 설정
        training_args = TrainingArguments(
            evaluation_strategy="no",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            ddp_backend="nccl",
            learning_rate=1e-5,
            fp16=fsdp_config.use_fp16,
            bf16=fsdp_config.pure_bf16, 
            max_steps=-1,
            num_train_epochs=1,
            save_strategy="epoch",
            logging_steps=100,
            output_dir=self.checkpoint_path,
            logging_dir=self.log_path,
            optim="paged_adamw_32bit",
            lr_scheduler_type="linear",
            fsdp="full_shard auto_wrap",
            fsdp_config={
                "fsdp_offload_params": fsdp_config.fsdp_cpu_offload,
                "fsdp_state_dict_type": fsdp_config.checkpoint_type,
                "fsdp_transformer_layer_cls_to_wrap": self.fsdp_layer,
            },
        )

        # SFTTrainer 설정 및 학습
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=dataset_final,
            dataset_text_field="text",
            max_seq_length=1024,
        )
        
        trainer.train()

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--model_id', type=str, required=True, help="ID of the model to be used for training")
    return parser.parse_args()

def main():
    args = parse_args()
    trainer = TrainerCAI(model_id=args.model_id)
    trainer.train()

if __name__ == "__main__":
    main()
