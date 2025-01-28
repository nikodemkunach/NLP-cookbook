import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from data_preprocessing import load_and_preprocess_data
from model_utils import load_model_and_tokenizer, get_peft_model

def train():
    model_name = "unsloth/gemma-2-9b-it-bnb-4bit"
    model, tokenizer = load_model_and_tokenizer(model_name)

    model = get_peft_model(model)

    formatted_dataset = load_and_preprocess_data()

    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer_stats = trainer.train()

    trainer.save_model("fine-tuned-diet")
    tokenizer.save_pretrained("fine-tuned-diet")

if __name__ == "__main__":
    train()