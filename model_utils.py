from unsloth import FastLanguageModel

def load_model_and_tokenizer(model_name):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        load_in_4bit=True,
        use_gradient_checkpointing=True,
    )
    return model, tokenizer

def get_peft_model(model):
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
        use_rslora = False,
        loftq_config = None,
)
    return model