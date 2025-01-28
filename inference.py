from unsloth import FastLanguageModel
import torch
from model_utils import load_model_and_tokenizer

def generate_diet_plan(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        inputs.input_ids,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.9,
        top_k=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    diet_plan = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return diet_plan

def main():
    model_name = "fine-tuned-diet"
    model, tokenizer = load_model_and_tokenizer(model_name)
    model = FastLanguageModel.for_inference(model)

    print("Welcome to the cookbook chatbot! Enter your meal to generete ingredients for it and instruction how to make such meal , and I'll generate plan for you.")
    print("Type 'exit' to end the conversation.\n")

    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() == "exit":
            print("Chatbot: Goodbye! Have a great day!")
            break
        diet_plan = generate_diet_plan(user_prompt, model, tokenizer)
        print("\nChatbot: Here is your meal instruction:\n")
        print(diet_plan)
        print("\n" + "-" * 50 + "\n")

if __name__ == "__main__":
    main()