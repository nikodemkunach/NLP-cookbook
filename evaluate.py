from nltk.translate.bleu_score import sentence_bleu
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt

def generate_directions(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_length=500, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def calculate_bleu_score(reference, candidate):
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

model_name = "fine-tuned-diet"
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = pd.read_csv("data/test_products.csv").head(50)

bleu_scores = []

for index, row in dataset.iterrows():
    prompt = f"{row['recipe_name']}, {row['directions']}"
    generated_directions = generate_directions(model, tokenizer, prompt)
    reference_directions = row["directions"]
    bleu_score = calculate_bleu_score(reference_directions, generated_directions)
    bleu_scores.append(bleu_score)
    print(f"\nPosiłek: {row['recipe_name']}")
    print(f"Prompt: {prompt}")
    print(f"Wygenerowane wskazówki: {generated_directions}")
    print(f"Oczekiwane wskazówki: {reference_directions}")
    print(f"BLEU Score: {bleu_score}")
    print("-" * 50)

average_bleu_score = sum(bleu_scores) / len(bleu_scores)
print(f"\nŚrednia wartość BLEU score: {average_bleu_score}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(bleu_scores) + 1), bleu_scores, marker='o', linestyle='-', color='b')
plt.axhline(y=average_bleu_score, color='r', linestyle='--', label=f'Średnia BLEU: {average_bleu_score:.4f}')
plt.xlabel('Numer przykładu')
plt.ylabel('BLEU Score')
plt.title('BLEU Score dla każdego przykładu')
plt.legend()
plt.grid(True)

output_file = "bleu_score_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nWykres został zapisany do pliku: {output_file}")

plt.show()
