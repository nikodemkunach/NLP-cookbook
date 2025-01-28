import pandas as pd
from datasets import Dataset

def load_and_preprocess_data(file_path='data/train_products.csv'):
    df = pd.read_csv(file_path)

    dataset = Dataset.from_pandas(df)

    def formatting_prompts_func(examples):
        instructions = []
        inputs = []
        outputs = []

        for name, description, recipe_instructions in zip(examples['recipe_name'], examples['ingredients'], examples['directions']):
            instruction = f"Przygotuj przepis na {name}"
            input = description
            output = recipe_instructions

            instructions.append(instruction)
            inputs.append(input)
            outputs.append(output)

        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
            texts.append(text)

        return {"text": texts}

    formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
    return formatted_dataset