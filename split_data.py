import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data/modified_recipes2_with_calories_cleaned.csv")

train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)

test_data, temp_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_data.to_csv("data/train_products.csv", index=False)
test_data.to_csv("data/test_products.csv", index=False)

print("Podział danych zakończony. Zbiory zapisane do plików")