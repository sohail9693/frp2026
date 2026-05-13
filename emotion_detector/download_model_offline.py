from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

model_name = "j-hartmann/emotion-english-distilroberta-base"
save_directory = "./local_model"

print(f"Downloading model '{model_name}' to '{save_directory}'...")
print("This may take a minute or two depending on your internet connection.")

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)

# Download and save the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained(save_directory)

print("\n✅ Success! The model is now saved locally.")
print("You can completely disconnect from the internet and it will still run!")
