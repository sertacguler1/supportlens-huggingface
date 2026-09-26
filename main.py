from datasets import load_dataset

dataset = load_dataset("Tobi-Bueck/customer-support-tickets")

train_data = dataset["train"]

print(len(train_data))




