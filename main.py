from datasets import load_dataset

dataset = load_dataset(
    "Tobi-Bueck/customer-support-tickets",
    split="train"
)

small_data=dataset.select(range(20))
print(small_data)

"""for row in small_data:
    print(row["subject"])
"""

refund_data = dataset.filter(
    lambda row:  row["body"] is not None and
    "refund" in row["body"].lower()
)
print(refund_data)







