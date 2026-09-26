from datasets import load_dataset
from transformers import pipeline


# 1. Load dataset
data = load_dataset(
    "Tobi-Bueck/customer-support-tickets",
    split="train"
)


# 2. Keep only English tickets with a body
english_data = data.filter(
    lambda row: row["language"] == "en"
    and row["body"] is not None
)


# 3. Create sentiment classifier
sentiment_classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


# 4. Sentiment analysis function
def analyze_sentiment(text):
    result = sentiment_classifier(text)[0]

    label = result["label"]
    score = result["score"]

    if score >= 0.90:
        confidence = "HIGH"
    elif score >= 0.70:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "label": label,
        "score": score,
        "confidence": confidence
    }


# 5. Analyze first 5 tickets
for i in range(5):
    ticket = english_data[i]

    body = ticket["body"]
    subject = ticket["subject"]

    analysis = analyze_sentiment(body)

    print("=" * 60)
    print(f"TICKET {i + 1}")
    print()

    print("SUBJECT:")
    print(subject)

    print("\nBODY:")
    print(body)

    print("\nSENTIMENT:")
    print(analysis["label"])

    print("\nSCORE:")
    print(round(analysis["score"], 4))

    print("\nCONFIDENCE:")
    print(analysis["confidence"])