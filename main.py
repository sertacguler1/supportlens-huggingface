import os

import torch
from datasets import load_dataset
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_NAME = "Tobi-Bueck/customer-support-tickets"

SENTIMENT_MODEL = (
    "distilbert-base-uncased-finetuned-sst-2-english"
)

GRAMMAR_MODEL = (
    "abdulmatinomotoso/English_Grammar_Checker"
)

ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

SUMMARIZATION_MODEL = "sshleifer/distilbart-cnn-12-6"

QA_MODEL = "distilbert-base-cased-distilled-squad"

QNLI_MODEL = "cross-encoder/qnli-electra-base"

GENERATION_MODEL = "openai-community/gpt2"

POLICY_PATH = "documents/company_policy.pdf"


CATEGORIES = [
    "payment or billing problem",
    "refund request",
    "software or technical problem",
    "login or account access problem",
    "product feature request",
    "general customer question",
]


# ============================================================
# LOAD DATASET
# ============================================================

data = load_dataset(
    DATASET_NAME,
    split="train",
)

english_data = data.filter(
    lambda row:
        row["language"] == "en"
        and row["body"] is not None
)


# ============================================================
# LOAD PIPELINES
# ============================================================

sentiment_classifier = pipeline(
    task="text-classification",
    model=SENTIMENT_MODEL,
)

grammar_checker = pipeline(
    task="text-classification",
    model=GRAMMAR_MODEL,
)

zero_shot_classifier = pipeline(
    task="zero-shot-classification",
    model=ZERO_SHOT_MODEL,
)

summarizer = pipeline(
    task="summarization",
    model=SUMMARIZATION_MODEL,
)

qa_pipeline = pipeline(
    task="question-answering",
    model=QA_MODEL,
)

qnli_classifier = pipeline(
    task="text-classification",
    model=QNLI_MODEL,
)

generator = pipeline(
    task="text-generation",
    model=GENERATION_MODEL,
)


# ============================================================
# AUTO CLASSES
# ============================================================

sentiment_tokenizer = AutoTokenizer.from_pretrained(
    SENTIMENT_MODEL
)

sentiment_model = (
    AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL
    )
)

comparison_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-cased"
)


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_text(value):
    return value or ""


def confidence_level(score):
    if score >= 0.90:
        return "HIGH"

    if score >= 0.70:
        return "MEDIUM"

    return "LOW"


# ============================================================
# SENTIMENT WITH PIPELINE
# ============================================================

def analyze_sentiment(text):
    result = sentiment_classifier(
        text,
        truncation=True,
    )[0]

    return {
        "label": result["label"],
        "score": result["score"],
        "confidence": confidence_level(
            result["score"]
        ),
    }


# ============================================================
# GRAMMAR
# ============================================================

def analyze_grammar(text):
    result = grammar_checker(
        text,
        truncation=True,
    )[0]

    return {
        "label": result["label"],
        "score": result["score"],
    }


# ============================================================
# ZERO-SHOT CATEGORY
# ============================================================

def analyze_category(text):
    result = zero_shot_classifier(
        text,
        candidate_labels=CATEGORIES,
    )

    top_score = result["scores"][0]

    if top_score >= 0.75:
        confidence = "HIGH"
    elif top_score >= 0.50:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "top_label": result["labels"][0],
        "top_score": top_score,
        "confidence": confidence,
        "labels": result["labels"],
        "scores": result["scores"],
    }


# ============================================================
# TOKENIZER INSPECTION
# ============================================================

def inspect_tokenization(text):
    tokens = sentiment_tokenizer.tokenize(
        text
    )

    encoding = sentiment_tokenizer(
        text,
        truncation=True,
        return_tensors="pt",
    )

    return {
        "tokens": tokens,
        "token_ids": encoding[
            "input_ids"
        ][0].tolist(),
        "attention_mask": encoding[
            "attention_mask"
        ][0].tolist(),
        "token_count": len(tokens),
        "special_tokens": {
            "cls_token":
                sentiment_tokenizer.cls_token,

            "sep_token":
                sentiment_tokenizer.sep_token,

            "pad_token":
                sentiment_tokenizer.pad_token,

            "unk_token":
                sentiment_tokenizer.unk_token,
        },
    }


# ============================================================
# TOKENIZER COMPARISON
# ============================================================

def compare_tokenizers(text):
    distilbert_tokens = (
        sentiment_tokenizer.tokenize(text)
    )

    bert_tokens = (
        comparison_tokenizer.tokenize(text)
    )

    return {
        "distilbert_uncased":
            distilbert_tokens,

        "bert_cased":
            bert_tokens,
    }


# ============================================================
# MANUAL SENTIMENT
# AutoTokenizer + AutoModel
# ============================================================

def analyze_sentiment_manual(text):

    inputs = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = sentiment_model(**inputs)

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    predicted_id = int(
        torch.argmax(
            probabilities,
            dim=-1,
        ).item()
    )

    score = float(
        probabilities[0][predicted_id]
    )

    label = sentiment_model.config.id2label[
        predicted_id
    ]

    return {
        "label": label,
        "score": score,
        "confidence":
            confidence_level(score),
        "logits":
            logits[0].tolist(),
        "probabilities":
            probabilities[0].tolist(),
    }


# ============================================================
# SUMMARIZATION
# ============================================================

def summarize_ticket(text):

    if len(text) < 300:
        return text

    result = summarizer(
        text,
        truncation=True,
        max_new_tokens=80,
        min_new_tokens=20,
    )

    return result[0]["summary_text"]


# ============================================================
# PDF EXTRACTION
# ============================================================

def load_policy_document():

    if not os.path.exists(POLICY_PATH):
        return None

    reader = PdfReader(
        POLICY_PATH
    )

    document_text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            document_text += (
                page_text + "\n"
            )

    return document_text


# ============================================================
# DOCUMENT QUESTION ANSWERING
# ============================================================

def ask_policy(question):

    document_text = (
        load_policy_document()
    )

    if not document_text:
        return {
            "answer":
                "Policy PDF not found.",
            "score": 0.0,
        }

    result = qa_pipeline(
        question=question,
        context=document_text,
    )

    return {
        "answer": result["answer"],
        "score": result["score"],
    }


# ============================================================
# QNLI
# ============================================================

def check_answer_relevance(
    question,
    answer,
):

    combined_text = (
        question
        + " [SEP] "
        + answer
    )

    result = qnli_classifier(
        combined_text,
        truncation=True,
    )[0]

    return {
        "label": result["label"],
        "score": result["score"],
    }


# ============================================================
# LOCAL TEXT GENERATION
# ============================================================

def generate_support_response(
    customer_text,
):

    prompt = (
        "Customer: "
        + customer_text
        + "\nSupport Agent:"
    )

    results = generator(
        prompt,
        max_new_tokens=60,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=
            generator.tokenizer.eos_token_id,
    )

    return results[0][
        "generated_text"
    ]


# ============================================================
# OPTIONAL REMOTE HUGGING FACE INFERENCE
# ============================================================

def remote_chat(message):

    token = os.getenv(
        "HF_TOKEN"
    )

    if not token:
        return (
            "HF_TOKEN environment "
            "variable is not set."
        )

    client = InferenceClient(
        api_key=token
    )

    response = (
        client.chat.completions.create(
            model=
                "deepseek-ai/DeepSeek-V3",

            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
        )
    )

    return (
        response
        .choices[0]
        .message
        .content
    )


# ============================================================
# COMPLETE TICKET ANALYSIS
# ============================================================

def analyze_ticket(text):

    sentiment = (
        analyze_sentiment(text)
    )

    manual_sentiment = (
        analyze_sentiment_manual(text)
    )

    grammar = (
        analyze_grammar(text)
    )

    category = (
        analyze_category(text)
    )

    summary = (
        summarize_ticket(text)
    )

    token_info = (
        inspect_tokenization(text)
    )

    return {
        "original_text": text,

        "sentiment":
            sentiment,

        "manual_sentiment":
            manual_sentiment,

        "grammar":
            grammar,

        "category":
            category,

        "summary":
            summary,

        "tokenization":
            token_info,
    }


# ============================================================
# OUTPUT
# ============================================================

def print_ticket_analysis(
    analysis,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SUPPORTLENS ANALYSIS"
    )

    print(
        "=" * 70
    )

    print("\nORIGINAL TEXT:")
    print(
        analysis[
            "original_text"
        ]
    )

    sentiment = (
        analysis["sentiment"]
    )

    print("\nSENTIMENT:")
    print(
        sentiment["label"]
    )

    print(
        "Score:",
        round(
            sentiment["score"],
            4,
        ),
    )

    print(
        "Confidence:",
        sentiment["confidence"],
    )

    manual = (
        analysis[
            "manual_sentiment"
        ]
    )

    print(
        "\nMANUAL AUTO MODEL "
        "SENTIMENT:"
    )

    print(
        manual["label"]
    )

    print(
        "Score:",
        round(
            manual["score"],
            4,
        ),
    )

    grammar = (
        analysis["grammar"]
    )

    print("\nGRAMMAR:")

    print(
        grammar["label"]
    )

    print(
        "Score:",
        round(
            grammar["score"],
            4,
        ),
    )

    category = (
        analysis["category"]
    )

    print("\nCATEGORY:")

    print(
        category["top_label"]
    )

    print(
        "Score:",
        round(
            category[
                "top_score"
            ],
            4,
        ),
    )

    print(
        "Confidence:",
        category[
            "confidence"
        ],
    )

    print(
        "\nTOP 3 CATEGORIES:"
    )

    for label, score in zip(
        category["labels"][:3],
        category["scores"][:3],
    ):
        print(
            f"{label}: "
            f"{score:.4f}"
        )

    print("\nSUMMARY:")

    print(
        analysis["summary"]
    )

    tokens = (
        analysis[
            "tokenization"
        ]
    )

    print("\nTOKEN COUNT:")

    print(
        tokens[
            "token_count"
        ]
    )

    print("\nTOKENS:")

    print(
        tokens["tokens"]
    )


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataset():

    print("\nDATASET")
    print("=" * 60)

    print(
        "Rows:",
        len(english_data),
    )

    print(
        "Columns:",
        english_data.column_names,
    )

    print(
        "\nExample ticket:"
    )

    ticket = (
        english_data[0]
    )

    print(
        safe_text(
            ticket["subject"]
        )
    )

    print()

    print(
        safe_text(
            ticket["body"]
        )
    )


# ============================================================
# DATASET FILTERING
# ============================================================

def filter_dataset():

    keyword = input(
        "Keyword: "
    ).lower()

    filtered = (
        english_data.filter(
            lambda row:
                keyword
                in safe_text(
                    row["body"]
                ).lower()
        )
    )

    print(
        f"\nFound "
        f"{len(filtered)} tickets."
    )

    for ticket in filtered.select(
        range(
            min(
                5,
                len(filtered),
            )
        )
    ):

        print(
            "\n"
            + "-" * 60
        )

        print(
            safe_text(
                ticket["subject"]
            )
        )

        print(
            safe_text(
                ticket["body"]
            )
        )


# ============================================================
# TOKENIZER MENU
# ============================================================

def tokenizer_menu():

    text = input(
        "\nEnter text: "
    )

    inspection = (
        inspect_tokenization(
            text
        )
    )

    comparison = (
        compare_tokenizers(
            text
        )
    )

    print("\nTOKENS:")

    print(
        inspection["tokens"]
    )

    print("\nTOKEN IDS:")

    print(
        inspection[
            "token_ids"
        ]
    )

    print(
        "\nATTENTION MASK:"
    )

    print(
        inspection[
            "attention_mask"
        ]
    )

    print(
        "\nSPECIAL TOKENS:"
    )

    print(
        inspection[
            "special_tokens"
        ]
    )

    print(
        "\nDISTILBERT TOKENS:"
    )

    print(
        comparison[
            "distilbert_uncased"
        ]
    )

    print(
        "\nBERT CASED TOKENS:"
    )

    print(
        comparison[
            "bert_cased"
        ]
    )


# ============================================================
# POLICY QA MENU
# ============================================================

def policy_menu():

    question = input(
        "\nQuestion: "
    )

    result = (
        ask_policy(question)
    )

    print("\nANSWER:")

    print(
        result["answer"]
    )

    print(
        "\nQA SCORE:",
        round(
            result["score"],
            4,
        ),
    )

    relevance = (
        check_answer_relevance(
            question,
            result["answer"],
        )
    )

    print(
        "\nQNLI RESULT:"
    )

    print(
        relevance["label"]
    )

    print(
        "Score:",
        round(
            relevance["score"],
            4,
        ),
    )


# ============================================================
# CLI
# ============================================================

def main():

    while True:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "SUPPORTLENS"
        )

        print(
            "=" * 70
        )

        print(
            "1. Inspect dataset"
        )

        print(
            "2. Filter dataset"
        )

        print(
            "3. Analyze ticket"
        )

        print(
            "4. Inspect tokenization"
        )

        print(
            "5. Ask company policy"
        )

        print(
            "6. Generate support response"
        )

        print(
            "7. Remote Hugging Face chat"
        )

        print(
            "8. Analyze dataset ticket"
        )

        print(
            "9. Exit"
        )

        choice = input(
            "\nChoose: "
        )

        if choice == "1":

            inspect_dataset()

        elif choice == "2":

            filter_dataset()

        elif choice == "3":

            text = input(
                "\nEnter ticket:\n"
            )

            analysis = (
                analyze_ticket(text)
            )

            print_ticket_analysis(
                analysis
            )

        elif choice == "4":

            tokenizer_menu()

        elif choice == "5":

            policy_menu()

        elif choice == "6":

            text = input(
                "\nCustomer message:\n"
            )

            response = (
                generate_support_response(
                    text
                )
            )

            print(
                "\nGENERATED RESPONSE:"
            )

            print(response)

        elif choice == "7":

            text = input(
                "\nMessage:\n"
            )

            print(
                "\nREMOTE RESPONSE:"
            )

            print(
                remote_chat(text)
            )

        elif choice == "8":

            index = int(
                input(
                    "Dataset row number: "
                )
            )

            ticket = (
                english_data[index]
            )

            text = safe_text(
                ticket["body"]
            )

            print(
                "\nSUBJECT:"
            )

            print(
                safe_text(
                    ticket[
                        "subject"
                    ]
                )
            )

            analysis = (
                analyze_ticket(text)
            )

            print_ticket_analysis(
                analysis
            )

        elif choice == "9":

            print(
                "Goodbye."
            )

            break

        else:

            print(
                "Invalid option."
            )


if __name__ == "__main__":
    main()