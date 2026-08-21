import sys
import argparse
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_PROCESSED, TEST_FILENAME


# Email categories for classification
EMAIL_CATEGORIES = [
    "Action Required",
    "Informational", 
    "Meeting",
    "Question"
]


def categorize_email(email_text: str) -> str:
    """
    Categorize an email based on keywords and content analysis.
    Returns one of: Action Required, Informational, Meeting, Question
    """
    email_lower = email_text.lower()
    
    # Priority action phrases
    action_phrases = [
        "please complete", "please submit", "please upload", "please send",
        "please review", "please provide", "upload the final", "complete these changes",
        "action required", "urgent", "required", "immediately", "deadline", "due by",
        "by end of day", "asap", "due tomorrow", "before tomorrow"
    ]
    if any(phrase in email_lower for phrase in action_phrases):
        return "Action Required"
        
    # Priority prefix checks for explicit indicators
    if any(email_lower.startswith(prefix) for prefix in ["action required", "urgent", "required"]):
        return "Action Required"
    if any(email_lower.startswith(prefix) for prefix in ["fyi", "announcement", "update", "reminder", "informational"]):
        return "Informational"
    if any(email_lower.startswith(prefix) for prefix in ["meeting", "call"]):
        return "Meeting"
    
    # Keywords for each category (with weighted importance)
    action_keywords = {
        "required": 5, "urgent": 5, "submit": 4, "action": 5, "asap": 4,
        "deadline": 4, "due": 4, "complete": 4, "finish": 3, "review": 3,
        "approval": 4, "approve": 4, "fix": 4, "immediately": 4, "upload": 4
    }
    
    meeting_keywords = {
        "meeting": 5, "schedule": 4, "call": 4, "conference": 4,
        "available": 3, "discuss": 2, "2pm": 3, "join": 3, "zoom": 3
    }
    
    question_keywords = {
        "question": 5, "how": 4, "what": 4, "why": 3, "explain": 4,
        "clarify": 4, "help": 3, "can you": 4, "could you": 4, "?": 3
    }
    
    info_keywords = {
        "fyi": 5, "announcement": 5, "update": 3, "informational": 5,
        "reminder": 3, "published": 3, "closed": 3, "holiday": 3
    }
    
    # Calculate scores
    scores = {
        "Action Required": sum(weight for kw, weight in action_keywords.items() if kw in email_lower),
        "Meeting": sum(weight for kw, weight in meeting_keywords.items() if kw in email_lower),
        "Question": sum(weight for kw, weight in question_keywords.items() if kw in email_lower),
        "Informational": sum(weight for kw, weight in info_keywords.items() if kw in email_lower),
    }
    
    # Return category with highest score, default to Informational
    if max(scores.values()) == 0:
        return "Informational"
    return max(scores, key=scores.get)


def evaluate_model(data_path: Path, max_samples: int = 200) -> dict:
    """Evaluate email classification model."""
    
    test_df = pd.read_csv(data_path).dropna(subset=["incoming_email"])
    if max_samples:
        test_df = test_df.head(max_samples)

    if len(test_df) == 0:
        raise ValueError(f"No samples found in {data_path}.")

    # Load ML classifier model if exists
    model_path = Path("models/email_classifier.pkl")
    classifier_model = None
    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                classifier_model = pickle.load(f)
            print(f"Loaded trained ML model from {model_path}")
        except Exception as e:
            print(f"Could not load ML model: {e}")

    # Generate predictions and ground truth
    predictions = []
    ground_truth = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Categorizing emails"):
        email_text = str(row["incoming_email"])
        
        # Predict category
        if classifier_model is not None:
            predicted_category = classifier_model.predict([email_text])[0]
        else:
            predicted_category = categorize_email(email_text)
            
        predictions.append(predicted_category)
        
        # Ground truth: use explicit category column if available, else categorize incoming email
        if "category" in row and pd.notna(row["category"]) and str(row["category"]).strip() != "":
            ground_truth.append(str(row["category"]).strip())
        else:
            ground_truth.append(categorize_email(email_text))
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, labels=EMAIL_CATEGORIES, average='weighted', zero_division=0)
    recall = recall_score(ground_truth, predictions, labels=EMAIL_CATEGORIES, average='weighted', zero_division=0)
    f1 = f1_score(ground_truth, predictions, labels=EMAIL_CATEGORIES, average='weighted', zero_division=0)
    
    # Generate classification report
    class_report = classification_report(
        ground_truth, 
        predictions,
        labels=EMAIL_CATEGORIES,
        zero_division=0,
        digits=2
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "classification_report": class_report,
        "num_samples": len(test_df),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate email classification model.")
    parser.add_argument("--data", type=Path, default=DATA_PROCESSED / TEST_FILENAME)
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Data not found at {args.data}.")

    results = evaluate_model(args.data, args.max_samples)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    print(f"\nAccuracy : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)")
    print(f"Recall   : {results['recall']:.4f} ({results['recall']*100:.2f}%)")
    print(f"F1 Score : {results['f1']:.4f} ({results['f1']*100:.2f}%)")
    
    print(f"\nClassification Report:")
    print()
    print(results['classification_report'])


if __name__ == "__main__":
    main()
