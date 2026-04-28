# evaluate_model.py
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---- Load model ----
MODEL_NAME = "nateraw/bert-base-uncased-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
id2label = model.config.id2label

# ---- Load test data ----
df = pd.read_csv("data/test_emotions.csv")

def predict_emotion(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        return id2label[pred].lower()

# ---- Run predictions ----
df["predicted"] = df["text"].apply(predict_emotion)

# ---- Evaluation ----
print("\nClassification Report:\n")
print(classification_report(df["label"], df["predicted"]))

accuracy = accuracy_score(df["label"], df["predicted"])
precision = precision_score(df["label"], df["predicted"], average="weighted")
recall = recall_score(df["label"], df["predicted"], average="weighted")
f1 = f1_score(df["label"], df["predicted"], average="weighted")

print("\nOverall Metrics:")
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

# ---- Confusion Matrix ----
cm = confusion_matrix(df["label"], df["predicted"], labels=sorted(df["label"].unique()))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=sorted(df["label"].unique()), yticklabels=sorted(df["label"].unique()))
plt.title("Emotion Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
