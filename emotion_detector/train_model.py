import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from transformers import pipeline
import joblib

# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")
df = pd.read_csv("student_data.csv")

# =========================
# LOAD EMOTION MODEL
# =========================

print("Loading Emotion AI Model...")

clf = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# =========================
# EMOTION → RISK MAP
# =========================

EMOTION_MAP = {
    "joy": 0,
    "neutral": 0,
    "surprise": 0,
    "optimism": 0,

    "sadness": 1,
    "fear": 1,
    "nervousness": 1,

    "anger": 2,
    "disgust": 2,
}

# =========================
# CUSTOM KEYWORDS
# =========================

MODERATE_WORDS = [
    "depressed",
    "overwhelmed",
    "hopeless",
    "panic",
    "worthless",
    "burned out",
    "exhausted",
]

HIGH_RISK_WORDS = [
    "suicide",
    "kill myself",
    "want to die",
    "end my life",
    "self harm",
    "cut myself",
    "better off dead",
]

NEUTRAL_WORDS = [
    "salary",
    "meeting",
    "project",
    "assignment",
    "office",
]

# =========================
# FEATURE EXTRACTION
# =========================

print("Analyzing student texts...")

ai_base_risks = []
ai_confidences = []

for idx, row in df.iterrows():

    text = str(row["Student_Text"])
    tl = text.lower()

    # -------------------------
    # HIGH RISK OVERRIDE
    # -------------------------

    if any(word in tl for word in HIGH_RISK_WORDS):
        ai_base_risks.append(3)
        ai_confidences.append(1.0)
        continue

    # -------------------------
    # RUN TRANSFORMER MODEL
    # -------------------------

    results = clf(text[:512])

    best = max(results[0], key=lambda x: x["score"])

    label = best["label"].lower()
    score = best["score"]

    # -------------------------
    # LOW CONFIDENCE FIX
    # -------------------------

    if score < 0.75:
        label = "neutral"

    # -------------------------
    # NEUTRAL OVERRIDE
    # -------------------------

    if any(word in tl for word in NEUTRAL_WORDS):
        label = "neutral"

    # -------------------------
    # EMOTION → RISK
    # -------------------------

    base_risk = EMOTION_MAP.get(label, 0)

    # -------------------------
    # MODERATE WORD BOOST
    # -------------------------

    if any(word in tl for word in MODERATE_WORDS):
        base_risk = max(base_risk, 2)

    ai_base_risks.append(base_risk)
    ai_confidences.append(score)

# =========================
# ADD FEATURES
# =========================

df["AI_Base_Risk"] = ai_base_risks
df["AI_Confidence"] = ai_confidences

# =========================
# TRAIN MODEL
# =========================

print("\nTraining Hybrid AI Model...")

X = df[
    [
        "AI_Base_Risk",
        "AI_Confidence",
        "Sleep_Hours",
        "Study_Hours"
    ]
]

y = df["Actual_Risk_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

rf_model.fit(X_train, y_train)

# =========================
# PREDICTIONS
# =========================

y_pred = rf_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

joblib.dump(rf_model, "student_hybrid_model.pkl")

print("\nModel saved as student_hybrid_model.pkl")

# =========================
# CONFUSION MATRIX
# =========================

plt.figure(figsize=(8, 6))

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

print("Saved confusion_matrix.png")

# =========================
# FEATURE IMPORTANCE
# =========================

plt.figure(figsize=(8, 5))

importances = rf_model.feature_importances_

features = [
    "Emotion Risk",
    "AI Confidence",
    "Sleep Hours",
    "Study Hours"
]

sns.barplot(
    x=importances,
    y=features
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig("feature_importance.png")

print("Saved feature_importance.png")

print("\nHybrid AI Training Complete!")
