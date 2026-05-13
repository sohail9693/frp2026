import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Loading dataset...")
df = pd.read_csv("student_data.csv")

print("Initializing Hugging Face Emotion Model (this takes a moment)...")
from transformers import pipeline
clf = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

# Recreate the base risk mapping from the app
EMOTION_MAP = {
    "joy": 0, "neutral": 0, "surprise": 0,
    "sadness": 1,
    "disgust": 2, "anger": 2, "fear": 2,
}

MODERATE_WORDS = [
    "depressed", "overwhelmed", "hopeless",
    "can't cope", "breaking down", "panic",
    "worthless", "exhausted", "falling apart",
    "no one cares", "give up", "burned out",
]

HIGH_RISK_WORDS = [
    "suicide", "kill myself", "end my life",
    "want to die", "i want to die", "kms",
    "end it all", "not want to be here",
    "self harm", "hurt myself", "cutting myself",
    "overdose", "take my life", "no reason to live",
    "better off dead", "can't go on",
]

print("Running NLP inference on 500 student texts (Extracting features)...")
ai_base_risks = []
ai_confidences = []

for idx, row in df.iterrows():
    if idx % 100 == 0 and idx > 0:
        print(f"  Processed {idx}/500 rows...")
        
    text = str(row['Student_Text'])
    tl = text.lower()
    
    # 1. Check hard crisis override
    if any(w in tl for w in HIGH_RISK_WORDS):
        ai_base_risks.append(3)
        ai_confidences.append(1.0)
        continue
        
    # 2. Run model
    result = clf(text[:512])[0] # Truncate to avoid errors
    label = result['label']
    score = result['score']
    
    base_risk = EMOTION_MAP.get(label, 0)
    
    # 3. Check moderate bump
    if any(w in tl for w in MODERATE_WORDS):
        base_risk = max(base_risk, 2)
        
    ai_base_risks.append(base_risk)
    ai_confidences.append(score)

df['AI_Base_Risk'] = ai_base_risks
df['AI_Confidence'] = ai_confidences

print("\n--- Training Hybrid Model ---")
# Features: The emotion risk from the text + sleep + study context
X = df[['AI_Base_Risk', 'AI_Confidence', 'Sleep_Hours', 'Study_Hours']]
y = df['Actual_Risk_Level']

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"[*] Model Accuracy: {acc * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Generate Confusion Matrix Plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal (0)', 'Low (1)', 'Moderate (2)', 'High (3)'],
            yticklabels=['Normal (0)', 'Low (1)', 'Moderate (2)', 'High (3)'])
plt.title('Confusion Matrix: Predicted vs Actual Risk Level')
plt.xlabel('Predicted Risk Level')
plt.ylabel('Actual Risk Level')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
print("Saved 'confusion_matrix.png'")

# Generate Feature Importance Plot (Ablation Proof)
plt.figure(figsize=(8, 5))
importances = rf_model.feature_importances_
features = ['Text Emotion Risk', 'Text AI Confidence', 'Sleep Hours', 'Study Hours']
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importance (What drives the risk prediction?)')
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
print("Saved 'feature_importance.png'")

print("\n[*] Training Complete! Your friend can drag and drop these PNGs into their thesis.")

