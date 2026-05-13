from transformers import pipeline

# =========================
# LOAD MODEL
# =========================

print("Loading Emotion AI Model...")

clf = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

# =========================
# WORD LISTS
# =========================

HIGH_RISK_WORDS = [
    "suicide",
    "kill myself",
    "want to die",
    "end my life",
    "self harm",
    "cut myself",
    "better off dead",
]

MODERATE_WORDS = [
    "depressed",
    "overwhelmed",
    "hopeless",
    "panic",
    "worthless",
    "burned out",
    "exhausted",
]

NEUTRAL_WORDS = [
    "salary",
    "payment",
    "meeting",
    "project",
    "assignment",
    "office",
]

# =========================
# MAIN ANALYSIS FUNCTION
# =========================

def analyze_emotion(text, sleep_hours, study_hours):

    text = str(text)
    tl = text.lower()

    # =====================
    # HIGH RISK OVERRIDE
    # =====================

    if any(word in tl for word in HIGH_RISK_WORDS):

        return {
            "emotion": "High Risk",
            "risk": "HIGH",
            "confidence": 100,
            "stress": 95,
            "anxiety": 98,
            "focus": 10,
            "engagement": 5,
            "label": "crisis"
        }

    # =====================
    # RUN TRANSFORMER MODEL
    # =====================

    result = clf(text[:512])[0]

    label = result["label"].lower()
    confidence = round(result["score"] * 100, 2)

    # =====================
    # LOW CONFIDENCE FIX
    # =====================

    if confidence < 75:
        label = "neutral"

    # =====================
    # NEUTRAL OVERRIDE
    # =====================

    if any(word in tl for word in NEUTRAL_WORDS):
        label = "neutral"

    # =====================
    # MODERATE WORD BOOST
    # =====================

    moderate_detected = any(word in tl for word in MODERATE_WORDS)

    # =====================
    # FINAL LOGIC
    # =====================

    emotion = "Neutral"
    risk = "Normal"

    stress = 10
    anxiety = 10
    focus = 70
    engagement = 75

    # ---------------------
    # HAPPY
    # ---------------------

    if label in ["joy", "optimism"]:

        emotion = "Joy / Happy"
        risk = "Normal"

        stress = 5
        anxiety = 5
        focus = 80
        engagement = 85

    # ---------------------
    # SAD
    # ---------------------

    elif label == "sadness":

        emotion = "Sadness"
        risk = "Low"

        stress = 40
        anxiety = 35
        focus = 45
        engagement = 40

    # ---------------------
    # FEAR / NERVOUS
    # ---------------------

    elif label in ["fear", "nervousness"]:

        emotion = "Anxiety"
        risk = "Moderate"

        stress = 70
        anxiety = 80
        focus = 30
        engagement = 40

    # ---------------------
    # ANGER
    # ---------------------

    elif label in ["anger", "disgust"]:

        emotion = "Anger / Frustration"
        risk = "Moderate"

        stress = 75
        anxiety = 60
        focus = 35
        engagement = 30

    # ---------------------
    # NEUTRAL
    # ---------------------

    elif label == "neutral":

        emotion = "Neutral"
        risk = "Normal"

        stress = 10
        anxiety = 10
        focus = 75
        engagement = 70

    # =====================
    # MODERATE BOOST
    # =====================

    if moderate_detected:

        risk = "Moderate"

        stress += 25
        anxiety += 25

        focus -= 15
        engagement -= 10

    # =====================
    # SLEEP FACTOR
    # =====================

    if sleep_hours < 5:

        stress += 15
        anxiety += 15

        focus -= 10

    # =====================
    # STUDY FACTOR
    # =====================

    if study_hours > 10:

        stress += 10
        anxiety += 10

    # =====================
    # LIMIT VALUES
    # =====================

    stress = max(0, min(stress, 100))
    anxiety = max(0, min(anxiety, 100))
    focus = max(0, min(focus, 100))
    engagement = max(0, min(engagement, 100))

    # =====================
    # RETURN RESULT
    # =====================

    return {

        "emotion": emotion,
        "risk": risk,
        "confidence": confidence,

        "stress": stress,
        "anxiety": anxiety,
        "focus": focus,
        "engagement": engagement,

        "label": label
    }


# =========================
# TEST
# =========================

text = input("Enter student text: ")

result = analyze_emotion(
    text=text,
    sleep_hours=7,
    study_hours=4
)

print("\n===== RESULT =====")

for k, v in result.items():
    print(f"{k}: {v}")
