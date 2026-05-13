import csv
import random

# ── Templates for different risk levels to ensure diversity ──

# RISK 0: Normal / Positive / Neutral
text_risk_0 = [
    "Feeling pretty good about the upcoming project.",
    "Just finished my assignment, going to hang out with friends.",
    "The lecture today was actually really interesting.",
    "Had a great weekend! Ready for the week.",
    "Everything is going fine, just regular coursework.",
    "I'm feeling confident about the midterms.",
    "Got an A on my last quiz, so happy!",
    "Just a normal day, nothing special to report.",
    "Studying with my group was really productive.",
    "I finally understand the calculus material.",
    "Looking forward to the winter break.",
    "Campus life is pretty chill right now."
]

# RISK 1: Low Risk (Mild stress, slight burnout, tiredness)
text_risk_1 = [
    "Feeling a bit tired today, lots of reading to do.",
    "The upcoming exam is stressing me out a little.",
    "I have so much homework, it's annoying.",
    "Wish I had more free time, feeling slightly drained.",
    "Not my best day, just feeling sluggish.",
    "A little worried about my grade in physics.",
    "Group projects are so frustrating when people don't contribute.",
    "I need coffee, I barely slept last night studying.",
    "Just trying to keep up with the syllabus.",
    "Feeling a little behind, need to catch up this weekend.",
    "The workload is starting to pick up."
]

# RISK 2: Moderate Risk (High anxiety, significant stress, overwhelmed)
text_risk_2 = [
    "I am completely overwhelmed by the amount of work I have.",
    "I feel like I'm failing no matter how hard I study.",
    "My anxiety is through the roof because of finals.",
    "I am so exhausted, I can't even focus anymore.",
    "I feel like breaking down, everything is piling up.",
    "I can't cope with this pressure from my professors and parents.",
    "I'm starting to panic, there's no way I can finish this in time.",
    "I feel totally hopeless about my GPA.",
    "Every day is a struggle right now, I'm just so burned out.",
    "I haven't been sleeping well because I keep thinking about failing.",
    "I feel so isolated and stressed, it's suffocating."
]

# RISK 3: High Risk (Crisis keywords, severe depression)
text_risk_3 = [
    "I honestly just want to die, I can't take this anymore.",
    "Sometimes I think it would be easier to just end it all.",
    "I'm feeling so depressed that I have no reason to live.",
    "I feel like hurting myself, the pressure is too much.",
    "There's no point anymore. I am better off dead.",
    "I'm thinking about cutting myself again.",
    "I just want to take my life, everything is pointless.",
    "I can't go on like this. I want to end my life.",
    "I'm having serious thoughts about suicide.",
    "I feel like taking an overdose, I just want it to stop."
]

def generate_row():
    # Randomly select a true risk level distribution
    # Let's make it realistic: 50% normal, 30% low, 15% moderate, 5% high
    rand = random.random()
    
    if rand < 0.50:
        risk = 0
        text = random.choice(text_risk_0)
        # Healthy habits
        sleep = round(random.uniform(6.5, 9.5) * 2) / 2
        study = round(random.uniform(2.0, 6.0) * 2) / 2
        
    elif rand < 0.80:
        risk = 1
        text = random.choice(text_risk_1)
        # Slightly worse habits
        sleep = round(random.uniform(4.5, 7.0) * 2) / 2
        study = round(random.uniform(4.0, 9.0) * 2) / 2
        
    elif rand < 0.95:
        risk = 2
        text = random.choice(text_risk_2)
        # Poor habits (High study or low sleep)
        sleep = round(random.uniform(2.5, 5.5) * 2) / 2
        study = round(random.uniform(8.0, 14.0) * 2) / 2
        
    else:
        risk = 3
        text = random.choice(text_risk_3)
        # Crisis habits (Usually severe deprivation, but can be anything)
        sleep = round(random.uniform(1.0, 4.5) * 2) / 2
        study = round(random.uniform(0.0, 15.0) * 2) / 2

    # Add slight random noise to text to increase diversity
    prefix = ["", "Honestly, ", "Right now ", "Lately ", "To be honest, ", "I don't know, "]
    suffix = ["", "...", " :(", " man.", " just saying.", " ugh."]
    
    if random.random() > 0.5:
        text = random.choice(prefix) + text.lower() if random.choice(prefix) != "" else text
    if random.random() > 0.5:
        text = text.rstrip('.') + random.choice(suffix)
        
    # Capitalize first letter
    text = text[0].upper() + text[1:] if text else text

    return [text, sleep, study, risk]

def main():
    filename = "student_data.csv"
    num_rows = 500
    
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Student_Text", "Sleep_Hours", "Study_Hours", "Actual_Risk_Level"])
        
        for _ in range(num_rows):
            writer.writerow(generate_row())
            
    print(f"✅ Generated {num_rows} highly diverse rows in '{filename}'!")

if __name__ == "__main__":
    main()
