"""
Student Emotional AI Test Detector  ·  v4.1 (Fixed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes applied over v4.0:
  • Bug #1  – Emotion color no longer overwritten after lifestyle risk bump
  • Bug #2  – Slider values rounded before passing to inference
  • Bug #3  – Fixed mixed pack/grid geometry manager in history panel
  • Fix #4  – FocusOut placeholder restoration added
  • Fix #5  – Input length warning (>1000 chars)
  • Fix #6  – Expanded crisis keyword list

Install:
    pip install customtkinter transformers torch pillow

Run:
    python emotion_detector.py
"""

import threading
import datetime
import customtkinter as ctk

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Design tokens ─────────────────────────────────────────────────────────────
BG_APP     = "#0F172A"
BG_SIDEBAR = "#111827"
BG_CARD    = "#1E293B"
BG_INPUT   = "#0D1B2E"

A_BLUE     = "#3B82F6"
A_BLUE2    = "#1D4ED8"
A_PURPLE   = "#8B5CF6"

S_GREEN    = "#22C55E"
S_YELLOW   = "#F59E0B"
S_RED      = "#EF4444"
S_ORANGE   = "#F97316"

T_PRIMARY  = "#F1F5F9"
T_SEC      = "#94A3B8"
T_MUTED    = "#475569"
T_ACCENT   = "#93C5FD"

BORDER     = "#1E3A5F"

# ── Fonts ─────────────────────────────────────────────────────────────────────
def F(size, weight="normal"):
    return ctk.CTkFont(family="Segoe UI", size=size, weight=weight)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

EMOTION_MAP = {
    "joy":      ("Joy / Happy",         S_GREEN,   0),
    "neutral":  ("Neutral",             A_BLUE,    0),
    "surprise": ("Surprised",           A_PURPLE,  0),
    "sadness":  ("Sadness",             "#60A5FA", 1),
    "disgust":  ("Disgust",             S_ORANGE,  2),
    "anger":    ("Anger / Frustration", S_ORANGE,  2),
    "fear":     ("Fear / Anxiety",      S_YELLOW,  2),
}

EMOTION_ICONS = {
    "joy":      "😊",
    "neutral":  "😐",
    "surprise": "😲",
    "sadness":  "😔",
    "disgust":  "😤",
    "anger":    "😠",
    "fear":     "😰",
}

# ── FIX #6: Expanded crisis keyword list ──────────────────────────────────────
HIGH_RISK_WORDS = [
    "suicide", "kill myself", "end my life",
    "want to die", "i want to die", "kms",
    "end it all", "not want to be here",
    "self harm", "hurt myself", "cutting myself",
    "overdose", "take my life", "no reason to live",
    "better off dead", "can't go on",
]
MODERATE_WORDS = [
    "depressed", "overwhelmed", "hopeless",
    "can't cope", "breaking down", "panic",
    "worthless", "exhausted", "falling apart",
    "no one cares", "give up", "burned out",
]

STATUS_MAP = {
    0: ("✓  Normal",         S_GREEN),
    1: ("⚠  Low Risk",       "#60A5FA"),
    2: ("⚠  Moderate Risk",  S_YELLOW),
    3: ("🚨 High Risk",       S_RED),
}

# ─────────────────────────────────────────────────────────────────────────────
# CORE LOGIC (pure functions — fully unit-testable without GUI)
# ─────────────────────────────────────────────────────────────────────────────
def classify_emotion(model_label: str, score: float, text_lower: str):
    """
    Returns (display_name, emotion_color, status_text, risk_level, icon)
    emotion_color is always the emotion's own color — never the status color.
    Risk levels: 0=normal  1=low  2=moderate  3=high
    """
    if any(w in text_lower for w in HIGH_RISK_WORDS):
        return "Distress", S_RED, "🚨 High Risk", 3, "😰"

    name, color, base_risk = EMOTION_MAP.get(model_label, ("Unknown", T_SEC, 0))

    if any(w in text_lower for w in MODERATE_WORDS):
        base_risk = max(base_risk, 2)

    status_text, _ = STATUS_MAP[base_risk]
    icon = EMOTION_ICONS.get(model_label, "😐")
    return name, color, status_text, base_risk, icon


def compute_metrics(risk: int, score: float, sleep_hrs: float, study_hrs: float):
    """Compute stress/focus/anxiety/engagement (0-100)."""
    if risk == 3:
        base = dict(stress=92, focus=8,  anxiety=90, engagement=10)
    elif risk == 2:
        s = int(score * 78)
        base = dict(stress=s, focus=max(10, 68 - s // 2),
                    anxiety=int(score * 72), engagement=max(12, 55 - s // 3))
    elif risk == 1:
        s = int(score * 45)
        base = dict(stress=s, focus=max(40, 75 - s // 2),
                    anxiety=int(score * 40), engagement=max(40, 72 - s // 4))
    else:
        base = dict(stress=max(5, int((1 - score) * 22)),
                    focus=min(95, int(score * 92)),
                    anxiety=max(5, int((1 - score) * 18)),
                    engagement=min(95, int(score * 88)))

    if sleep_hrs < 4:
        base["stress"]     = min(100, base["stress"]     + 20)
        base["anxiety"]    = min(100, base["anxiety"]    + 18)
        base["focus"]      = max(0,   base["focus"]      - 25)
        base["engagement"] = max(0,   base["engagement"] - 20)
    elif sleep_hrs < 6:
        base["stress"]     = min(100, base["stress"]     + 10)
        base["anxiety"]    = min(100, base["anxiety"]    + 8)
        base["focus"]      = max(0,   base["focus"]      - 12)
        base["engagement"] = max(0,   base["engagement"] - 10)
    elif sleep_hrs >= 8:
        base["focus"]      = min(100, base["focus"]      + 5)
        base["engagement"] = min(100, base["engagement"] + 5)
        base["stress"]     = max(0,   base["stress"]     - 5)

    if study_hrs > 10:
        base["stress"]     = min(100, base["stress"]     + 15)
        base["anxiety"]    = min(100, base["anxiety"]    + 12)
        base["focus"]      = max(0,   base["focus"]      - 10)
    elif study_hrs > 8:
        base["stress"]     = min(100, base["stress"]     + 6)
    elif study_hrs < 2:
        base["engagement"] = max(0,   base["engagement"] - 15)
        base["focus"]      = max(0,   base["focus"]      - 8)

    return base


def build_alerts(risk: int, metrics: dict, sleep_hrs: float, study_hrs: float):
    """Build prioritised alert list including lifestyle factors."""
    alerts = []
    if risk == 3:
        alerts.append(("🚨", "Crisis keywords detected — escalate immediately.", S_RED))
    if metrics["stress"] > 75:
        alerts.append(("⚠", "High stress detected — consider counselling referral.", S_YELLOW))
    if metrics["anxiety"] > 68:
        alerts.append(("⚠", "Elevated anxiety — recommend a short break.", S_YELLOW))
    if metrics["focus"] < 30:
        alerts.append(("⚠", "Low focus — student may need academic support.", S_YELLOW))
    if sleep_hrs < 5:
        alerts.append(("🛌", f"Only {sleep_hrs:.0f}h sleep reported — affects cognition.", S_ORANGE))
    if study_hrs > 10:
        alerts.append(("📚", f"{study_hrs:.0f}h study reported — burnout risk, suggest breaks.", S_ORANGE))
    if study_hrs < 2 and risk < 2:
        alerts.append(("📖", "Low study hours — encourage consistent revision schedule.", A_BLUE))
    if not alerts:
        alerts.append(("✓", "No significant risk detected. Continue monitoring.", S_GREEN))
    return alerts


def contextual_risk_bump(base_risk: int, sleep_hrs: float, study_hrs: float) -> int:
    """Bump risk level up if lifestyle factors are severe."""
    if sleep_hrs < 4 or study_hrs > 12:
        return min(3, base_risk + 1)
    if sleep_hrs < 5.5 or study_hrs > 9:
        return min(2, max(base_risk, 1))
    return base_risk


# ─────────────────────────────────────────────────────────────────────────────
# WIDGET HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def mk_card(parent, **kw):
    d = dict(fg_color=BG_CARD, corner_radius=10,
             border_width=1, border_color=BORDER)
    d.update(kw)
    return ctk.CTkFrame(parent, **d)

def sep(parent, padx=0):
    ctk.CTkFrame(parent, fg_color=BORDER, height=1,
                 corner_radius=0).pack(fill="x", padx=padx)

def micro_lbl(parent, text, **kw):
    defaults = dict(font=F(9, "bold"), text_color=T_MUTED, anchor="w")
    defaults.update(kw)
    return ctk.CTkLabel(parent, text=text.upper(), **defaults)

def blend(hex_c, alpha):
    r,g,b = int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)
    br,bg,bb = 0x1E, 0x29, 0x3B
    return "#{:02x}{:02x}{:02x}".format(
        int(br+(r-br)*alpha), int(bg+(g-bg)*alpha), int(bb+(b-bb)*alpha))


class MiniMetric(ctk.CTkFrame):
    def __init__(self, parent, icon, title, color, **kw):
        super().__init__(parent, fg_color=BG_CARD, corner_radius=10,
                         border_width=1, border_color=BORDER, **kw)
        self._color = color
        inn = ctk.CTkFrame(self, fg_color="transparent")
        inn.pack(fill="both", expand=True, padx=10, pady=8)

        top = ctk.CTkFrame(inn, fg_color="transparent")
        top.pack(fill="x")

        pill = ctk.CTkFrame(top, fg_color=blend(color, 0.18),
                            corner_radius=6, width=26, height=26)
        pill.pack_propagate(False)
        pill.pack(side="left")
        ctk.CTkLabel(pill, text=icon, font=F(13)).pack(expand=True)

        ctk.CTkLabel(top, text=title, font=F(10),
                     text_color=T_SEC, anchor="w").pack(side="left", padx=(7,0))

        self._val = ctk.CTkLabel(top, text="—%", font=F(11,"bold"),
                                  text_color=color, anchor="e")
        self._val.pack(side="right")

        self._bar = ctk.CTkProgressBar(inn, height=4,
                                        progress_color=color,
                                        fg_color=BG_INPUT, corner_radius=2)
        self._bar.pack(fill="x", pady=(5,0))
        self._bar.set(0)

    def update(self, pct):
        self._val.configure(text=f"{pct}%")
        self._bar.set(min(pct,100) / 100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Student Emotional AI  ·  Detector  v4.1")
        self.geometry("1200x720")
        self.minsize(1080, 660)
        self.configure(fg_color=BG_APP)

        self._clf      = None
        self._history  = []

        self._build_topbar()
        self._build_body()

        self._set_status("loading")
        threading.Thread(target=self._load_model, daemon=True).start()

    # ── Model ─────────────────────────────────────────────────────────────────
    def _load_model(self):
        import os
        from transformers import pipeline
        
        # Check if we have the model saved completely offline
        local_path = "./local_model"
        if os.path.exists(local_path):
            print("Loading completely offline model...")
            self._clf = pipeline("text-classification", model=local_path, tokenizer=local_path)
        else:
            print("Loading model from Hugging Face cache...")
            self._clf = pipeline("text-classification", model=MODEL_NAME)
            
        self.after(0, lambda: self._set_status("ready"))

    def _set_status(self, state):
        cfg = {
            "loading": (S_YELLOW, "Loading AI Model…"),
            "ready":   (S_GREEN,  "Model Ready · 7 Emotions"),
            "busy":    (A_BLUE,   "Analyzing…"),
        }[state]
        self._dot.configure(fg_color=cfg[0])
        self._stxt.configure(text=cfg[1], text_color=cfg[0])
        self._btn_analyze.configure(
            state="disabled" if state != "ready" else "normal",
            text=("  ⚡  Analyze Emotion" if state == "ready"
                  else "  ⏳  Loading…"   if state == "loading"
                  else "  ⏳  Analyzing…")
        )

    # ── Topbar ────────────────────────────────────────────────────────────────
    def _build_topbar(self):
        bar = ctk.CTkFrame(self, fg_color="#0B1120", height=48, corner_radius=0)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        logo = ctk.CTkFrame(bar, fg_color="transparent")
        logo.pack(side="left", padx=16)
        ctk.CTkLabel(logo, text="🧠", font=F(18)).pack(side="left", padx=(0,7))
        ctk.CTkLabel(logo, text="Student Emotional AI",
                     font=F(14,"bold"), text_color=T_PRIMARY).pack(side="left")
        ctk.CTkLabel(logo, text=" · Test Detector",
                     font=F(12), text_color=T_SEC).pack(side="left")

        vbadge = ctk.CTkFrame(bar, fg_color=blend(A_PURPLE, 0.25),
                               corner_radius=6, border_width=1,
                               border_color=A_PURPLE)
        vbadge.pack(side="left", padx=10, pady=14)
        ctk.CTkLabel(vbadge, text="v4.1 · 7-Emotion AI",
                     font=F(9,"bold"), text_color=A_PURPLE).pack(padx=8, pady=2)

        pill = ctk.CTkFrame(bar, fg_color=BG_CARD, corner_radius=16,
                            border_width=1, border_color=BORDER)
        pill.pack(side="right", padx=16, pady=10)
        self._dot = ctk.CTkFrame(pill, fg_color=S_YELLOW,
                                  corner_radius=5, width=8, height=8)
        self._dot.pack_propagate(False)
        self._dot.pack(side="left", padx=(10,5), pady=9)
        self._stxt = ctk.CTkLabel(pill, text="Loading…",
                                   font=F(11), text_color=S_YELLOW)
        self._stxt.pack(side="left", padx=(0,12))

    # ── Body ──────────────────────────────────────────────────────────────────
    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color=BG_APP)
        body.pack(fill="both", expand=True, padx=12, pady=10)
        body.columnconfigure(0, weight=0, minsize=268)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        self._build_left_panel(body)
        self._build_right_panel(body)

    # ── Left panel ────────────────────────────────────────────────────────────
    def _build_left_panel(self, parent):
        sb = ctk.CTkFrame(parent, fg_color=BG_SIDEBAR, corner_radius=12,
                          border_width=1, border_color=BORDER, width=268)
        sb.grid(row=0, column=0, sticky="nsew", padx=(0,10))
        sb.pack_propagate(False)

        hdr = mk_card(sb)
        hdr.pack(fill="x", padx=11, pady=(11,0))
        hr = ctk.CTkFrame(hdr, fg_color="transparent")
        hr.pack(fill="x", padx=11, pady=9)
        ctk.CTkLabel(hr, text="🎓", font=F(17)).pack(side="left", padx=(0,7))
        col = ctk.CTkFrame(hr, fg_color="transparent")
        col.pack(side="left")
        ctk.CTkLabel(col, text="Student Emotion Input",
                     font=F(12,"bold"), text_color=T_PRIMARY, anchor="w").pack(anchor="w")
        ctk.CTkLabel(col, text="Text + context for analysis",
                     font=F(9), text_color=T_SEC, anchor="w").pack(anchor="w")

        # Text input
        ic = mk_card(sb)
        ic.pack(fill="x", padx=11, pady=(7,0))
        micro_lbl(ic, "Student Text / Response").pack(anchor="w", padx=11, pady=(9,3))
        self.text_area = ctk.CTkTextbox(
            ic, height=118, font=F(11),
            fg_color=BG_INPUT, border_color=BORDER, border_width=1,
            corner_radius=7, text_color=T_SEC, wrap="word",
            scrollbar_button_color=BG_CARD)
        self.text_area.pack(fill="x", padx=11, pady=(0,9))
        self._ph = "e.g. I feel really stressed about my exams and can't sleep…"
        self.text_area.insert("end", self._ph)
        # FIX #4: bind both FocusIn (clear) and FocusOut (restore)
        self.text_area.bind("<FocusIn>",  self._clear_ph)
        self.text_area.bind("<FocusOut>", self._restore_ph)

        # FIX #5: character count warning label
        self._char_warn = ctk.CTkLabel(ic, text="",
                                        font=F(9), text_color=S_YELLOW, anchor="e")
        self._char_warn.pack(anchor="e", padx=11, pady=(0,4))
        self.text_area.bind("<KeyRelease>", self._check_length)

        # Sleep slider
        sleep_card = mk_card(sb)
        sleep_card.pack(fill="x", padx=11, pady=(7,0))
        s_row = ctk.CTkFrame(sleep_card, fg_color="transparent")
        s_row.pack(fill="x", padx=11, pady=(9,2))
        ctk.CTkLabel(s_row, text="🛌  Sleep Hours (last night)",
                     font=F(11,"bold"), text_color=T_PRIMARY).pack(side="left")
        self._sleep_val_lbl = ctk.CTkLabel(s_row, text="7h",
                                            font=F(11,"bold"), text_color=S_GREEN)
        self._sleep_val_lbl.pack(side="right")
        self._sleep_slider = ctk.CTkSlider(
            sleep_card, from_=0, to=12, number_of_steps=24,
            progress_color=S_GREEN, button_color=S_GREEN,
            button_hover_color="#16A34A", fg_color=BG_INPUT,
            command=self._on_sleep_change)
        self._sleep_slider.set(7)
        self._sleep_slider.pack(fill="x", padx=11, pady=(0,4))
        self._sleep_hint = ctk.CTkLabel(sleep_card, text="✓ Good sleep",
                                         font=F(9), text_color=S_GREEN)
        self._sleep_hint.pack(anchor="w", padx=11, pady=(0,7))

        # Study slider
        study_card = mk_card(sb)
        study_card.pack(fill="x", padx=11, pady=(7,0))
        st_row = ctk.CTkFrame(study_card, fg_color="transparent")
        st_row.pack(fill="x", padx=11, pady=(9,2))
        ctk.CTkLabel(st_row, text="📚  Study Hours (today)",
                     font=F(11,"bold"), text_color=T_PRIMARY).pack(side="left")
        self._study_val_lbl = ctk.CTkLabel(st_row, text="4h",
                                            font=F(11,"bold"), text_color=A_BLUE)
        self._study_val_lbl.pack(side="right")
        self._study_slider = ctk.CTkSlider(
            study_card, from_=0, to=16, number_of_steps=32,
            progress_color=A_BLUE, button_color=A_BLUE,
            button_hover_color=A_BLUE2, fg_color=BG_INPUT,
            command=self._on_study_change)
        self._study_slider.set(4)
        self._study_slider.pack(fill="x", padx=11, pady=(0,4))
        self._study_hint = ctk.CTkLabel(study_card, text="✓ Healthy study load",
                                         font=F(9), text_color=A_BLUE)
        self._study_hint.pack(anchor="w", padx=11, pady=(0,7))

        ctk.CTkFrame(sb, fg_color=BORDER, height=1).pack(fill="x", padx=11, pady=(9,7))

        self._btn_analyze = ctk.CTkButton(
            sb, text="  ⚡  Analyze Emotion",
            command=self._analyze,
            font=F(12,"bold"), fg_color=A_BLUE, hover_color=A_BLUE2,
            text_color="#fff", height=40, corner_radius=10)
        self._btn_analyze.pack(fill="x", padx=11, pady=(0,6))

        ctk.CTkButton(
            sb, text="  ↺  Clear", command=self._clear,
            font=F(11), fg_color="transparent", hover_color=BG_CARD,
            text_color=T_SEC, border_width=1, border_color=BORDER,
            height=32, corner_radius=10).pack(fill="x", padx=11)

        ctk.CTkLabel(sb, text="AI-Based Student Emotional Risk Detection",
                     font=F(9), text_color=T_MUTED).pack(side="bottom", pady=8)

    # ── Slider callbacks ──────────────────────────────────────────────────────
    def _on_sleep_change(self, val):
        hrs = round(val * 2) / 2
        if hrs < 4:
            color, hint = S_RED,    "⚠ Severe sleep deprivation"
        elif hrs < 6:
            color, hint = S_YELLOW, "⚠ Poor sleep — affects focus"
        elif hrs < 7:
            color, hint = S_YELLOW, "△ Below optimal sleep"
        else:
            color, hint = S_GREEN,  "✓ Good sleep"
        self._sleep_val_lbl.configure(text=f"{hrs:.1f}h", text_color=color)
        self._sleep_hint.configure(text=hint, text_color=color)
        self._sleep_slider.configure(progress_color=color, button_color=color)

    def _on_study_change(self, val):
        hrs = round(val * 2) / 2
        if hrs > 12:
            color, hint = S_RED,    "⚠ Over-studying — burnout risk"
        elif hrs > 9:
            color, hint = S_YELLOW, "△ Heavy study load"
        elif hrs < 2:
            color, hint = T_SEC,    "△ Very low study time"
        else:
            color, hint = A_BLUE,   "✓ Healthy study load"
        self._study_val_lbl.configure(text=f"{hrs:.1f}h", text_color=color)
        self._study_hint.configure(text=hint, text_color=color)
        self._study_slider.configure(progress_color=color, button_color=color)

    # ── Right panel ───────────────────────────────────────────────────────────
    def _build_right_panel(self, parent):
        rp = ctk.CTkFrame(parent, fg_color=BG_APP)
        rp.grid(row=0, column=1, sticky="nsew")
        rp.rowconfigure(0, weight=0)
        rp.rowconfigure(1, weight=0)
        rp.rowconfigure(2, weight=1)
        rp.columnconfigure(0, weight=1)
        self._build_top_row(rp)
        self._build_metrics_row(rp)
        self._build_bottom_row(rp)

    # ── Row 0: Result + History ───────────────────────────────────────────────
    def _build_top_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color=BG_APP)
        row.grid(row=0, column=0, sticky="ew", pady=(0,7))
        row.columnconfigure(0, weight=3)
        row.columnconfigure(1, weight=2)

        rc = mk_card(row)
        rc.grid(row=0, column=0, sticky="nsew", padx=(0,7))
        rc.columnconfigure(1, weight=1)

        self._accent = ctk.CTkFrame(rc, fg_color=A_BLUE, width=4, corner_radius=2)
        self._accent.grid(row=0, column=0, sticky="ns", rowspan=2)

        id_f = ctk.CTkFrame(rc, fg_color="transparent")
        id_f.grid(row=0, column=1, sticky="ew", padx=(12,10), pady=(12,12))

        self._eicon = ctk.CTkLabel(id_f, text="😐", font=F(36))
        self._eicon.pack(side="left", padx=(0,12))

        info = ctk.CTkFrame(id_f, fg_color="transparent")
        info.pack(side="left")
        micro_lbl(info, "Detected Emotion").pack(anchor="w")
        self._ename = ctk.CTkLabel(info, text="Awaiting Analysis",
                                    font=F(18,"bold"), text_color=A_BLUE)
        self._ename.pack(anchor="w")
        self._estatus = ctk.CTkLabel(info, text="Enter text and click Analyze",
                                      font=F(10), text_color=T_SEC)
        self._estatus.pack(anchor="w")

        cf = mk_card(rc, fg_color=BG_INPUT, corner_radius=8)
        cf.grid(row=0, column=2, sticky="ns", padx=(0,12), pady=10)
        cf_i = ctk.CTkFrame(cf, fg_color="transparent")
        cf_i.pack(padx=12, pady=8)
        micro_lbl(cf_i, "Confidence").pack(anchor="center")
        self._conf_pct = ctk.CTkLabel(cf_i, text="—%",
                                       font=F(20,"bold"), text_color=A_BLUE)
        self._conf_pct.pack()
        self._conf_bar = ctk.CTkProgressBar(cf_i, width=82, height=5,
                                             progress_color=A_BLUE,
                                             fg_color=BORDER, corner_radius=2)
        self._conf_bar.pack(pady=(3,0))
        self._conf_bar.set(0)

        # History card
        hc = mk_card(row)
        hc.grid(row=0, column=1, sticky="nsew")
        hc_hdr = ctk.CTkFrame(hc, fg_color="transparent")
        hc_hdr.pack(fill="x", padx=11, pady=(9,5))
        ctk.CTkLabel(hc_hdr, text="📈  Session History",
                     font=F(11,"bold"), text_color=T_PRIMARY).pack(side="left")
        micro_lbl(hc_hdr, "Last 5").pack(side="right", pady=(2,0))
        sep(hc)
        self._history_box = ctk.CTkFrame(hc, fg_color="transparent")
        self._history_box.pack(fill="both", expand=True, padx=11, pady=(5,9))
        self._add_history_placeholder()

    # ── Row 1: Metric cards ───────────────────────────────────────────────────
    def _build_metrics_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color=BG_APP)
        row.grid(row=1, column=0, sticky="ew", pady=(0,7))
        for i in range(4):
            row.columnconfigure(i, weight=1)
        cfg = [
            ("😰", "Stress Level",  "stress",     S_RED),
            ("🎯", "Focus Level",   "focus",       A_BLUE),
            ("😟", "Anxiety Score", "anxiety",     S_YELLOW),
            ("💡", "Engagement",    "engagement",  S_GREEN),
        ]
        self._metrics = {}
        for i, (icon, title, key, color) in enumerate(cfg):
            mc = MiniMetric(row, icon, title, color)
            mc.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 0))
            self._metrics[key] = mc

    # ── Row 2: Keywords+Alerts | Analysis Output ──────────────────────────────
    def _build_bottom_row(self, parent):
        row = ctk.CTkFrame(parent, fg_color=BG_APP)
        row.grid(row=2, column=0, sticky="nsew")
        row.columnconfigure(0, weight=1)
        row.columnconfigure(1, weight=1)
        row.rowconfigure(0, weight=1)

        lc = ctk.CTkFrame(row, fg_color=BG_APP)
        lc.grid(row=0, column=0, sticky="nsew", padx=(0,6))
        lc.rowconfigure(0, weight=0)
        lc.rowconfigure(1, weight=1)
        lc.columnconfigure(0, weight=1)

        kc = mk_card(lc)
        kc.grid(row=0, column=0, sticky="ew", pady=(0,6))
        kc_h = ctk.CTkFrame(kc, fg_color="transparent")
        kc_h.pack(fill="x", padx=11, pady=(9,5))
        ctk.CTkLabel(kc_h, text="🔍  Keywords Detected",
                     font=F(11,"bold"), text_color=T_PRIMARY).pack(side="left")
        sep(kc)
        self._kw_lbl = ctk.CTkLabel(kc, text="Run analysis to detect keywords.",
                                     font=F(10), text_color=T_MUTED,
                                     wraplength=280, justify="left", anchor="w")
        self._kw_lbl.pack(anchor="w", padx=11, pady=(5,9))

        ac = mk_card(lc)
        ac.grid(row=1, column=0, sticky="nsew")
        ac_h = ctk.CTkFrame(ac, fg_color="transparent")
        ac_h.pack(fill="x", padx=11, pady=(9,5))
        ctk.CTkLabel(ac_h, text="🔔  Alerts & Recommendations",
                     font=F(11,"bold"), text_color=T_PRIMARY).pack(side="left")
        sep(ac)
        self._alerts_box = ctk.CTkFrame(ac, fg_color="transparent")
        self._alerts_box.pack(fill="both", expand=True, padx=11, pady=(5,9))
        self._add_alert("ℹ", "Run analysis to see alerts.", T_MUTED)

        oc = mk_card(row)
        oc.grid(row=0, column=1, sticky="nsew", padx=(6,0))
        oc.rowconfigure(2, weight=1)
        oc.columnconfigure(0, weight=1)

        oc_h = ctk.CTkFrame(oc, fg_color="transparent")
        oc_h.pack(fill="x", padx=11, pady=(9,5))
        ctk.CTkLabel(oc_h, text="📋  Analysis Output",
                     font=F(11,"bold"), text_color=T_PRIMARY).pack(side="left")
        micro_lbl(oc_h, "AI Detail").pack(side="right", pady=(2,0))
        sep(oc)

        meta = ctk.CTkFrame(oc, fg_color="transparent")
        meta.pack(fill="x", padx=11, pady=(7,4))

        def meta_row(p, key, attr, val_color=T_PRIMARY):
            r = ctk.CTkFrame(p, fg_color="transparent")
            r.pack(fill="x", pady=1)
            ctk.CTkLabel(r, text=key, font=F(10), text_color=T_SEC,
                         width=110, anchor="w").pack(side="left")
            lbl = ctk.CTkLabel(r, text="—", font=F(10),
                                text_color=val_color, anchor="w")
            lbl.pack(side="left")
            setattr(self, attr, lbl)

        meta_row(meta, "AI Model Label:", "_raw_lbl")
        meta_row(meta, "Risk Level:",     "_risk_lbl")
        meta_row(meta, "Sleep Factor:",   "_sleep_factor_lbl")
        meta_row(meta, "Study Factor:",   "_study_factor_lbl")
        sep(oc)

        self._out_box = ctk.CTkTextbox(
            oc, font=F(10),
            fg_color=BG_INPUT, border_color=BORDER, border_width=1,
            corner_radius=7, text_color=T_SEC, wrap="word",
            state="disabled", scrollbar_button_color=BG_CARD)
        self._out_box.pack(fill="both", expand=True, padx=11, pady=(7,11))

    # ── History helpers ───────────────────────────────────────────────────────
    def _add_history_placeholder(self):
        ctk.CTkLabel(self._history_box,
                     text="No analyses yet.\nRun your first detection.",
                     font=F(10), text_color=T_MUTED,
                     justify="center").pack(expand=True)

    def _refresh_history_panel(self):
        for w in self._history_box.winfo_children():
            w.destroy()
        if not self._history:
            self._add_history_placeholder()
            return
        for entry in reversed(self._history[-5:]):
            # FIX #3: use pack() exclusively — no mixing with grid()
            r = ctk.CTkFrame(self._history_box,
                              fg_color=BG_INPUT, corner_radius=6)
            r.pack(fill="x", pady=2)
            dot = ctk.CTkFrame(r, fg_color=entry["color"],
                                width=6, corner_radius=3)
            dot.pack_propagate(False)
            dot.pack(side="left", padx=(7,6), pady=6)
            ctk.CTkLabel(r, text=entry["emotion"],
                         font=F(10,"bold"), text_color=T_PRIMARY,
                         anchor="w").pack(side="left", fill="x", expand=True)
            ctk.CTkLabel(r, text=entry["time"],
                         font=F(9), text_color=T_MUTED,
                         anchor="e").pack(side="right", padx=(0,8))

    def _push_history(self, emotion, color, risk):
        self._history.append({
            "emotion": emotion,
            "color":   color,
            "risk":    risk,
            "time":    datetime.datetime.now().strftime("%H:%M"),
        })
        if len(self._history) > 5:
            self._history.pop(0)

    # ── Alert helper ──────────────────────────────────────────────────────────
    def _add_alert(self, icon, text, color):
        r = ctk.CTkFrame(self._alerts_box, fg_color="transparent")
        r.pack(fill="x", pady=2)
        ctk.CTkLabel(r, text=icon, font=F(12), text_color=color,
                     width=20).pack(side="left")
        ctk.CTkLabel(r, text=text, font=F(10), text_color=color,
                     anchor="w", justify="left", wraplength=270).pack(
                         side="left", padx=(5,0))

    # ── FIX #5: character length check ───────────────────────────────────────
    def _check_length(self, _=None):
        txt = self.text_area.get("1.0", "end").strip()
        if txt == self._ph:
            self._char_warn.configure(text="")
            return
        n = len(txt)
        if n > 1000:
            self._char_warn.configure(
                text=f"⚠ {n} chars — long text will be truncated to 512 tokens",
                text_color=S_YELLOW)
        elif n > 800:
            self._char_warn.configure(
                text=f"{n} chars", text_color=T_MUTED)
        else:
            self._char_warn.configure(text="")

    # ── ANALYZE ───────────────────────────────────────────────────────────────
    def _analyze(self):
        if not self._clf:
            return
        txt = self.text_area.get("1.0", "end").strip()
        if not txt or txt == self._ph:
            self._flash()
            return
        self._set_status("busy")
        # FIX #2: round slider values to 0.5 steps before inference
        sleep_hrs = round(self._sleep_slider.get() * 2) / 2
        study_hrs = round(self._study_slider.get() * 2) / 2
        threading.Thread(
            target=self._infer,
            args=(txt, sleep_hrs, study_hrs),
            daemon=True).start()

    def _infer(self, txt, sleep_hrs, study_hrs):
        tl     = txt.lower()
        result = self._clf(txt)[0]
        label  = result["label"]
        score  = result["score"]

        emotion, color, status, base_risk, icon = classify_emotion(label, score, tl)

        # FIX #1: after lifestyle bump, update status text only — do NOT
        # overwrite `color` so the emotion's own color is preserved
        final_risk = contextual_risk_bump(base_risk, sleep_hrs, study_hrs)
        if final_risk != base_risk:
            status = STATUS_MAP[final_risk][0]
            # color stays as the emotion's own color

        metrics  = compute_metrics(final_risk, score, sleep_hrs, study_hrs)
        detected = [w for w in HIGH_RISK_WORDS + MODERATE_WORDS if w in tl]
        alerts   = build_alerts(final_risk, metrics, sleep_hrs, study_hrs)
        summary  = self._make_summary(
            emotion, status, score, detected, final_risk, sleep_hrs, study_hrs)

        self.after(0, lambda: self._update_ui(
            emotion, color, status, icon, score,
            metrics, detected, alerts,
            label, final_risk, sleep_hrs, study_hrs, summary))

    def _update_ui(self, emotion, color, status, icon, score,
                   metrics, detected, alerts,
                   raw_label, risk, sleep_hrs, study_hrs, summary):
        self._eicon.configure(text=icon)
        self._ename.configure(text=emotion, text_color=color)
        self._estatus.configure(text=status, text_color=color)
        self._accent.configure(fg_color=color)
        self._conf_bar.configure(progress_color=color)
        self._conf_bar.set(score)
        self._conf_pct.configure(text=f"{score*100:.0f}%", text_color=color)

        for key, mc in self._metrics.items():
            mc.update(metrics[key])

        self._push_history(emotion, color, risk)
        self._refresh_history_panel()

        kw = "  •  ".join(detected) if detected else "None detected."
        self._kw_lbl.configure(
            text=kw, text_color=S_RED if detected else T_MUTED)

        for w in self._alerts_box.winfo_children():
            w.destroy()
        for ic, tx, col in alerts:
            self._add_alert(ic, tx, col)

        self._raw_lbl.configure(text=raw_label)
        rl_text  = {3:"High Risk", 2:"Moderate", 1:"Low Risk"}.get(risk,"Normal")
        rl_color = {3:S_RED, 2:S_YELLOW, 1:"#60A5FA"}.get(risk, S_GREEN)
        self._risk_lbl.configure(text=rl_text, text_color=rl_color)

        if sleep_hrs < 4:
            sf, sfc = f"{sleep_hrs:.1f}h — Severe deprivation", S_RED
        elif sleep_hrs < 6:
            sf, sfc = f"{sleep_hrs:.1f}h — Poor sleep", S_YELLOW
        else:
            sf, sfc = f"{sleep_hrs:.1f}h — OK", S_GREEN
        self._sleep_factor_lbl.configure(text=sf, text_color=sfc)

        if study_hrs > 10:
            stf, stfc = f"{study_hrs:.1f}h — Burnout risk", S_RED
        elif study_hrs > 8:
            stf, stfc = f"{study_hrs:.1f}h — Heavy load", S_YELLOW
        elif study_hrs < 2:
            stf, stfc = f"{study_hrs:.1f}h — Very low", T_SEC
        else:
            stf, stfc = f"{study_hrs:.1f}h — Healthy", S_GREEN
        self._study_factor_lbl.configure(text=stf, text_color=stfc)

        self._out_box.configure(state="normal")
        self._out_box.delete("1.0", "end")
        self._out_box.insert("end", summary)
        self._out_box.configure(state="disabled")

        self._set_status("ready")

    def _make_summary(self, emotion, status, score,
                      detected, risk, sleep_hrs, study_hrs):
        clean_status = status.replace("🚨 ","").replace("⚠  ","").replace("✓  ","")
        lines = [
            f"Emotion Detected  :  {emotion}",
            f"Risk Status       :  {clean_status}",
            f"AI Confidence     :  {score*100:.1f}%",
        ]
        if detected:
            lines.append(f"Risk Keywords     :  {', '.join(detected)}")
        lines.append(f"Sleep Context     :  {sleep_hrs:.1f} hrs")
        lines.append(f"Study Context     :  {study_hrs:.1f} hrs")
        lines.append("")
        if risk == 3:
            lines.append("⚠ Immediate professional intervention required.")
        elif risk == 2:
            lines.append("Student shows signs of emotional distress.\nConsider a wellbeing check-in.")
        elif risk == 1:
            lines.append("Low-level risk detected. Monitor closely.")
        else:
            lines.append("No immediate concern. Regular monitoring recommended.")
        return "\n".join(lines)

    # ── CLEAR ─────────────────────────────────────────────────────────────────
    def _clear(self):
        self.text_area.delete("1.0","end")
        self.text_area.insert("end", self._ph)
        self.text_area.configure(text_color=T_SEC)
        self._char_warn.configure(text="")

        self._sleep_slider.set(7)
        self._on_sleep_change(7)
        self._study_slider.set(4)
        self._on_study_change(4)

        self._eicon.configure(text="😐")
        self._ename.configure(text="Awaiting Analysis", text_color=A_BLUE)
        self._estatus.configure(text="Enter text and click Analyze", text_color=T_SEC)
        self._accent.configure(fg_color=A_BLUE)
        self._conf_bar.set(0)
        self._conf_bar.configure(progress_color=A_BLUE)
        self._conf_pct.configure(text="—%", text_color=A_BLUE)

        for mc in self._metrics.values():
            mc.update(0)

        for w in self._history_box.winfo_children():
            w.destroy()
        self._add_history_placeholder()
        self._history.clear()

        self._kw_lbl.configure(text="Run analysis to detect keywords.", text_color=T_MUTED)
        for w in self._alerts_box.winfo_children():
            w.destroy()
        self._add_alert("ℹ","Run analysis to see alerts.", T_MUTED)

        for attr in ("_raw_lbl","_risk_lbl","_sleep_factor_lbl","_study_factor_lbl"):
            getattr(self, attr).configure(text="—", text_color=T_PRIMARY)

        self._out_box.configure(state="normal")
        self._out_box.delete("1.0","end")
        self._out_box.configure(state="disabled")

    # ── Placeholder helpers ───────────────────────────────────────────────────
    def _clear_ph(self, _=None):
        if self.text_area.get("1.0","end").strip() == self._ph:
            self.text_area.delete("1.0","end")
            self.text_area.configure(text_color=T_PRIMARY)

    # FIX #4: restore placeholder when focus leaves an empty text area
    def _restore_ph(self, _=None):
        if not self.text_area.get("1.0","end").strip():
            self.text_area.insert("end", self._ph)
            self.text_area.configure(text_color=T_SEC)
            self._char_warn.configure(text="")

    def _flash(self):
        self.text_area.configure(border_color=S_RED)
        self.after(600, lambda: self.text_area.configure(border_color=BORDER))


# ── Entry ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()
