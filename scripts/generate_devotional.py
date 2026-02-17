import json
import os
import random
import datetime
import openai
from dotenv import load_dotenv
from dateutil import tz

# Load environment variables from .env file
load_dotenv()

# DONE: I want to add following content
# - 加入希望蒙神保守祝福个人和家庭健康幸福, 以及希望才干能够被神使用的祷告内容
# - 加入希望小儿子变得听话靠谱的祷告
# - 加入求主赐予智慧来管理时间, 在人工智能的时代, 做出智慧的判断, 的祷告内容

# DONE: the previous change should be part of theme, not global. And you should add corresponding related bible verse for them

# --- Unified theme configuration to avoid redundant theme data ---
THEME_CONFIG = {
    "Morning Seeking and Quietness": {
        "weight": 1.2,
        "scriptures": [
            "Psalm 63:1-8",
            "Psalm 62:1-2,5-8",
            "Psalm 5:1-3",
            "Mark 1:35",
            "Lamentations 3:22-26",
        ],
        "prayer_focus": "",
    },
    "AI/Work Anxiety and Trust": {
        "weight": 1.3,
        "scriptures": [
            "Psalm 46",
            "Psalm 37:3-7",
            "Matthew 6:25-34",
            "Philippians 4:6-8",
            "1 Peter 5:6-7",
            "Proverbs 16:3",
            "Psalm 55:22",
        ],
        "prayer_focus": "",
    },
    "Time Management and Wisdom": {
        "weight": 1.2,
        "scriptures": [
            "Psalm 90:12",
            "Proverbs 3:5-6",
            "James 1:5",
            "Ephesians 5:15-17",
            "Colossians 4:5",
            "Proverbs 16:9",
            "Philippians 1:9-10",
        ],
        "prayer_focus": (
            "- Include prayer asking the Lord for wisdom to manage time and to make wise judgments "
            "in the age of artificial intelligence."
        ),
    },
    "Gratitude and Hope": {
        "weight": 1.1,
        "scriptures": [
            "Psalm 103:1-5",
            "Isaiah 40:28-31",
            "1 Thessalonians 5:16-18",
            "Romans 15:13",
            "Lamentations 3:21-23",
        ],
        "prayer_focus": "",
    },
    "Family Responsibility and Care": {
        "weight": 1.2,
        "scriptures": [
            "Psalm 91",
            "Psalm 121",
            "Psalm 23",
            "Joshua 24:15",
            "1 Timothy 5:8",
            "Proverbs 22:6",
            "1 Peter 4:10",
        ],
        "prayer_focus": (
            "- Include prayer for God's protection and blessing over personal and family health and happiness, "
            "and that my talents may be used by God."
        ),
    },
    "Children's Education/Prayer for Children": {
        "weight": 1.1,
        "scriptures": [
            "Psalm 121",
            "Proverbs 3:5-6",
            "Deuteronomy 6:6-7",
            "Isaiah 54:13",
            "Psalm 127:3-5",
            "Ephesians 6:1-4",
        ],
        "prayer_focus": "- Include prayer for my younger son to become obedient and dependable.",
    },
    "Emotional Self-Control and Patience": {
        "weight": 1.2,
        "scriptures": [
            "Ephesians 4:26-32",
            "Proverbs 15:1",
            "Proverbs 16:32",
            "James 1:19-20",
            "Colossians 3:12-14",
            "Galatians 5:22-23",
            "Proverbs 29:11",
        ],
        "prayer_focus": "",
    },
    "Seek for Wisdom from God": {
        "weight": 1.2,
        "scriptures": [
            "James 1:5",
            "Proverbs 2:1-6",
            "Proverbs 3:5-6",
            "Ephesians 1:17",
            "Colossians 1:9",
        ],
        "prayer_focus": "",
    },
    "Seek for God's Blessing": {
        "weight": 1.1,
        "scriptures": [
            "Numbers 6:24-26",
            "Psalm 67:1-2",
            "Ephesians 1:3",
            "Deuteronomy 28:2",
            "Psalm 115:12-15",
        ],
        "prayer_focus": "",
    },
}

def weekday_bias(dt_local):
    """Applies a light bias for weekdays vs. weekends."""
    wd = dt_local.weekday()  # Mon=0 ... Sun=6
    bias = {}
    if wd <= 4:  # Weekday
        bias["AI/Work Anxiety and Trust"] = 1.2
        bias["Time Management and Wisdom"] = 1.15
    else:  # Weekend
        bias["Family Responsibility and Care"] = 1.15
        bias["Gratitude and Hope"] = 1.1
    # Constant preference for morning quiet time
    bias["Morning Seeking and Quietness"] = 1.1
    return bias

def pick_theme(dt_local):
    """Picks a theme based on weights and biases."""
    weights = {theme: cfg.get("weight", 1.0) for theme, cfg in THEME_CONFIG.items()}
    for k, v in weekday_bias(dt_local).items():
        weights[k] = weights.get(k, 1.0) * v
    
    items = list(weights.items())
    total = sum(w for _, w in items)
    r = random.uniform(0, total)
    upto = 0.0
    for theme, w in items:
        if upto + w >= r:
            return theme
        upto += w
    return items[-1][0]

def pick_scriptures(theme):
    """Picks 1 or 2 scripture references for the given theme."""
    candidates = THEME_CONFIG.get(theme, {}).get("scriptures", [])
    if not candidates:
        return ["Psalm 23"]
    k = 2 if len(candidates) > 1 and random.random() < 0.35 else 1
    return random.sample(candidates, k)

def generate_devotional_with_ai(theme, refs, dt_local):
    """Generates devotional content using OpenAI API."""
    
    scripture_references = ", ".join(refs)
    theme_prayer_focus = THEME_CONFIG.get(theme, {}).get("prayer_focus")
    prayer_focus_instructions = ""
    if theme_prayer_focus:
        prayer_focus_instructions = (
            "\nPrayer section must explicitly include this theme-specific focus:\n"
            f"{theme_prayer_focus}\n"
        )
    
    prompt = f"""
You are a wise and compassionate Chinese theologian and pastor. 
Generate a daily devotional in Markdown format, all the text must using Simplified Chinese.

Today's Date: {dt_local.strftime('%Y-%m-%d')}
Theme: {theme}
Scripture References: {scripture_references}

Please provide the following sections in your response:
1.  **Full Scripture Text**: Provide the full text for the scripture reference(s) above. Use a well-regarded English translation (like NIV or ESV).
2.  **Interpretation**: Explain the meaning and context of these verses. What is the main message?
3.  **Application**: How can I apply this to my daily life? Make it practical and connect it to the theme of '{theme}'.
4.  **Prayer**: Write a short, heartfelt prayer based on the scripture and application.
{prayer_focus_instructions}

Structure the output clearly with Markdown headings.
"""

    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a wise and compassionate theologian and pastor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"**Error**: Could not generate devotional content due to an API error.\n\n**Theme**: {theme}\n**Scriptures**: {', '.join(refs)}"


def main():
    # Set timezone to America/Los_Angeles to align with "morning devotion"
    la_tz = tz.gettz("America/Los_Angeles")
    now_utc = datetime.datetime.utcnow().replace(tzinfo=tz.UTC)
    now_local = now_utc.astimezone(la_tz)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "YOUR_OPENAI_API_KEY":
        print("Error: OPENAI_API_KEY is not set. Please set it in your .env file.")
        return

    theme = pick_theme(now_local)
    refs = pick_scriptures(theme)

    # --- Output File Setup ---
    out_dir = "daily"
    os.makedirs(out_dir, exist_ok=True)
    date_str = now_local.strftime("%Y-%m-%d")
    out_path = os.path.join(out_dir, f"{date_str}.md")

    # If file already exists, don't regenerate to avoid duplicate commits
    if os.path.exists(out_path):
        print(f"{out_path} already exists. Skip writing.")
        return

    print(f"Generating devotional for {date_str} with theme '{theme}' and scriptures '{', '.join(refs)}'...")
    
    # --- Generate Content ---
    title = f"# Daily Scripture and Devotion · {date_str}"
    body = generate_devotional_with_ai(theme, refs, now_local)
    generation_time = f"\n_Generated on: {now_local.strftime('%Y-%m-%d %H:%M %Z')}_"
    content = f"{title}\n\n{body}\n{generation_time}\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Successfully wrote devotional to {out_path}")

if __name__ == "__main__":
    main()
