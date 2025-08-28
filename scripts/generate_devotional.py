import json
import os
import random
import datetime
import openai
from dotenv import load_dotenv
from dateutil import tz

# Load environment variables from .env file
load_dotenv()

# --- User Profile/Weights (can be fine-tuned based on your current situation) ---
USER_WEIGHTS = {
    "Family Responsibility and Care": 1.2,
    "Time Management and Wisdom": 1.2,
    "AI/Work Anxiety and Trust": 1.3,
    "Gratitude and Hope": 1.1,
    "Children's Education/Prayer for Children": 1.1,
    "Morning Seeking and Quietness": 1.2
}

# --- Theme to Scripture Mapping (covers themes from verses.json) ---
THEME_MAP = {
    "Morning Seeking and Quietness": ["Psalm 63:1-8", "Psalm 62:1-2,5-8"],
    "AI/Work Anxiety and Trust": ["Psalm 46", "Psalm 37:3-7", "Matthew 6:25-34", "Philippians 4:6-8"],
    "Time Management and Wisdom": ["Psalm 90:12", "Proverbs 3:5-6", "James 1:5"],
    "Gratitude and Hope": ["Psalm 103:1-5", "Isaiah 40:28-31"],
    "Family Responsibility and Care": ["Psalm 91", "Psalm 121", "Psalm 23"],
    "Children's Education/Prayer for Children": ["Psalm 121", "Proverbs 3:5-6"]
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
    weights = USER_WEIGHTS.copy()
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
    candidates = THEME_MAP.get(theme, [])
    if not candidates:
        return ["Psalm 23"]
    k = 2 if len(candidates) > 1 and random.random() < 0.35 else 1
    return random.sample(candidates, k)

def generate_devotional_with_ai(theme, refs, dt_local):
    """Generates devotional content using OpenAI API."""
    
    scripture_references = ", ".join(refs)
    
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
