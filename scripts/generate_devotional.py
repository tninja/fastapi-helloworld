import json, os, random, datetime
from dateutil import tz

# -------- 用户画像/权重（结合你的近况，可继续微调） --------
USER_WEIGHTS = {
    "家庭责任与看顾": 1.2,
    "时间管理与智慧": 1.2,
    "AI/工作焦虑与交托": 1.3,
    "感恩与盼望": 1.1,
    "孩子教育/为儿女祷告": 1.1,
    "清晨寻求与安静": 1.2
}

# 主题到经文的粗映射（覆盖 verses.json 中的主题）
THEME_MAP = {
    "清晨寻求与安静": ["诗篇 63:1-8", "诗篇 62:1-2,5-8"],
    "AI/工作焦虑与交托": ["诗篇 46", "诗篇 37:3-7", "马太福音 6:25-34", "腓立比书 4:6-8"],
    "时间管理与智慧": ["诗篇 90:12", "箴言 3:5-6", "雅各书 1:5"],
    "感恩与盼望": ["诗篇 103:1-5", "以赛亚书 40:28-31"],
    "家庭责任与看顾": ["诗篇 91", "诗篇 121", "诗篇 23"],
    "孩子教育/为儿女祷告": ["诗篇 121", "箴言 3:5-6"]
}

# 工作日/周末可做轻量偏好（工作日偏“交托/智慧”，周末偏“家庭/感恩”）
def weekday_bias(dt_local):
    wd = dt_local.weekday()  # Mon=0 ... Sun=6
    bias = {}
    if wd <= 4:  # 工作日
        bias["AI/工作焦虑与交托"] = 1.2
        bias["时间管理与智慧"] = 1.15
    else:  # 周末
        bias["家庭责任与看顾"] = 1.15
        bias["感恩与盼望"] = 1.1
    # 清晨恒定偏好
    bias["清晨寻求与安静"] = 1.1
    return bias

def load_verses():
    with open("data/verses.json", "r", encoding="utf-8") as f:
        return json.load(f)

def pick_theme(dt_local):
    weights = USER_WEIGHTS.copy()
    for k, v in weekday_bias(dt_local).items():
        weights[k] = weights.get(k, 1.0) * v
    # 归一化 + 加一点随机性
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
    candidates = THEME_MAP.get(theme, [])
    if not candidates:
        return ["诗篇 23"]
    # 从候选中选择 1-2 段
    k = 2 if len(candidates) > 1 and random.random() < 0.35 else 1
    return random.sample(candidates, k)

def build_reflection(theme, refs, dt_local):
    # 面向“你”的简短解读模板（可继续深化）
    lines = []
    lines.append(f"**主题**：{theme}")
    tip = {
        "清晨寻求与安静": "把今天最先的注意力交给主，让心先安静，再投入工作与育儿的忙碌。",
        "AI/工作焦虑与交托": "行业变化很快，但主不改变。把不确定与焦虑放在祷告里，以行动回应、以交托止息。",
        "时间管理与智慧": "数算日子、定优先级：先重要、再紧急；愿主赐下作抉择的智慧与勇气。",
        "感恩与盼望": "从数算恩典开始新的一天，感恩能让心从匮乏转向丰盛与盼望。",
        "家庭责任与看顾": "为妻儿与家人的身心灵祷告，相信神的看顾超越我们能照料的边界。",
        "孩子教育/为儿女祷告": "不单追求成绩与技能，求主塑造孩子的品格、怜悯与刚强。"
    }.get(theme, "")
    if tip:
        lines.append(f"**灵修提示**：{tip}")
    lines.append("")
    lines.append("**建议读经**：")
    for r in refs:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("**简短解经/应用**：")
    apply_map = {
        "清晨寻求与安静": "清晨的渴慕不仅是情感，更是‘秩序’：先以神为中心，再安排任务；当心归位，效率与专注自然提升。",
        "AI/工作焦虑与交托": "‘地虽改变，山虽摇动’，仍不致动摇。技术更迭中，坚持正直与服事他人，神的话使脚步稳当。",
        "时间管理与智慧": "‘求你指教我们怎样数算自己的日子’——计划要具体，目标要有限，把时间投资在长期有果效的事上。",
        "感恩与盼望": "感恩不是忽视难处，而是承认神在难处中仍做王；感恩让我们不被焦虑驱动，而被爱与盼望引导。",
        "家庭责任与看顾": "你尽力守护家人，神在你看不见处也在看顾；把孩子与配偶交在主手中，是爱的最高形式之一。",
        "孩子教育/为儿女祷告": "以智慧与温柔引导，以界限与榜样塑造；先做‘可跟随的人’，孩子才学会跟随真理。"
    }
    lines.append(apply_map.get(theme, "让神的话成为你今天的路标——先寻求、再行动；先交托、再奔跑。"))
    lines.append("")
    lines.append("> 说明：为版权与版本差异，这里只列‘经文出处’；你可在常用中文/英文译本中对照阅读。")
    lines.append("")
    lines.append(f"_生成时间：{dt_local.strftime('%Y-%m-%d %H:%M %Z')}_")
    return "\n".join(lines)

def main():
    # 将时间固定在 America/Los_Angeles，便于你对齐“清晨灵修”
    la = tz.gettz("America/Los_Angeles")
    now_utc = datetime.datetime.utcnow().replace(tzinfo=tz.UTC)
    now_local = now_utc.astimezone(la)

    verses = load_verses()
    theme = pick_theme(now_local)
    refs = pick_scriptures(theme)

    # 输出文件
    out_dir = "daily"
    os.makedirs(out_dir, exist_ok=True)
    date_str = now_local.strftime("%Y-%m-%d")
    out_path = os.path.join(out_dir, f"{date_str}.md")

    title = f"# 每日经文与灵修 · {date_str}"
    body = build_reflection(theme, refs, now_local)
    content = f"{title}\n\n{body}\n"

    # 若文件已存在则不重复写（避免重复 commit）
    if os.path.exists(out_path):
        print(f"{out_path} already exists. Skip writing.")
        return

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
