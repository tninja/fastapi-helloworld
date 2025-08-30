import os, json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# 直接用 OpenAI 官方 SDK；它会从 OPENAI_API_KEY 环境变量读取密钥
client = OpenAI()

app = FastAPI(title="Comfort API (OpenAI SDK)")

# 修改为你的 GitHub Pages 域名（用户页和/或项目页）
ALLOWED_ORIGINS = [
    "https://tninja.github.io",
    "https://tninja.github.io/fastapi-helloworld"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ComfortQuery(BaseModel):
    language: str = "zh"               # "zh" 或 "en"
    keywords: str                      # 关键词：如“工作焦虑、AI变动、家庭责任”
    situation: Optional[str] = ""      # 处境描述
    faith_background: Optional[str] = "christian"
    max_passages: int = 3              # 返回几段经文（1-3）

class Passage(BaseModel):
    ref: str
    short_quote: str
    reason: str

class ComfortResponse(BaseModel):
    passages: List[Passage]
    devotional: str
    prayer: str
    disclaimer: str

SYSTEM_PROMPT = """You are a gentle Christian pastoral counselor and Bible study helper.
Respond in the user's requested language (zh or en).
Propose Bible passages (book chapter:verse) fitting the user's situation.
For quotes: provide at most a very short paraphrase (<= 20 words/chars) or leave empty.
Write a concise pastoral devotional (150-250 zh characters / 150-200 English words) and a short prayer (2-4 sentences).
Avoid doctrinal disputes, be comforting and practical.
Return STRICT JSON only, matching the schema the user supplies.
If unsure about an exact verse, choose one you are confident in.
Do NOT include long verbatim quotes from copyrighted translations.
"""

USER_PROMPT_TMPL = """User language: {language}
Faith background: {faith_background}
User keywords: {keywords}
Situation detail: {situation}

Return JSON with fields:
- passages: array of at most {max_passages} objects with fields:
  - ref (string, e.g., "Psalm 46:1-3" or "诗篇 46:1-3")
  - short_quote (string, <= 20 words/chars; a paraphrase or public-domain-short snippet; MAY be empty)
  - reason (string, 1-2 sentences why this fits)
- devotional: a 150-250 {lang_unit} pastoral reflection applying these passages to the user's situation.
- prayer: 2-4 sentences prayer.
- disclaimer: one sentence kindly asking the user to verify in their preferred translation.

Use the requested language for everything.
"""

def build_messages(q: ComfortQuery):
    lang_unit = "字" if q.language.startswith("zh") else "words"
    uprompt = USER_PROMPT_TMPL.format(
        language=q.language,
        faith_background=q.faith_background or "christian",
        keywords=q.keywords,
        situation=q.situation or "",
        max_passages=max(1, min(q.max_passages, 3)),
        lang_unit=lang_unit,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": uprompt},
    ]

@app.post("/comfort", response_model=ComfortResponse)
def comfort(q: ComfortQuery):
    # 基本校验
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Server missing OPENAI_API_KEY")

    messages = build_messages(q)

    try:
        # 使用 Chat Completions（也可换成 Responses API，按你的账号可用性选择）
        resp = client.chat.completions.create(
            model="gpt-5-mini",  # 可换成你账户可用、性价比合适的模型
            messages=messages,
            # temperature=0.7, # gpt-5-mini 不支持 temperature
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        if not content:
            raise HTTPException(status_code=502, detail="LLM returned empty content")
        data = json.loads(content)

        # 约束：裁剪 passages 数量与短摘长度，避免版权/超长
        max_passages = max(1, min(q.max_passages, 3))
        passages = (data.get("passages") or [])[:max_passages]
        for p in passages:
            sq = (p.get("short_quote") or "").strip()
            if q.language.startswith("zh"):
                if len(sq) > 40:  # 粗略控制中文长度
                    p["short_quote"] = ""
            else:
                if len(sq.split()) > 20:
                    p["short_quote"] = ""

        data["passages"] = passages

        if not data.get("disclaimer"):
            data["disclaimer"] = (
                "请在你常用的圣经译本中核对经文原文与上下文；以上解读仅作灵修参考。"
                if q.language.startswith("zh")
                else "Please verify these references in your preferred Bible translation; the reflection is for devotional support."
            )

        return data

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")
