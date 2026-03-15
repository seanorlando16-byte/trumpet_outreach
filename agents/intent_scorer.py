"""
Intent Scoring Agent

Finds and scores buying intent signals that indicate a company is likely
to need or be actively looking for a Digital Sales Room like Trumpet.

Score: 0–100
 75+  → HOT  (reach out today)
 50–74 → WARM (prioritise this week)
 25–49 → LUKEWARM (good nurture target)
  0–24 → COLD (low priority right now)
"""

from __future__ import annotations

import json
import anthropic
from pydantic import BaseModel

from config import ANTHROPIC_API_KEY, MODEL, TRUMPET_CONTEXT, RESEARCH_TOOLS
from agents.account_researcher import AccountProfile
from utils.display import console, section, stream_text


class IntentSignal(BaseModel):
    signal: str               # short label for the signal
    strength: str             # "high" | "medium" | "low"
    evidence: str             # what you found, with source/date if possible
    source_url: str = ""


class IntentReport(BaseModel):
    company_name: str
    intent_score: int                    # 0–100
    score_rationale: str                 # why this score
    signals: list[IntentSignal]
    timing_assessment: str               # is now a good time? why?
    competitor_signals: list[str]        # evidence they use/evaluate competitors
    trigger_events: list[str]           # specific events that increase urgency
    recommended_hook: str               # the most compelling opening hook to use


SYSTEM_PROMPT = """You are a sales intelligence assistant for Trumpet (sendtrumpet.com), a B2B Digital Sales Room platform.

Your job is to find buying intent signals for a target company. Use web_search and web_fetch to search for:
1. JOB POSTINGS — Sales Enablement, RevOps, Sales Excellence, AE/SDR hiring
2. LEADERSHIP CHANGES — new CRO/VP Sales in last 12 months
3. FUNDING EVENTS — recent investment rounds
4. COMPETITOR SIGNALS — using/evaluating Qwilr, Aligned, GetAccept, DealRoom
5. PUBLIC PAIN POINTS — posts/blogs about sales process challenges or proposal inefficiency
6. GROWTH SIGNALS — new markets, M&A, rapid headcount expansion

Output ONLY a JSON object (no markdown):
{
  "company_name": "string",
  "intent_score": integer (0-100),
  "score_rationale": "string",
  "signals": [{"signal": "string", "strength": "high|medium|low", "evidence": "string", "source_url": "string"}],
  "timing_assessment": "string",
  "competitor_signals": ["string"],
  "trigger_events": ["string"],
  "recommended_hook": "string"
}

Score 75+ for: leadership change + funding + active hiring. Score 40-60 for general growth. Score <25 for no signals.
"""


def score_intent(
    company: str,
    profile: AccountProfile | None = None,
    verbose: bool = True,
) -> IntentReport:
    """Research buying intent signals and score a target account."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if verbose:
        section(f"Scoring buying intent for {company}")
        console.print()

    profile_context = ""
    if profile:
        profile_context = f"""
Company context:
- Industry: {profile.industry}
- Size: {profile.company_size}
- Funding: {profile.funding_stage} {profile.funding_amount}
- Sales motion: {profile.current_sales_motion}
- Recent news: {'; '.join(profile.recent_news[:3]) if profile.recent_news else 'unknown'}
"""

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Find all buying intent signals for {company} as a Trumpet prospect.\n"
                f"{profile_context}\n"
                "Search job boards, LinkedIn, Crunchbase, G2 reviews, and news."
            ),
        }
    ]

    raw_json = ""

    while True:
        with client.messages.stream(
            model=MODEL,
            max_tokens=6000,
            system=SYSTEM_PROMPT,
            tools=RESEARCH_TOOLS,
            messages=messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        if verbose:
                            stream_text(event.delta.text)

            response = stream.get_final_message()

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            for block in response.content:
                if getattr(block, "type", None) == "text" and block.text:
                    raw_json = block.text.strip()
            break

        if response.stop_reason in ("tool_use", "pause_turn"):
            continue

        break

    if verbose:
        console.print()

    try:
        data = json.loads(raw_json)
        return IntentReport(**data)
    except Exception:
        import re
        match = re.search(r'\{[\s\S]+\}', raw_json)
        if match:
            data = json.loads(match.group())
            return IntentReport(**data)
        return IntentReport(
            company_name=company,
            intent_score=0,
            score_rationale="Intent research incomplete — please re-run.",
            signals=[],
            timing_assessment="Unknown",
            competitor_signals=[],
            trigger_events=[],
            recommended_hook="Unable to determine — please re-run.",
        )
