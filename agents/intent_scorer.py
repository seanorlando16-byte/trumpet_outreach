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


SYSTEM_PROMPT = f"""{TRUMPET_CONTEXT}

Your job is to find buying intent signals — evidence that a company is likely
to need or be actively looking for a Digital Sales Room (DSR) like Trumpet.

Use web_search and web_fetch to search for:

1. JOB POSTINGS — search their current open roles for:
   - "Sales Enablement", "Revenue Operations", "RevOps", "Sales Excellence"
   - "Sales Cycle", "Deal Management", "Buyer Experience"
   - These indicate they're actively investing in sales process

2. LEADERSHIP CHANGES — new CRO, VP Sales, Head of Sales in the last 12 months
   (new leaders often bring in new tools in their first 90 days)

3. FUNDING EVENTS — recent investment rounds (they'll be scaling the sales team)

4. HIRING SIGNALS — lots of AE / SDR / BDR hiring (growing sales team = more complexity)

5. COMPETITOR SIGNALS — are they using or evaluating Qwilr, Aligned, GetAccept,
   DealRoom, or similar tools? (reviews on G2, LinkedIn posts, job listings)

6. PUBLIC PAIN POINTS — LinkedIn posts, blog posts, podcast interviews where
   leadership or AEs talk about sales process challenges, buyer experience,
   deal stalling, or proposal inefficiency

7. COMPANY GROWTH SIGNALS — expansion into new markets, new product lines,
   M&A activity (all create sales complexity)

8. CONTENT SIGNALS — white papers, webinars, or events they run about
   sales excellence, revenue growth, or GTM strategy

After researching, output a JSON object matching this schema:
{{
  "company_name": "string",
  "intent_score": integer (0-100),
  "score_rationale": "string (2-3 sentences explaining the score)",
  "signals": [
    {{
      "signal": "string (short label)",
      "strength": "high | medium | low",
      "evidence": "string (what you found, with detail)",
      "source_url": "string (if available)"
    }}
  ],
  "timing_assessment": "string (is now a good time? specific reason why)",
  "competitor_signals": ["string", ...],
  "trigger_events": ["string (specific event + date if known)", ...],
  "recommended_hook": "string (the single most compelling opening hook based on signals)"
}}

Score high (75+) when you find: leadership changes + funding + active hiring.
Score medium (40-60) when you find: general growth signals + relevant industry.
Score low (<25) when there are no clear intent signals.

Output ONLY the JSON, no markdown fences.
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
            thinking={"type": "adaptive"},
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
                if hasattr(block, "text"):
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
