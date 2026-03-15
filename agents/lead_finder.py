"""
Lead Finder Agent

Identifies the right people to contact at a target account.
Prioritises Trumpet's key personas: VP Sales, CRO, Sales Enablement,
RevOps, and champion-level AEs.
"""

from __future__ import annotations

import json
import anthropic
from pydantic import BaseModel

from config import ANTHROPIC_API_KEY, MODEL, TRUMPET_CONTEXT, RESEARCH_TOOLS
from agents.account_researcher import AccountProfile
from utils.display import console, section, stream_text


class Lead(BaseModel):
    name: str
    title: str
    seniority: str                    # "economic_buyer" | "champion" | "influencer"
    persona_fit: str                  # why this person maps to a Trumpet buyer persona
    linkedin_url: str = ""
    email_guess: str = ""             # educated guess or format
    contact_hint: str = ""            # how to find/reach them
    talking_points: list[str] = []   # specific hooks for this individual
    priority: str = "medium"         # "high" | "medium" | "low"


class LeadList(BaseModel):
    company_name: str
    leads: list[Lead]
    outreach_strategy: str           # who to contact first and why


SYSTEM_PROMPT = f"""You are a sales intelligence assistant for Trumpet (sendtrumpet.com), a B2B Digital Sales Room platform. Target personas: VP Sales, CRO, Sales Enablement Manager, RevOps Lead, Head of GTM, Senior AEs.

Your job is to identify the best people to contact at a target company for
Trumpet outreach. Use web_search and web_fetch to find real people.

Search LinkedIn, the company website's team page, press releases, podcasts,
conference speaker lists, and news articles to find named individuals.

TARGET PERSONAS (in priority order):
1. VP Sales / Head of Sales / SVP Sales — economic buyer, feels the pain daily
2. Chief Revenue Officer (CRO) — strategic buyer at larger companies
3. Sales Enablement Manager / Director — evaluates and rolls out tools
4. Revenue Operations / RevOps Lead — owns tech stack decisions
5. Head of GTM / VP GTM — cares about efficiency and speed
6. Senior AEs / Enterprise AEs — champions who will push the tool up

For each person found, output their:
- Full name
- Exact current title
- Seniority classification (economic_buyer / champion / influencer)
- Why they map to a Trumpet persona
- LinkedIn URL (if findable)
- Email guess (firstname@company.com or format used by company)
- Contact hint (how to reach / find them if no direct contact)
- 2-3 specific personalised talking points for THIS individual
  (reference their background, tenure, role, anything public about them)
- Priority rating (high / medium / low)

After researching, output a JSON object matching this schema:
{{
  "company_name": "string",
  "leads": [
    {{
      "name": "string",
      "title": "string",
      "seniority": "economic_buyer | champion | influencer",
      "persona_fit": "string (1-2 sentences)",
      "linkedin_url": "string (or empty)",
      "email_guess": "string (e.g. john.smith@company.com or empty)",
      "contact_hint": "string",
      "talking_points": ["string", "string", "string"],
      "priority": "high | medium | low"
    }}
  ],
  "outreach_strategy": "string (paragraph: who to contact first, in what sequence, and why)"
}}

Find 3-6 real leads. Output ONLY the JSON, no markdown fences.
"""


def find_leads(
    company: str,
    profile: AccountProfile | None = None,
    verbose: bool = True,
) -> LeadList:
    """Find the right leads at a company for Trumpet outreach."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if verbose:
        section(f"Finding leads at {company}")
        console.print()

    # Build context from profile if available
    profile_context = ""
    if profile:
        profile_context = f"""
Company context already researched:
- Industry: {profile.industry}
- Size: {profile.company_size}
- Business model: {profile.business_model}
- Sales motion: {profile.current_sales_motion}
- Recommended angle: {profile.recommended_angle}
"""

    messages: list[dict] = [
        {
            "role": "user",
            "content": (
                f"Find the best Trumpet leads at {company}.\n{profile_context}\n"
                "Search LinkedIn, their team page, and press releases to find real people."
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
            collected_text = ""
            for event in stream:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        if verbose:
                            stream_text(event.delta.text)
                        collected_text += event.delta.text

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
        return LeadList(**data)
    except Exception:
        import re
        match = re.search(r'\{[\s\S]+\}', raw_json)
        if match:
            data = json.loads(match.group())
            return LeadList(**data)
        return LeadList(
            company_name=company,
            leads=[],
            outreach_strategy="Lead research incomplete — please re-run.",
        )
