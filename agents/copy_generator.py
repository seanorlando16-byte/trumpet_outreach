"""
Copy Generator Agent

Takes the account research, leads, and intent signals and generates
highly personalised outreach copy for each lead.

Outputs:
- Cold email (subject + body)
- LinkedIn connection request message
- LinkedIn InMail / DM follow-up
- Call opening / voicemail script
- Key talking points for discovery
"""

from __future__ import annotations

import anthropic
from pydantic import BaseModel

from config import ANTHROPIC_API_KEY, MODEL, TRUMPET_CONTEXT
from agents.account_researcher import AccountProfile
from agents.lead_finder import Lead
from agents.intent_scorer import IntentReport
from utils.display import console, section, stream_text


class OutreachCopy(BaseModel):
    lead_name: str
    lead_title: str
    company_name: str

    email_subject: str
    email_body: str

    linkedin_connection_note: str     # 300 char limit
    linkedin_dm: str                  # follow-up after connection

    call_opener: str                  # first 30 seconds on a cold call
    voicemail_script: str             # if they don't pick up (under 30s)

    discovery_questions: list[str]    # 3-5 questions to ask in a meeting
    objection_responses: dict[str, str]  # common objections + how to respond


SYSTEM_PROMPT = f"""{TRUMPET_CONTEXT}

You are a world-class B2B sales copywriter specialising in SaaS outreach.
Write highly personalised, human-sounding outreach copy for Trumpet.

WRITING PRINCIPLES:
1. Lead with THEIR world, not Trumpet's features — reference something real about them
2. One specific insight or trigger event per message (not generic)
3. Short and punchy — emails under 150 words, LinkedIn notes under 300 chars
4. Clear single call-to-action — one ask per message (usually "15 min chat?")
5. Tone: confident, warm, curious — never pushy or desperate
6. Never start an email with "I" — lead with "You", their company name, or a question
7. Use the recommended angle and intent signals as the hook
8. Email subject lines: specific + benefit-hinting, under 8 words
9. Sound like a human, not a template — reference specifics from the research
10. For LinkedIn: even more conversational and brief

WHAT MAKES TRUMPET OUTREACH LAND:
- Reference a specific pain point (e.g. "deal visibility" or "proposal black holes")
- Connect it to something real about them (hiring SDRs, raised funding, new CRO)
- Show you understand their sales motion (PLG vs. sales-led matters)
- Make the ask small (15 min, not "a demo")
- Social proof from similar companies if possible

Generate all copy formats (email, LinkedIn, call script, discovery questions,
and objection handling). Be specific — reference the lead's name, role, company,
and the specific signals found in research.
"""


def generate_copy(
    lead: Lead,
    profile: AccountProfile,
    intent: IntentReport | None = None,
    verbose: bool = True,
) -> OutreachCopy:
    """Generate personalised outreach copy for a specific lead."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if verbose:
        section(f"Drafting copy for {lead.name} at {profile.company_name}")
        console.print()

    # Build rich context for the copy generator
    intent_summary = ""
    if intent:
        top_signals = [
            f"- {s.signal}: {s.evidence}"
            for s in (intent.signals or [])
            if s.strength in ("high", "medium")
        ][:5]
        intent_summary = f"""
INTENT SIGNALS FOUND:
{chr(10).join(top_signals) if top_signals else 'No strong signals found'}

Trigger events: {', '.join(intent.trigger_events[:3]) if intent.trigger_events else 'none'}
Competitor signals: {', '.join(intent.competitor_signals[:2]) if intent.competitor_signals else 'none'}
Recommended hook: {intent.recommended_hook}
"""

    user_prompt = f"""
LEAD TO WRITE FOR:
Name: {lead.name}
Title: {lead.title}
Seniority: {lead.seniority}
Why they're a target: {lead.persona_fit}
Personal talking points: {'; '.join(lead.talking_points)}

COMPANY CONTEXT:
Company: {profile.company_name}
Industry: {profile.industry}
Size: {profile.company_size}
Funding: {profile.funding_stage} {profile.funding_amount}
Product: {profile.product_description}
Sales motion: {profile.current_sales_motion}
Tech stack: {', '.join(profile.tech_stack_signals[:5]) if profile.tech_stack_signals else 'unknown'}
Recent news: {'; '.join(profile.recent_news[:3]) if profile.recent_news else 'none'}
Why Trumpet fits: {profile.why_trumpet_fits}
Recommended angle: {profile.recommended_angle}

{intent_summary}

Write ALL the outreach copy formats for this lead. Make it feel like a
Trumpet rep who has done genuine research — not a template.

Return your response in this exact format (use these exact section headers):

## EMAIL SUBJECT
[subject line here]

## EMAIL BODY
[email body here]

## LINKEDIN CONNECTION NOTE
[under 300 characters]

## LINKEDIN DM
[follow-up message after connecting]

## CALL OPENER
[first 30 seconds of a cold call]

## VOICEMAIL SCRIPT
[under 30 seconds if they don't pick up]

## DISCOVERY QUESTIONS
1. [question]
2. [question]
3. [question]
4. [question]
5. [question]

## OBJECTION RESPONSES
**"We're happy with what we have":**
[response]

**"We don't have budget right now":**
[response]

**"Send me some information":**
[response]

**"We already use [competitor]":**
[response]
"""

    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    # Use streaming for copy generation (long output)
    full_text = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        for event in stream:
            if event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    if verbose:
                        stream_text(event.delta.text)
                    full_text += event.delta.text

    if verbose:
        console.print()

    # Parse the structured response
    return _parse_copy(full_text, lead, profile)


def _parse_copy(text: str, lead: Lead, profile: AccountProfile) -> OutreachCopy:
    """Parse the model's formatted response into an OutreachCopy object."""

    def extract_section(header: str, next_headers: list[str]) -> str:
        import re
        pattern = rf"## {re.escape(header)}\n([\s\S]+?)(?=## (?:{'|'.join(re.escape(h) for h in next_headers)})|$)"
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""

    all_headers = [
        "EMAIL SUBJECT", "EMAIL BODY", "LINKEDIN CONNECTION NOTE",
        "LINKEDIN DM", "CALL OPENER", "VOICEMAIL SCRIPT",
        "DISCOVERY QUESTIONS", "OBJECTION RESPONSES",
    ]

    def get_section(header: str) -> str:
        idx = all_headers.index(header)
        next_h = all_headers[idx + 1:] if idx + 1 < len(all_headers) else ["END"]
        return extract_section(header, next_h)

    # Parse discovery questions
    dq_text = get_section("DISCOVERY QUESTIONS")
    import re
    questions = re.findall(r'^\d+\.\s+(.+)$', dq_text, re.MULTILINE)

    # Parse objection responses
    obj_text = get_section("OBJECTION RESPONSES")
    objection_pattern = r'\*\*"([^"]+)"\*\*:\s*([\s\S]+?)(?=\*\*"|$)'
    objections = {}
    for m in re.finditer(objection_pattern, obj_text):
        objections[m.group(1)] = m.group(2).strip()

    return OutreachCopy(
        lead_name=lead.name,
        lead_title=lead.title,
        company_name=profile.company_name,
        email_subject=get_section("EMAIL SUBJECT"),
        email_body=get_section("EMAIL BODY"),
        linkedin_connection_note=get_section("LINKEDIN CONNECTION NOTE"),
        linkedin_dm=get_section("LINKEDIN DM"),
        call_opener=get_section("CALL OPENER"),
        voicemail_script=get_section("VOICEMAIL SCRIPT"),
        discovery_questions=questions or [],
        objection_responses=objections or {},
    )
