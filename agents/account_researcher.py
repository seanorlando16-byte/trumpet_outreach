"""
Account Research Agent

Researches a target company using Claude + web search.
Returns a structured AccountProfile with everything needed
to personalise outreach for Trumpet.
"""

from __future__ import annotations

import json
import anthropic
from pydantic import BaseModel

from config import ANTHROPIC_API_KEY, MODEL, TRUMPET_CONTEXT, RESEARCH_TOOLS
from utils.display import console, section, stream_text


class AccountProfile(BaseModel):
    company_name: str
    website: str = ""
    industry: str = ""
    company_size: str = ""
    funding_stage: str = ""
    funding_amount: str = ""
    hq_location: str = ""
    business_model: str = ""          # B2B SaaS / Professional Services / etc.
    product_description: str = ""
    target_customers: str = ""
    current_sales_motion: str = ""    # PLG / sales-led / hybrid
    tech_stack_signals: list[str] = []
    recent_news: list[str] = []
    key_challenges: list[str] = []    # challenges relevant to Trumpet
    why_trumpet_fits: str = ""        # one-paragraph fit assessment
    recommended_angle: str = ""       # the hook/angle to lead with


SYSTEM_PROMPT = f"""{TRUMPET_CONTEXT}

Your job is to deeply research a target company so a Trumpet sales rep can
craft highly personalised outreach.

Use web_search and web_fetch to gather:
- Company overview (industry, size, funding, HQ, business model)
- Their product and who they sell to
- Their current sales motion (PLG vs. sales-led, ACV range, sales team size)
- Technology signals (CRM, sales tools, proposal tools, content tools)
- Recent news (funding, leadership hires, product launches, partnerships)
- Pain points their sales team likely faces that Trumpet solves
- Public content about their sales process or buyer experience

After researching, output a JSON object that exactly matches this schema:
{{
  "company_name": "string",
  "website": "string",
  "industry": "string",
  "company_size": "string (e.g. '50-200 employees')",
  "funding_stage": "string (e.g. 'Series B')",
  "funding_amount": "string (e.g. '$40M')",
  "hq_location": "string",
  "business_model": "string (e.g. 'B2B SaaS')",
  "product_description": "string (2-3 sentences)",
  "target_customers": "string (who they sell to)",
  "current_sales_motion": "string (sales-led / PLG / hybrid, ACV range if known)",
  "tech_stack_signals": ["list of sales/marketing tools they use"],
  "recent_news": ["list of 3-5 notable recent developments"],
  "key_challenges": ["list of 3-5 sales challenges relevant to Trumpet"],
  "why_trumpet_fits": "string (paragraph explaining why Trumpet is a great fit)",
  "recommended_angle": "string (1-2 sentences — the specific angle/hook to lead with in outreach)"
}}

Output ONLY the JSON, no markdown fences, no explanation.
"""


def research_account(company: str, verbose: bool = True) -> AccountProfile:
    """Research a company and return a structured AccountProfile."""

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    if verbose:
        section(f"Researching {company}")
        console.print()

    messages: list[dict] = [
        {"role": "user", "content": f"Research this company for Trumpet outreach: {company}"}
    ]

    raw_json = ""

    # Agentic loop: Claude uses web search until it has enough information,
    # then outputs the structured JSON.
    while True:
        with client.messages.stream(
            model=MODEL,
            max_tokens=8000,
            thinking={"type": "adaptive"},
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

        # Append assistant response to message history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract the JSON the model wrote
            for block in response.content:
                if hasattr(block, "text"):
                    raw_json = block.text.strip()
            break

        if response.stop_reason == "tool_use":
            # Execute server-side tools (web search / fetch) — results come back
            # automatically; we just need to send the tool_result messages.
            tool_results = []
            for block in response.content:
                if block.type == "server_tool_use":
                    # Server-side tools return their results in the next response;
                    # we don't execute them client-side — just continue the loop.
                    pass
            # The next iteration will include tool results automatically
            # because they're server-side tools.
            # For server-side tools we just loop.
            continue

        if response.stop_reason == "pause_turn":
            continue

        break

    if verbose:
        console.print()

    # Parse and return
    try:
        data = json.loads(raw_json)
        return AccountProfile(**data)
    except (json.JSONDecodeError, Exception):
        # Fallback: try to find JSON in the text
        import re
        match = re.search(r'\{[\s\S]+\}', raw_json)
        if match:
            data = json.loads(match.group())
            return AccountProfile(**data)
        # Return minimal profile if parsing fails
        return AccountProfile(
            company_name=company,
            why_trumpet_fits="Research incomplete — please re-run.",
            recommended_angle="Unable to determine — please re-run.",
        )
