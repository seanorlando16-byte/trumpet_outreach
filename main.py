#!/usr/bin/env python3
"""
Trumpet Sales Outreach Engine
------------------------------
AI-powered account research, lead finding, intent scoring, and
personalised copy generation for Trumpet (sendtrumpet.com).

Usage:
  python main.py research <company>            # Deep account research
  python main.py leads <company>               # Find decision makers
  python main.py intent <company>              # Score buying intent
  python main.py draft <company>               # Generate outreach copy
  python main.py run <company>                 # Full pipeline (all of the above)
"""

import os
import sys
import json
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
from rich.table import Table

from utils.display import (
    console,
    header,
    section,
    info,
    success,
    warn,
    error,
    print_leads_table,
    print_intent_table,
    score_badge,
)

load_dotenv()

# Ensure output directory exists
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def check_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        error("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)


def save_output(company: str, stage: str, data: dict) -> Path:
    """Save results to a JSON file in the output directory."""
    slug = company.lower().replace(" ", "_").replace("/", "_")
    path = OUTPUT_DIR / f"{slug}_{stage}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry points
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """Trumpet Sales Outreach Engine — AI-powered account intelligence."""
    pass


@cli.command()
@click.argument("company")
@click.option("--quiet", is_flag=True, help="Suppress streaming output")
def research(company: str, quiet: bool) -> None:
    """Research an account: industry, funding, sales motion, tech stack, and Trumpet fit."""
    check_api_key()
    from agents.account_researcher import research_account

    header(
        "Trumpet Account Research",
        f"Researching {company} for Trumpet outreach",
    )

    profile = research_account(company, verbose=not quiet)

    section("Account Profile")
    table = Table(box=box.ROUNDED, border_style="blue", show_header=False)
    table.add_column("Field", style="bold cyan", width=22)
    table.add_column("Value")

    rows = [
        ("Company", profile.company_name),
        ("Website", profile.website),
        ("Industry", profile.industry),
        ("Size", profile.company_size),
        ("Funding", f"{profile.funding_stage} {profile.funding_amount}".strip()),
        ("HQ", profile.hq_location),
        ("Business Model", profile.business_model),
        ("Sales Motion", profile.current_sales_motion),
    ]
    for label, val in rows:
        if val.strip():
            table.add_row(label, val)

    console.print(table)
    console.print()

    if profile.tech_stack_signals:
        section("Tech Stack Signals")
        for t in profile.tech_stack_signals:
            info(t)
        console.print()

    if profile.recent_news:
        section("Recent News")
        for n in profile.recent_news:
            info(n)
        console.print()

    if profile.key_challenges:
        section("Key Challenges (Trumpet Relevant)")
        for c in profile.key_challenges:
            info(c)
        console.print()

    section("Trumpet Fit Assessment")
    console.print(Panel(profile.why_trumpet_fits, border_style="green", padding=(0, 2)))

    section("Recommended Outreach Angle")
    console.print(Panel(
        f"[bold yellow]{profile.recommended_angle}[/bold yellow]",
        border_style="yellow", padding=(0, 2)
    ))

    path = save_output(company, "research", profile.model_dump())
    success(f"Saved to {path}")


@cli.command()
@click.argument("company")
@click.option("--quiet", is_flag=True, help="Suppress streaming output")
def leads(company: str, quiet: bool) -> None:
    """Find decision makers and champions at a target account."""
    check_api_key()
    from agents.account_researcher import research_account
    from agents.lead_finder import find_leads

    header("Lead Finder", f"Finding Trumpet leads at {company}")

    # Try loading cached research first
    slug = company.lower().replace(" ", "_").replace("/", "_")
    cache_path = OUTPUT_DIR / f"{slug}_research.json"
    profile = None
    if cache_path.exists():
        from agents.account_researcher import AccountProfile
        profile = AccountProfile(**json.loads(cache_path.read_text()))
        info(f"Loaded cached research from {cache_path}")
    else:
        info("No cached research found — running account research first...")
        profile = research_account(company, verbose=not quiet)

    lead_list = find_leads(company, profile, verbose=not quiet)

    section("Leads Found")
    leads_data = [
        {
            "name": l.name,
            "title": l.title,
            "why_target": l.persona_fit[:80] + "..." if len(l.persona_fit) > 80 else l.persona_fit,
            "linkedin_url": l.linkedin_url or l.contact_hint,
        }
        for l in lead_list.leads
    ]
    print_leads_table(leads_data)
    console.print()

    # Show talking points for high-priority leads
    high_priority = [l for l in lead_list.leads if l.priority == "high"]
    if high_priority:
        section("High-Priority Lead Talking Points")
        for lead in high_priority:
            console.print(f"\n[bold white]{lead.name}[/bold white] — [cyan]{lead.title}[/cyan]")
            for tp in lead.talking_points:
                info(tp)

    section("Outreach Strategy")
    console.print(Panel(lead_list.outreach_strategy, border_style="green", padding=(0, 2)))

    path = save_output(company, "leads", lead_list.model_dump())
    success(f"Saved to {path}")


@cli.command()
@click.argument("company")
@click.option("--quiet", is_flag=True, help="Suppress streaming output")
def intent(company: str, quiet: bool) -> None:
    """Score buying intent — find signals that a company needs Trumpet now."""
    check_api_key()
    from agents.account_researcher import research_account, AccountProfile
    from agents.intent_scorer import score_intent

    header("Intent Scorer", f"Finding buying signals for {company}")

    slug = company.lower().replace(" ", "_").replace("/", "_")
    cache_path = OUTPUT_DIR / f"{slug}_research.json"
    profile = None
    if cache_path.exists():
        profile = AccountProfile(**json.loads(cache_path.read_text()))
        info(f"Loaded cached research from {cache_path}")

    report = score_intent(company, profile, verbose=not quiet)

    section("Intent Score")
    console.print(f"\n  {score_badge(report.intent_score)}\n")
    console.print(Panel(report.score_rationale, border_style="cyan", padding=(0, 2)))
    console.print()

    if report.signals:
        section("Buying Signals")
        signals_data = [
            {
                "signal": s.signal,
                "strength": s.strength,
                "evidence": (s.evidence[:120] + "...") if len(s.evidence) > 120 else s.evidence,
            }
            for s in report.signals
        ]
        print_intent_table(signals_data)
        console.print()

    if report.trigger_events:
        section("Trigger Events")
        for event in report.trigger_events:
            info(event)
        console.print()

    if report.competitor_signals:
        section("Competitor Signals")
        for sig in report.competitor_signals:
            warn(sig)
        console.print()

    section("Timing Assessment")
    console.print(Panel(report.timing_assessment, border_style="yellow", padding=(0, 2)))

    section("Recommended Hook")
    console.print(Panel(
        f"[bold yellow]{report.recommended_hook}[/bold yellow]",
        border_style="yellow", padding=(0, 2),
    ))

    path = save_output(company, "intent", report.model_dump())
    success(f"Saved to {path}")


@cli.command()
@click.argument("company")
@click.option("--lead-name", default="", help="Specific lead name to write for")
@click.option("--lead-title", default="", help="Lead's job title")
@click.option("--lead-index", default=0, help="Index of lead from find-leads results (0-based)")
@click.option("--quiet", is_flag=True, help="Suppress streaming output")
def draft(
    company: str,
    lead_name: str,
    lead_title: str,
    lead_index: int,
    quiet: bool,
) -> None:
    """Generate personalised outreach copy for a lead."""
    check_api_key()
    from agents.account_researcher import research_account, AccountProfile
    from agents.lead_finder import find_leads, Lead, LeadList
    from agents.intent_scorer import score_intent, IntentReport
    from agents.copy_generator import generate_copy

    header("Copy Generator", f"Drafting personalised Trumpet outreach for {company}")

    slug = company.lower().replace(" ", "_").replace("/", "_")

    # Load or generate profile
    research_path = OUTPUT_DIR / f"{slug}_research.json"
    if research_path.exists():
        profile = AccountProfile(**json.loads(research_path.read_text()))
        info(f"Using cached research from {research_path}")
    else:
        info("Running account research...")
        profile = research_account(company, verbose=not quiet)

    # Load or generate leads
    leads_path = OUTPUT_DIR / f"{slug}_leads.json"
    if leads_path.exists():
        lead_list = LeadList(**json.loads(leads_path.read_text()))
        info(f"Using cached leads from {leads_path}")
    else:
        info("Finding leads...")
        lead_list = find_leads(company, profile, verbose=not quiet)

    # Load or generate intent
    intent_path = OUTPUT_DIR / f"{slug}_intent.json"
    intent_report = None
    if intent_path.exists():
        intent_report = IntentReport(**json.loads(intent_path.read_text()))
        info(f"Using cached intent from {intent_path}")

    # Select which lead to write for
    selected_lead: Lead | None = None
    if lead_name:
        for l in lead_list.leads:
            if lead_name.lower() in l.name.lower():
                selected_lead = l
                break
        if not selected_lead:
            # Create a minimal lead from the provided info
            selected_lead = Lead(
                name=lead_name,
                title=lead_title or "Unknown",
                seniority="economic_buyer",
                persona_fit="Specified by user",
                talking_points=[],
                priority="high",
            )
    elif lead_list.leads:
        if 0 <= lead_index < len(lead_list.leads):
            selected_lead = lead_list.leads[lead_index]
        else:
            selected_lead = lead_list.leads[0]
    else:
        error("No leads found. Run `python main.py leads <company>` first.")
        sys.exit(1)

    info(f"Writing for: {selected_lead.name} — {selected_lead.title}")

    copy = generate_copy(selected_lead, profile, intent_report, verbose=not quiet)

    # Display the copy beautifully
    _display_copy(copy)

    # Save
    slug_lead = selected_lead.name.lower().replace(" ", "_")
    path = save_output(company, f"copy_{slug_lead}", copy.model_dump())
    success(f"Saved to {path}")


def _display_copy(copy) -> None:
    """Render all copy formats to the terminal."""
    from rich.panel import Panel

    section("Cold Email")
    console.print(Panel(
        f"[bold]Subject:[/bold] {copy.email_subject}\n\n{copy.email_body}",
        border_style="green", title="Email", padding=(1, 2)
    ))
    console.print()

    section("LinkedIn")
    console.print(Panel(
        f"[bold]Connection Note[/bold] (≤300 chars):\n{copy.linkedin_connection_note}"
        f"\n\n[bold]Follow-up DM:[/bold]\n{copy.linkedin_dm}",
        border_style="blue", title="LinkedIn", padding=(1, 2)
    ))
    console.print()

    section("Phone")
    console.print(Panel(
        f"[bold]Cold Call Opener:[/bold]\n{copy.call_opener}"
        f"\n\n[bold]Voicemail:[/bold]\n{copy.voicemail_script}",
        border_style="magenta", title="Phone", padding=(1, 2)
    ))
    console.print()

    if copy.discovery_questions:
        section("Discovery Questions")
        for i, q in enumerate(copy.discovery_questions, 1):
            console.print(f"  [cyan]{i}.[/cyan] {q}")
        console.print()

    if copy.objection_responses:
        section("Objection Handling")
        for obj, resp in copy.objection_responses.items():
            console.print(f'\n  [bold yellow]"{obj}"[/bold yellow]')
            console.print(f"  {resp}")
        console.print()


@cli.command()
@click.argument("company")
@click.option("--quiet", is_flag=True, help="Suppress streaming output")
def run(company: str, quiet: bool) -> None:
    """Full pipeline: research → leads → intent → copy for the top lead."""
    check_api_key()
    from agents.account_researcher import research_account, AccountProfile
    from agents.lead_finder import find_leads, LeadList
    from agents.intent_scorer import score_intent, IntentReport
    from agents.copy_generator import generate_copy

    header(
        "Trumpet Full Pipeline",
        f"Complete account intelligence + outreach copy for {company}",
    )

    slug = company.lower().replace(" ", "_").replace("/", "_")

    # 1. Account research
    console.print("\n[bold blue]Step 1/4 — Account Research[/bold blue]\n")
    research_cache = OUTPUT_DIR / f"{slug}_research.json"
    if research_cache.exists():
        profile = AccountProfile(**json.loads(research_cache.read_text()))
        info(f"Loaded cached research from {research_cache}")
    else:
        profile = research_account(company, verbose=not quiet)
        save_output(company, "research", profile.model_dump())

    # 2. Leads
    console.print("\n[bold blue]Step 2/4 — Lead Discovery[/bold blue]\n")
    leads_cache = OUTPUT_DIR / f"{slug}_leads.json"
    if leads_cache.exists():
        lead_list = LeadList(**json.loads(leads_cache.read_text()))
        info(f"Loaded cached leads from {leads_cache}")
    else:
        lead_list = find_leads(company, profile, verbose=not quiet)
        save_output(company, "leads", lead_list.model_dump())

    # 3. Intent
    console.print("\n[bold blue]Step 3/4 — Intent Scoring[/bold blue]\n")
    intent_cache = OUTPUT_DIR / f"{slug}_intent.json"
    if intent_cache.exists():
        intent_report = IntentReport(**json.loads(intent_cache.read_text()))
        info(f"Loaded cached intent from {intent_cache}")
    else:
        intent_report = score_intent(company, profile, verbose=not quiet)
        save_output(company, "intent", intent_report.model_dump())

    # 4. Copy for the highest-priority lead
    console.print("\n[bold blue]Step 4/4 — Outreach Copy[/bold blue]\n")
    high_priority = [l for l in lead_list.leads if l.priority == "high"]
    top_lead = high_priority[0] if high_priority else (lead_list.leads[0] if lead_list.leads else None)

    if top_lead:
        copy = generate_copy(top_lead, profile, intent_report, verbose=not quiet)
        slug_lead = top_lead.name.lower().replace(" ", "_")
        save_output(company, f"copy_{slug_lead}", copy.model_dump())
        _display_copy(copy)
    else:
        warn("No leads found — skipping copy generation.")

    # Summary
    console.print()
    section("Pipeline Summary")
    slug = company.lower().replace(" ", "_").replace("/", "_")
    table = Table(box=box.ROUNDED, border_style="green", show_header=False)
    table.add_column("", style="bold cyan", width=22)
    table.add_column("")
    table.add_row("Company", profile.company_name)
    table.add_row("Industry", profile.industry)
    table.add_row("Trumpet Fit", profile.recommended_angle)
    table.add_row("Intent Score", score_badge(intent_report.intent_score))
    table.add_row("Leads Found", str(len(lead_list.leads)))
    if top_lead:
        table.add_row("Copy Written For", f"{top_lead.name} ({top_lead.title})")
    table.add_row("Output Files", str(OUTPUT_DIR.absolute()))
    console.print(table)


if __name__ == "__main__":
    cli()
