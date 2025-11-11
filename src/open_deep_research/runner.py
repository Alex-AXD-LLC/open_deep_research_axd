"""Simple runner to produce a final report without requiring the LangGraph UI.

This module provides a minimal flow that uses the existing prompt functions
and model integrations to turn a user topic into a final report. It is
intended for CLI or lightweight web front ends.

Behavior:
- Accepts a research topic (string).
- Runs the research-brief generation step to produce a focused brief.
- Runs the final report generator using the brief and any provided messages.
- Writes the final report to a Markdown file by default.

Notes/assumptions:
- This runner uses the existing LLM integrations. Make sure model API keys
  and relevant environment variables are set (see `Configuration`).
- This is a simplified flow and does not execute the full supervisor/researcher
  graphs; it intentionally avoids LangGraph runtime invocation to provide a
  non-proprietary UI/CLI entrypoint.
"""

import asyncio
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage

from open_deep_research.configuration import Configuration
from open_deep_research.deep_researcher import (
    write_research_brief,
    final_report_generation,
)


async def _async_run_report(topic: str, output_path: str | None = "report.md") -> str:
    """Async implementation that runs the simplified flow and writes markdown.

    Returns the path to the written markdown file.
    """
    # Prepare initial state using simple message list
    state = {
        "messages": [HumanMessage(content=topic)],
    }

    # Call the research brief generator (returns a Command-like object)
    cmd = await write_research_brief(state, None)

    # The function returns a Command with an update payload that should contain
    # the research_brief. Try to extract it robustly.
    research_brief = None
    try:
        # langgraph Command exposes `.update`
        research_brief = getattr(cmd, "update", {}).get("research_brief")
    except Exception:
        research_brief = None

    if not research_brief:
        # Fallback: use the original topic as the brief
        research_brief = topic

    # Build state for final report generation. We include messages and an
    # initially-empty notes list so the final report generator has context.
    final_state = {
        "research_brief": research_brief,
        "messages": [HumanMessage(content=research_brief)],
        "notes": [],
    }

    result = await final_report_generation(final_state, None)
    final_report = result.get("final_report") if isinstance(result, dict) else None

    if not final_report:
        # Try to extract from returned object (some code returns AIMessage)
        if isinstance(result, AIMessage):
            final_report = str(result.content)
        else:
            final_report = """Error: final report generation failed or returned empty output."""

    # Ensure markdown output
    md = final_report if isinstance(final_report, str) else str(final_report)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

    return output_path or md


def run_report(topic: str, output_path: Optional[str] = "report.md") -> str:
    """Synchronous wrapper for running the async report generator."""
    return asyncio.run(_async_run_report(topic, output_path))


if __name__ == "__main__":
    # Quick manual runner for development
    import argparse

    parser = argparse.ArgumentParser(description="Generate a research report to Markdown.")
    parser.add_argument("topic", help="Research topic or user message")
    parser.add_argument("--out", "-o", default="report.md", help="Output markdown file path")
    args = parser.parse_args()

    path = run_report(args.topic, args.out)
    print(f"Wrote report to: {path}")
