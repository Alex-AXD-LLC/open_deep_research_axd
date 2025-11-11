"""Command-line entrypoint for Open Deep Research (non-LangGraph mode).

Usage examples:
  python -m open_deep_research.cli "Write a report about transfer learning in vision"
  python -m open_deep_research.cli "Topic" --out my_report.md

This module uses `runner.run_report` and writes a Markdown file by default.
"""

import argparse
import httpx
import sys


def main():
  parser = argparse.ArgumentParser(description="Generate a research report to Markdown via FastAPI endpoint.")
  parser.add_argument("topic", help="Research topic or user message")
  parser.add_argument("--out", "-o", default="report.md", help="Output markdown file path")
  parser.add_argument("--api-url", default="http://127.0.0.1:8000/run", help="FastAPI /run endpoint URL")
  args = parser.parse_args()

  api_url = args.api_url
  messages = [args.topic]

  # Conversation loop: keep calling the API until we receive a final report
  while True:
    payload = {"messages": messages}
    try:
      resp = httpx.post(api_url, json=payload, timeout=120.0)
    except Exception as e:
      print(f"Error contacting API at {api_url}: {e}", file=sys.stderr)
      sys.exit(2)

    if resp.status_code != 200:
      print(f"API returned status {resp.status_code}: {resp.text}", file=sys.stderr)
      sys.exit(3)

    # Try to parse JSON response
    try:
      data = resp.json()
    except Exception:
      # Not JSON — assume raw markdown text
      md = resp.text
      with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
      print(f"Wrote report to: {args.out}")
      return

    status = data.get("status")

    if status == "complete":
      final_report = data.get("final_report")
      if final_report is None:
        print("API reported complete but no final_report field present", file=sys.stderr)
        sys.exit(4)
      with open(args.out, "w", encoding="utf-8") as f:
        f.write(final_report)
      print(f"Wrote report to: {args.out}")
      return

    if status == "need_clarification":
      question = data.get("message") or "The agent asked for clarification. Please reply:"
      print("Agent:")
      print(question)
      try:
        reply = input("You: ")
      except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
      # Append user's reply and continue the loop
      messages.append(reply)
      continue

    # Unknown response — print and exit
    print("Unexpected API response:", data, file=sys.stderr)
    sys.exit(5)


if __name__ == "__main__":
  main()
