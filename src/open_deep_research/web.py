"""Lightweight Flask web front end for generating Markdown reports.

This provides a minimal non-proprietary browser UI. It accepts a topic and
returns the generated Markdown report. For production use, run behind a WSGI
server or add authentication as needed.
"""

from flask import Flask, request, render_template_string, send_file, abort
import json
import httpx
import os

APP = Flask(__name__)

# The FastAPI endpoint that performs orchestration and multi-turn conversation.
# Default to localhost:8000 but allow override via OPEN_DEEP_RESEARCH_API_URL env var.
API_URL = os.environ.get("OPEN_DEEP_RESEARCH_API_URL", "http://127.0.0.1:8000/run")

FORM_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Open Deep Research — Report Generator</title>
  </head>
  <body>
    <h1>Open Deep Research — Report Generator</h1>
    <form method="post">
      <label for="topic">Your message / reply</label><br/>
      <textarea id="topic" name="topic" rows="6" cols="80">{{ initial }}</textarea><br/>
      <input type="hidden" id="messages" name="messages" value='{{ messages | tojson | safe }}' />
      <label for="fname">Output filename (optional)</label>
      <input id="fname" name="fname" value="report.md" />
      <button type="submit">Send</button>
    </form>
    {% if agent_message %}
      <h2>Agent</h2>
      <div style="white-space:pre-wrap;border:1px solid #ddd;padding:10px;margin:10px 0">{{ agent_message }}</div>
    {% endif %}
    {% if final_report %}
      <h2>Final Report</h2>
      <div style="white-space:pre-wrap;border:1px solid #888;padding:10px;margin:10px 0">{{ final_report }}</div>
      <a href="/download?path={{ out_name }}">Download report</a>
    {% endif %}
  </body>
</html>
"""


@APP.route("/", methods=["GET", "POST"])
def index():
    # messages are kept as a JSON array of strings representing the conversation history
    if request.method == "GET":
        # initial page — no messages
        return render_template_string(FORM_HTML, initial="", messages=[], agent_message=None, final_report=None, out_name="report.md")

    # POST: either first message (topic) or a reply in an ongoing conversation
    topic = request.form.get("topic", "").strip()
    messages_raw = request.form.get("messages", "[]")
    out_name = request.form.get("fname") or "report.md"

    try:
        messages = json.loads(messages_raw)
        if not isinstance(messages, list):
            messages = []
    except Exception:
        messages = []

    if topic:
        messages.append(topic)

    if not messages:
        return render_template_string(FORM_HTML, initial="", messages=[], agent_message="Please provide a topic.", final_report=None, out_name=out_name)

    # Call the FastAPI orchestration endpoint which runs the full LangGraph flow.
    try:
        resp = httpx.post(API_URL, json={"messages": messages}, timeout=120.0)
    except Exception as e:
        return f"Error contacting orchestration API at {API_URL}: {e}", 500

    if resp.status_code != 200:
        return f"Orchestration API returned status {resp.status_code}: {resp.text}", 500

    # Expect JSON response with 'status'
    try:
        data = resp.json()
    except Exception:
        # fallback: raw markdown
        md = resp.text
        # write file and return download
        try:
            with open(out_name, "w", encoding="utf-8") as f:
                f.write(md)
            return send_file(out_name, as_attachment=True)
        except Exception as e:
            abort(500, f"Could not save or return raw report: {e}")

    status = data.get("status")

    if status == "need_clarification":
        agent_message = data.get("message")
        # render the form with updated messages and the agent question
        return render_template_string(FORM_HTML, initial="", messages=messages, agent_message=agent_message, final_report=None, out_name=out_name)

    if status == "complete":
        final_report = data.get("final_report")
        # Optionally save the file to disk for download
        try:
            with open(out_name, "w", encoding="utf-8") as f:
                f.write(final_report or "")
        except Exception as e:
            return f"Failed to write final report to {out_name}: {e}", 500

        return render_template_string(FORM_HTML, initial="", messages=messages, agent_message=None, final_report=final_report, out_name=out_name)

    return f"Unexpected response from orchestration API: {data}", 500


@APP.route("/download")
def download():
    path = request.args.get("path")
    if not path:
        abort(400, "path param required")
    try:
        return send_file(path, as_attachment=True)
    except Exception as e:
        abort(500, f"Could not return file: {e}")


def run_server(host: str = "127.0.0.1", port: int = 5000):
    APP.run(host=host, port=port)


if __name__ == "__main__":
    run_server()
