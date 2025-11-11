"""FastAPI endpoints to run the Deep Research workflow without LangGraph Studio UI.

This module exposes a minimal HTTP API to run the compiled `deep_researcher`
StateGraph. It intentionally avoids LangGraph Studio/tracing and provides a
non-proprietary entrypoint for orchestration.

Endpoints:
- POST /run  -> run a research workflow given a `topic` (JSON) and return a
                Markdown report (either as text or as a downloadable file).
- GET  /health -> simple health check

Notes:
- The endpoint calls `deep_researcher.ainvoke` directly. The Configuration
  is read from environment variables (see `Configuration.from_runnable_config`).
"""

import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, FileResponse

from langchain_core.messages import HumanMessage, AIMessage

from open_deep_research.deep_researcher import deep_researcher

app = FastAPI(title="Open Deep Research API")


def _extract_final_report(result) -> Optional[str]:
    """Try several heuristics to extract the final_report string from the
    workflow result.
    """
    # case: direct dict with final_report
    try:
        if isinstance(result, dict) and "final_report" in result:
            return result.get("final_report")
    except Exception:
        pass

    # case: result has attribute or key-like access
    try:
        # common object types might expose .final_report
        final = getattr(result, "final_report", None)
        if final:
            return final
    except Exception:
        pass

    # case: if result is a list or nested, search heuristically
    try:
        if isinstance(result, (list, tuple)) and len(result) > 0:
            for item in result:
                fr = _extract_final_report(item)
                if fr:
                    return fr
    except Exception:
        pass

    return None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run")
async def run(payload: dict, out: Optional[str] = None):
    """Run the deep researcher workflow.

    Accepts JSON payload with either:
      - {"topic": "..."}
      - {"messages": ["msg1", "msg2", ...]}

    Returns JSON with status:
      - {"status": "need_clarification", "message": "..."}
      - {"status": "complete", "final_report": "..."}

    If `out` is supplied as a query param, the server will write and return
    a file with the markdown content.
    """
    topic = payload.get("topic") if isinstance(payload, dict) else None
    messages_list = payload.get("messages") if isinstance(payload, dict) else None

    if not topic and not messages_list:
        raise HTTPException(status_code=400, detail="Provide either `topic` or `messages` in JSON payload")

    # Build messages state
    if messages_list:
        msgs = [HumanMessage(content=str(m)) for m in messages_list]
    else:
        msgs = [HumanMessage(content=str(topic).strip())]

    input_state = {"messages": msgs}

    try:
        result = await deep_researcher.ainvoke(input_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {e}")

    # Try to extract a final report
    final_report = _extract_final_report(result)

    # If final report exists, optionally write file and return
    if final_report:
        if out:
            try:
                with open(out, "w", encoding="utf-8") as f:
                    f.write(final_report)
                return FileResponse(out, media_type="text/markdown", filename=out)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to write or return file: {e}")

        return {"status": "complete", "final_report": final_report}

    # No final report — attempt to extract conversational AI message (clarification)
    try:
        # If result is a dict with supervisor/agent messages, try to find AIMessage
        ai_text = None
        if isinstance(result, dict):
            messages = result.get("messages") or result.get("supervisor_messages") or []
            for m in messages:
                if isinstance(m, AIMessage):
                    ai_text = m.content
                    break
                # If it's a plain dict-like message
                try:
                    if isinstance(m, dict) and m.get("type") == "ai":
                        ai_text = m.get("content") or m.get("text")
                        break
                except Exception:
                    pass

        # If still nothing, convert top-level result to string and return as message
        if not ai_text:
            ai_text = str(result)

        return {"status": "need_clarification", "message": ai_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not interpret workflow result: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("open_deep_research.api:app", host="127.0.0.1", port=8000, reload=False)
