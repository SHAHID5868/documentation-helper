"""Core backend interfaces used by the Streamlit app."""

from typing import Any, Dict


def run_llm(prompt: str, **kwargs: Any) -> Dict[str, Any]:
    """Temporary LLM entrypoint for UI wiring.

    Replace this implementation with the real model call pipeline.
    """
    return {
        "answer": "run_llm is not implemented yet.",
        "context_docs": [],
        "meta": {"prompt": prompt, "kwargs": kwargs},
    }
