from __future__ import annotations

import json

from app.api.dependencies import get_query_agent, get_vector_store

TEST_QUERIES = [
    "What is the total number of jobs reported, and where exactly is this stated?",
    "Compare the concentration of Pure-Play cybersecurity firms in the South-West against the national average.",
    "Based on our 2022 baseline and the stated 2030 job target, what is the required compound annual growth rate (CAGR) to hit that goal?",
]


def main() -> None:
    vector_store = get_vector_store()
    if vector_store.count() == 0:
        raise RuntimeError("Index is empty. Run 'python -m scripts.ingest_report' first.")

    agent = get_query_agent()

    for index, query in enumerate(TEST_QUERIES, start=1):
        result = agent.answer(query)
        payload = {
            "query": query,
            "answer": result.answer,
            "citations": [citation.model_dump() for citation in result.citations],
            "reasoning_steps": [step.model_dump(mode="json") for step in result.reasoning_steps],
        }
        print(f"\n=== TEST {index} ===")
        print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
