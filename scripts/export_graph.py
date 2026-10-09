"""Write the compiled LangGraph workflow as a Mermaid diagram.

Usage (from the project root):  python scripts/export_graph.py
Output: docs/langgraph_workflow.mmd (paste into https://mermaid.live or view on GitHub)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.graph.nodes import WorkflowNodes  # noqa: E402
from app.graph.workflow import QAWorkflow  # noqa: E402


def main() -> None:
    settings = Settings()
    # Building the graph only needs the node functions, not live services.
    workflow = QAWorkflow(
        WorkflowNodes(retrieval=None, answers=None),  # type: ignore[arg-type]
        max_retries=settings.max_retries,
        max_grounding_retries=settings.max_grounding_retries,
    )
    target = ROOT / "docs" / "langgraph_workflow.mmd"
    target.write_text(workflow.mermaid(), encoding="utf-8")
    print(f"Wrote {target.relative_to(ROOT)}")
    print(workflow.mermaid())


if __name__ == "__main__":
    main()
