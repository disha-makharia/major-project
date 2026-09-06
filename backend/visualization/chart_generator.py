"""Visualization Agent support: decide whether a chart helps, then render it.

Charts are only ever built from the actual SQL result rows returned by the
SQL Agent - never fabricated - and saved to generated/charts/.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display needed
import matplotlib.pyplot as plt

from backend.config import settings
from backend.models.schemas import VisualizationResult
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_DATE_HINTS = re.compile(r"(date|month|year|quarter|period|day|time)", re.IGNORECASE)
_PIE_HINTS = re.compile(r"(share|percentage|percent|proportion|distribution|breakdown)", re.IGNORECASE)


def _is_numeric(values: list[Any]) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values if v is not None)


def _column_kind(name: str, values: list[Any]) -> str:
    if _DATE_HINTS.search(name):
        return "date"
    if _is_numeric(values):
        return "numeric"
    return "categorical"


def decide_chart(question: str, columns: list[str], rows: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Returns (chart_type, reason_if_none)."""
    if not rows or not columns:
        return None, "No data rows to visualize."
    if len(rows) == 1 and len(columns) <= 2:
        return None, "Result is a single aggregate value; a chart would not add insight."

    kinds = {col: _column_kind(col, [r.get(col) for r in rows]) for col in columns}
    numeric_cols = [c for c, k in kinds.items() if k == "numeric"]
    categorical_cols = [c for c, k in kinds.items() if k == "categorical"]
    date_cols = [c for c, k in kinds.items() if k == "date"]

    if not numeric_cols:
        return None, "Result has no numeric column to plot."

    if date_cols and numeric_cols:
        return "line", None

    if categorical_cols and numeric_cols:
        n_categories = len(rows)
        if _PIE_HINTS.search(question) and 2 <= n_categories <= 8:
            return "pie", None
        if n_categories <= 50:
            return "bar", None
        return None, "Too many categories to render a readable chart."

    if len(numeric_cols) >= 2:
        return "scatter", None

    if len(numeric_cols) == 1 and len(rows) > 5:
        return "histogram", None

    return None, "No suitable chart type identified for this result shape."


def render_chart(
    chart_type: str,
    question: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    chart_dir: str | None = None,
) -> VisualizationResult:
    chart_dir_path = Path(chart_dir or settings.chart_dir)
    chart_dir_path.mkdir(parents=True, exist_ok=True)

    kinds = {col: _column_kind(col, [r.get(col) for r in rows]) for col in columns}
    numeric_cols = [c for c, k in kinds.items() if k == "numeric"]
    categorical_cols = [c for c, k in kinds.items() if k in ("categorical", "date")]

    fig, ax = plt.subplots(figsize=(8, 5))
    try:
        if chart_type == "bar":
            cat_col, val_col = categorical_cols[0], numeric_cols[0]
            labels = [str(r.get(cat_col)) for r in rows]
            values = [r.get(val_col) or 0 for r in rows]
            ax.bar(labels, values, color="#4C72B0")
            ax.set_xlabel(cat_col)
            ax.set_ylabel(val_col)
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "line":
            x_col = categorical_cols[0] if categorical_cols else columns[0]
            y_col = numeric_cols[0]
            labels = [str(r.get(x_col)) for r in rows]
            values = [r.get(y_col) or 0 for r in rows]
            ax.plot(labels, values, marker="o", color="#4C72B0")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.xticks(rotation=45, ha="right")
        elif chart_type == "pie":
            cat_col, val_col = categorical_cols[0], numeric_cols[0]
            labels = [str(r.get(cat_col)) for r in rows]
            values = [max(r.get(val_col) or 0, 0) for r in rows]
            ax.pie(values, labels=labels, autopct="%1.1f%%")
        elif chart_type == "scatter":
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            xs = [r.get(x_col) for r in rows]
            ys = [r.get(y_col) for r in rows]
            ax.scatter(xs, ys, color="#4C72B0")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
        elif chart_type == "histogram":
            val_col = numeric_cols[0]
            values = [r.get(val_col) for r in rows if r.get(val_col) is not None]
            ax.hist(values, bins=min(20, max(5, len(values) // 2)), color="#4C72B0")
            ax.set_xlabel(val_col)
            ax.set_ylabel("Frequency")
        else:
            plt.close(fig)
            return VisualizationResult(generated=False, reason=f"Unsupported chart type '{chart_type}'.")

        ax.set_title(question[:80])
        fig.tight_layout()

        filename = f"chart_{uuid.uuid4().hex[:12]}.png"
        out_path = chart_dir_path / filename
        fig.savefig(out_path, dpi=120)
        logger.info("chart_generated", extra={"extra_fields": {"type": chart_type, "file": filename}})
        return VisualizationResult(
            generated=True,
            chart_type=chart_type,
            chart_path=str(out_path),
            chart_url=f"/charts/{filename}",
        )
    except Exception as exc:
        logger.info("chart_generation_failed", extra={"extra_fields": {"error": str(exc)}})
        return VisualizationResult(generated=False, reason=f"Chart generation failed: {exc}")
    finally:
        plt.close(fig)
