"""Jupyter notebook parser for security scanning."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CodeCell:
    cell_number: int
    source_code: str
    line_offset: int  # Approximate line number in the notebook JSON


@dataclass
class OutputCell:
    cell_number: int
    output_text: str
    output_type: str  # stream, display_data, execute_result, error


@dataclass
class NotebookContent:
    code_cells: List[CodeCell] = field(default_factory=list)
    output_cells: List[OutputCell] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    nbformat: int = 4


class NotebookParser:
    """Parses Jupyter notebooks and extracts code/output cells for analysis."""

    def parse(self, notebook_path: str) -> Optional[NotebookContent]:
        """Parse a notebook file and return its content."""
        try:
            with open(notebook_path, "r", encoding="utf-8", errors="replace") as f:
                raw = json.load(f)
        except (OSError, IOError) as e:
            logger.warning("Cannot open notebook %s: %s", notebook_path, e)
            return None
        except json.JSONDecodeError as e:
            logger.warning("Malformed notebook JSON in %s: %s", notebook_path, e)
            return None

        try:
            return self._parse_notebook(raw)
        except Exception as e:  # noqa: BLE001
            logger.warning("Error parsing notebook %s: %s", notebook_path, e)
            return None

    def _parse_notebook(self, raw: Dict[str, Any]) -> NotebookContent:
        nbformat = raw.get("nbformat", 4)
        metadata = raw.get("metadata", {})
        cells = raw.get("cells", [])

        # nbformat 3 uses "worksheets"
        if nbformat < 4:
            worksheets = raw.get("worksheets", [{}])
            cells = worksheets[0].get("cells", []) if worksheets else []

        content = NotebookContent(nbformat=nbformat, metadata=metadata)
        line_offset = 1

        for cell_num, cell in enumerate(cells, start=1):
            cell_type = cell.get("cell_type", "")
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)

            if cell_type == "code":
                content.code_cells.append(CodeCell(
                    cell_number=cell_num,
                    source_code=source,
                    line_offset=line_offset,
                ))
                # Extract outputs
                for output in cell.get("outputs", []):
                    output_type = output.get("output_type", "")
                    text = self._extract_output_text(output)
                    if text:
                        content.output_cells.append(OutputCell(
                            cell_number=cell_num,
                            output_text=text,
                            output_type=output_type,
                        ))

            line_offset += source.count("\n") + 1

        return content

    def extract_code_cells(self, notebook: Dict[str, Any]) -> List[CodeCell]:
        """Extract executable code cells from a raw notebook dict."""
        content = self._parse_notebook(notebook)
        return content.code_cells

    def extract_output_cells(self, notebook: Dict[str, Any]) -> List[OutputCell]:
        """Extract output cells from a raw notebook dict."""
        content = self._parse_notebook(notebook)
        return content.output_cells

    @staticmethod
    def _extract_output_text(output: Dict[str, Any]) -> str:
        """Extract text content from an output cell."""
        output_type = output.get("output_type", "")

        if output_type == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            return text

        if output_type in {"display_data", "execute_result"}:
            data = output.get("data", {})
            text = data.get("text/plain", "") or data.get("text/html", "")
            if isinstance(text, list):
                text = "".join(text)
            return text

        if output_type == "error":
            traceback = output.get("traceback", [])
            return "\n".join(traceback)

        return ""
