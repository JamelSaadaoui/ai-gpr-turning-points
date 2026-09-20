"""Execute native notebook code cells in order, without a Jupyter dependency.

The notebook has no magics or kernel-specific state. This runner uses one fresh
Python namespace, preserves outputs/execution counts, and stops at the first
error. It is a sequential-cell execution check, not a Jupyter-kernel test.
"""

from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import argparse
import base64
import json
import os
import platform
import time
import traceback


@dataclass
class Image:
    filename: str


def execute_notebook(path: Path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__", "_REPLICATION_CELL_RUNNER": True}
    execution_count = 0
    started = time.perf_counter()
    original_directory = Path.cwd()
    os.chdir(path.parent)
    try:
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            cell["execution_count"] = execution_count
            cell["outputs"] = []
            buffer = StringIO()

            def flush():
                content = buffer.getvalue()
                if content:
                    cell["outputs"].append({"output_type": "stream", "name": "stdout", "text": content.splitlines(True)})
                    buffer.seek(0)
                    buffer.truncate(0)

            def display(value):
                flush()
                if isinstance(value, Image):
                    content = Path(value.filename).read_bytes()
                    output = {"image/png": base64.b64encode(content).decode("ascii"), "text/plain": [f"Image: {value.filename}"]}
                elif hasattr(value, "to_html"):
                    output = {"text/html": [value.to_html()], "text/plain": [str(value)]}
                else:
                    output = {"text/plain": [str(value)]}
                cell["outputs"].append({"output_type": "display_data", "metadata": {}, "data": output})

            namespace.update({"display": display, "Image": Image})
            print(f"Executing code cell {execution_count} (notebook position {index + 1})", flush=True)
            try:
                source = cell["source"]
                source = "".join(source) if isinstance(source, list) else source
                with redirect_stdout(buffer), redirect_stderr(buffer):
                    exec(compile(source, f"{path.name}:cell-{execution_count}", "exec"), namespace)
                flush()
            except Exception as error:
                flush()
                cell["outputs"].append({"output_type": "error", "ename": type(error).__name__,
                                         "evalue": str(error), "traceback": traceback.format_exc().splitlines()})
                path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
                raise
        elapsed = time.perf_counter() - started
        metadata = {"method": "Sequential code-cell execution in a fresh Python process; no Jupyter kernel",
                    "status": "passed", "code_cells": execution_count,
                    "elapsed_seconds": round(elapsed, 3), "python": platform.python_version(),
                    "completed_utc": datetime.now(timezone.utc).isoformat()}
        notebook["metadata"]["replication_execution"] = metadata
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
        report = path.parent / "results" / "notebook_execution.json"
        report.parent.mkdir(exist_ok=True)
        report.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Completed {execution_count} code cells in {elapsed:.1f} seconds.")
    finally:
        os.chdir(original_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", default="Geopolitical_Turning_Points_Replication.ipynb")
    args = parser.parse_args()
    target = Path(args.notebook).resolve()
    execute_notebook(target)
