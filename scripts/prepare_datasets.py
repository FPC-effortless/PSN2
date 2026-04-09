"""
Prepare external datasets into unified JSONL format under data/.

Outputs:
  data/d5_gsm8k/train.jsonl        - GSM8K math problems
  data/d5_gsm8k/test.jsonl
  data/d6_bbh/train.jsonl          - BIG-Bench Hard (all 27 tasks merged)
  data/d5_arc_agi2/train.jsonl     - ARC-AGI-2 training tasks
  data/d5_arc_agi2/eval.jsonl      - ARC-AGI-2 evaluation tasks

Each record has a consistent schema:
  { "task": str, "input": any, "target": any, "split": str, "source": str }
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SRC_BASE = Path(r"C:\Users\user\Documents\SJS Programs\psn2_kaggle_full_repo\data")
OUT_BASE = Path("data")


# ── helpers ──────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):>6,} rows  →  {path}")


# ── GSM8K ────────────────────────────────────────────────────────────────────

def prepare_gsm8k() -> None:
    print("\n[GSM8K] grade-school-math")
    src = SRC_BASE / "grade-school-math-master" / "grade_school_math" / "data"

    for split_file in ["train.jsonl", "test.jsonl"]:
        raw = (src / split_file).read_text(encoding="utf-8").strip().splitlines()
        records = []
        for line in raw:
            obj = json.loads(line)
            # Extract final numeric answer after ####
            answer_text = obj["answer"]
            match = re.search(r"####\s*(.+)$", answer_text, re.MULTILINE)
            final_answer = match.group(1).strip() if match else answer_text.strip()
            records.append({
                "task": "math_word_problem",
                "input": obj["question"],
                "target": final_answer,
                "chain_of_thought": answer_text,
                "split": split_file.replace(".jsonl", ""),
                "source": "gsm8k",
            })
        split_name = split_file.replace(".jsonl", "")
        write_jsonl(OUT_BASE / "d5_gsm8k" / f"{split_name}.jsonl", records)


# ── BIG-Bench Hard ────────────────────────────────────────────────────────────

def prepare_bbh() -> None:
    print("\n[BBH] BIG-Bench Hard")
    src = SRC_BASE / "BIG-Bench-Hard-main" / "bbh"
    all_records: list[dict] = []

    for task_file in sorted(src.glob("*.json")):
        task_name = task_file.stem
        data = json.loads(task_file.read_text(encoding="utf-8"))
        examples = data.get("examples", [])
        for ex in examples:
            all_records.append({
                "task": task_name,
                "input": ex["input"],
                "target": ex["target"],
                "split": "test",   # BBH is an eval benchmark — all examples are test
                "source": "bbh",
            })

    write_jsonl(OUT_BASE / "d6_bbh" / "test.jsonl", all_records)


# ── ARC-AGI-2 ────────────────────────────────────────────────────────────────

def load_arc_split(split_dir: Path, split_name: str) -> list[dict]:
    records = []
    for task_file in sorted(split_dir.glob("*.json")):
        task_id = task_file.stem
        data = json.loads(task_file.read_text(encoding="utf-8"))
        records.append({
            "task": "arc_agi2",
            "task_id": task_id,
            "train_pairs": data["train"],   # list of {input, output} demonstration pairs
            "test_input": data["test"][0]["input"],
            "target": data["test"][0]["output"],
            "split": split_name,
            "source": "arc_agi2",
        })
    return records


def prepare_arc_agi2() -> None:
    print("\n[ARC-AGI-2]")
    src = SRC_BASE / "ARC-AGI-2-main" / "data"

    train_records = load_arc_split(src / "training", "train")
    write_jsonl(OUT_BASE / "d5_arc_agi2" / "train.jsonl", train_records)

    eval_records = load_arc_split(src / "evaluation", "eval")
    write_jsonl(OUT_BASE / "d5_arc_agi2" / "eval.jsonl", eval_records)


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    prepare_gsm8k()
    prepare_bbh()
    prepare_arc_agi2()
    print("\nAll datasets prepared.")
