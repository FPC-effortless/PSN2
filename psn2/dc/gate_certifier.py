"""Gate Certifier — blocks stage progression until all gates pass."""
from __future__ import annotations

from typing import Dict, Callable, Optional


class GateCertifier:
    """
    Evaluates gate conditions for a developmental stage.
    Blocks progression until all gates pass.
    """

    def __init__(self, stage_name: str, gates: Dict[str, Callable[[], float]],
                 thresholds: Dict[str, float], comparators: Dict[str, str] = None):
        """
        gates: {gate_name: callable returning current metric value}
        thresholds: {gate_name: threshold value}
        comparators: {gate_name: '>=' or '<'} (default '>=' for all)
        """
        self.stage_name = stage_name
        self.gates = gates
        self.thresholds = thresholds
        self.comparators = comparators or {k: ">=" for k in gates}
        self._certified = False

    def evaluate(self) -> Dict[str, bool]:
        results = {}
        for name, fn in self.gates.items():
            val = fn()
            thresh = self.thresholds[name]
            cmp = self.comparators.get(name, ">=")
            if cmp == ">=":
                results[name] = val >= thresh
            elif cmp == ">":
                results[name] = val > thresh
            elif cmp == "<":
                results[name] = val < thresh
            elif cmp == "<=":
                results[name] = val <= thresh
            else:
                results[name] = val >= thresh
        return results

    def is_certified(self) -> bool:
        if self._certified:
            return True
        results = self.evaluate()
        self._certified = all(results.values())
        return self._certified

    def report(self) -> str:
        results = self.evaluate()
        lines = [f"Stage {self.stage_name} gate report:"]
        for name, passed in results.items():
            val = self.gates[name]()
            thresh = self.thresholds[name]
            cmp = self.comparators.get(name, ">=")
            status = "PASS" if passed else "FAIL"
            lines.append(f"  [{status}] {name}: {val:.4f} {cmp} {thresh}")
        return "\n".join(lines)
