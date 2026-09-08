"""Read the VitisAI EP's observed placement report without creating a session."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass
class EPReport:
    path: Path
    raw: dict
    sha256: str

    def summary(self) -> dict:
        nodes = self.raw["nodeStat"]
        counts = Counter(n["device"] for n in nodes)
        ops = Counter((n["device"], n["opType"]) for n in nodes)
        return {"report_sha256": self.sha256, "total": len(nodes),
                "npu": counts.get("NPU", 0), "device_counts": dict(counts),
                "ops_by_device": {f"{dev}:{op}": count for (dev, op), count in sorted(ops.items())},
                "non_npu_count": len(nodes) - counts.get("NPU", 0),
                "non_npu_first_25": [{k: n.get(k) for k in ("device", "opType", "input", "output")}
                                     for n in nodes if n["device"] != "NPU"][:25],
                "deviceStat": self.raw["deviceStat"]}


def read_report(path: Path) -> EPReport:
    path = Path(path)
    content = path.read_bytes()
    raw = json.loads(content)
    if not isinstance(raw.get("nodeStat"), list) or not isinstance(raw.get("deviceStat"), list):
        raise ValueError(f"Missing placement lists in {path}")
    if any(not n.get("device") or not n.get("opType") for n in raw["nodeStat"]):
        raise ValueError(f"Incomplete per-node placement in {path}")
    actual_npu = sum(n["device"] == "NPU" for n in raw["nodeStat"])
    devices = {d["name"]: d for d in raw["deviceStat"]}
    if devices.get("NPU", {}).get("nodeNum", 0) != actual_npu:
        raise ValueError(f"NPU totals disagree inside {path}")
    if "all" in devices and devices["all"]["nodeNum"] != len(raw["nodeStat"]):
        raise ValueError(f"Total node counts disagree inside {path}")
    return EPReport(path, raw, hashlib.sha256(content).hexdigest())
