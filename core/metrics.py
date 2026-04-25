"""
metrics. measures what matters: accuracy, specialization, routing efficiency.
"""

import json
from pathlib import Path


class Metrics:
    def __init__(self):
        self.records = []
        self.correct = 0
        self.total = 0
        self.memory_hits = 0

    def record(self, task, path, success, memory_used=False):
        self.total += 1
        if success:
            self.correct += 1
        if memory_used:
            self.memory_hits += 1

        self.records.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "path": path,
            "hops": len(path),
            "success": success,
            "result": task.result,
            "expected": task.expected_answer,
            "execution_time": task.execution_time,
            "memory_used": memory_used,
        })

    @property
    def accuracy(self):
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def avg_hops(self):
        if not self.records:
            return 0.0
        return sum(r["hops"] for r in self.records) / len(self.records)

    def accuracy_by_type(self):
        """accuracy broken down by task type."""
        by_type = {}
        for r in self.records:
            tt = r["task_type"]
            if tt not in by_type:
                by_type[tt] = {"correct": 0, "total": 0}
            by_type[tt]["total"] += 1
            if r["success"]:
                by_type[tt]["correct"] += 1

        return {
            tt: d["correct"] / d["total"] if d["total"] > 0 else 0.0
            for tt, d in sorted(by_type.items())
        }

    def specialization_depth(self, agents):
        """how much agent capability vectors diverge from each other.
        higher = more specialized."""
        if len(agents) < 2:
            return 0.0

        abilities_list = [list(a.abilities.values()) for a in agents.values()]
        n = len(abilities_list)
        total_variance = 0.0
        dims = len(abilities_list[0])

        for d in range(dims):
            values = [a[d] for a in abilities_list]
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n
            total_variance += variance

        return total_variance / dims

    def routing_patterns(self):
        """which agents execute most, which forward most."""
        executor_counts = {}
        for r in self.records:
            if r["path"]:
                executor = r["path"][-1]
                executor_counts[executor] = executor_counts.get(executor, 0) + 1
        return executor_counts

    @property
    def memory_usage(self):
        return self.memory_hits / self.total if self.total > 0 else 0.0

    def summary(self):
        return {
            "accuracy": f"{self.accuracy:.1%}",
            "total_tasks": self.total,
            "correct": self.correct,
            "avg_hops": f"{self.avg_hops:.2f}",
            "memory_usage": f"{self.memory_usage:.1%}",
            "memory_hits": self.memory_hits,
        }

    def full_report(self, agents=None):
        report = self.summary()
        report["accuracy_by_type"] = self.accuracy_by_type()
        report["routing_patterns"] = self.routing_patterns()
        if agents:
            report["specialization_depth"] = f"{self.specialization_depth(agents):.4f}"
            report["agent_abilities"] = {
                aid: dict(a.abilities) for aid, a in agents.items()
            }
        return report

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.full_report(), indent=2, default=str))
