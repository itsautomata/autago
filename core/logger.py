"""
detailed logger. captures every agent decision, reasoning, and state change.
writes to both terminal (summary) and file (full detail).
"""

import json
import time
from pathlib import Path


# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"

AGENT_COLORS = {
    0: CYAN,
    1: YELLOW,
    2: MAGENTA,
    3: "\033[36m",
    4: "\033[35m",
    5: "\033[97m",
    6: "\033[34m",
}


def _color(agent_id):
    return AGENT_COLORS.get(agent_id, "\033[97m")


class Logger:
    def __init__(self, log_dir=None, verbose=True):
        self.verbose = verbose
        self.entries = []
        self.log_file = None

        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_file = open(log_dir / "detailed.log", "w")

    def close(self):
        if self.log_file:
            self.log_file.close()

    def _write(self, text):
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()

    def task_start(self, task, initial_agent_id):
        entry = {
            "event": "task_start",
            "task_id": task.task_id,
            "task_type": task.task_type,
            "description": task.description[:100],
            "initial_agent": initial_agent_id,
            "time": time.time(),
        }
        self.entries.append(entry)

        self._write(f"\n{'='*60}")
        self._write(f"TASK {task.task_id} [{task.task_type}]")
        self._write(f"  {task.description[:200]}")
        self._write(f"  initial agent: {initial_agent_id}")

        if self.verbose:
            print(f"\n  {DIM}task {task.task_id} [{task.task_type}]{RESET}")

    def routing_decision(self, agent_id, task, decision, reasoning, capabilities):
        entry = {
            "event": "routing_decision",
            "agent": agent_id,
            "task_id": task.task_id,
            "decision": decision,
            "reasoning": reasoning,
            "capabilities": capabilities,
        }
        self.entries.append(entry)

        self._write(f"  agent {agent_id} decides: {decision}")
        self._write(f"    reasoning: {reasoning}")
        self._write(f"    capabilities: {json.dumps(capabilities, default=str)}")

        if self.verbose:
            c = _color(agent_id)
            symbol = {"EXECUTE": ">>", "FORWARD": "->", "SPLIT": "<>"}
            print(f"  {c}agent {agent_id}{RESET} {symbol.get(decision, '??')} {DIM}{decision}: {reasoning[:80]}{RESET}")

    def forward(self, source_id, target_id, task, score=None):
        entry = {
            "event": "forward",
            "source": source_id,
            "target": target_id,
            "task_id": task.task_id,
            "score": score,
        }
        self.entries.append(entry)

        self._write(f"  forward: agent {source_id} -> agent {target_id}")

        if self.verbose:
            sc = _color(source_id)
            tc = _color(target_id)
            print(f"  {sc}agent {source_id}{RESET} -> {tc}agent {target_id}{RESET}")

    def execution(self, agent_id, task, answer, reasoning):
        entry = {
            "event": "execution",
            "agent": agent_id,
            "task_id": task.task_id,
            "answer": answer,
            "reasoning": reasoning[:200],
        }
        self.entries.append(entry)

        self._write(f"  agent {agent_id} executes:")
        self._write(f"    answer: {answer}")
        self._write(f"    reasoning: {reasoning[:200]}")

    def task_result(self, task, path, success):
        entry = {
            "event": "task_result",
            "task_id": task.task_id,
            "task_type": task.task_type,
            "path": path,
            "success": success,
            "result": task.result,
            "expected": task.expected_answer,
            "execution_time": task.execution_time,
        }
        self.entries.append(entry)

        status_str = "CORRECT" if success else "WRONG"
        self._write(f"  result: {status_str}")
        self._write(f"    got: {task.result}")
        self._write(f"    expected: {task.expected_answer}")
        self._write(f"    path: {path}")
        self._write(f"    time: {task.execution_time:.1f}s")

        if self.verbose:
            if success:
                print(f"  {GREEN}{BOLD}correct{RESET} {DIM}path={path}{RESET}")
            else:
                print(f"  {RED}{BOLD}wrong{RESET} got={task.result[:40]} expected={task.expected_answer[:40]} {DIM}path={path}{RESET}")

    def ability_update(self, agent_id, task_type, old_abilities, new_abilities, success):
        changes = {}
        for k in old_abilities:
            if abs(old_abilities[k] - new_abilities[k]) > 0.001:
                changes[k] = f"{old_abilities[k]:.2f} -> {new_abilities[k]:.2f}"

        if not changes:
            return

        entry = {
            "event": "ability_update",
            "agent": agent_id,
            "task_type": task_type,
            "success": success,
            "changes": changes,
        }
        self.entries.append(entry)

        self._write(f"  agent {agent_id} abilities updated ({task_type}, {'success' if success else 'failure'}):")
        for k, v in changes.items():
            self._write(f"    {k}: {v}")

        if self.verbose:
            c = _color(agent_id)
            change_str = ", ".join(f"{k}: {v}" for k, v in changes.items())
            print(f"  {c}agent {agent_id}{RESET} {DIM}{change_str}{RESET}")

    def edge_update(self, source_id, target_id, old_weight, new_weight, pruned=False):
        entry = {
            "event": "edge_update",
            "source": source_id,
            "target": target_id,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "pruned": pruned,
        }
        self.entries.append(entry)

        if pruned:
            self._write(f"  edge {source_id}->{target_id} PRUNED (weight {old_weight:.3f} -> {new_weight:.3f})")
            if self.verbose:
                print(f"  {RED}edge {source_id}->{target_id} pruned{RESET}")
        else:
            self._write(f"  edge {source_id}->{target_id}: {old_weight:.3f} -> {new_weight:.3f}")

    def phase_summary(self, phase_name, metrics, agents):
        self._write(f"\n{'='*60}")
        self._write(f"{phase_name.upper()} SUMMARY")
        self._write(f"  accuracy: {metrics.accuracy:.1%}")
        self._write(f"  avg hops: {metrics.avg_hops:.2f}")
        self._write(f"  specialization: {metrics.specialization_depth(agents):.4f}")
        for aid, agent in agents.items():
            self._write(f"  agent {aid}: {json.dumps(agent.abilities, default=str)}")
        self._write(f"{'='*60}\n")

    def save_entries(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.entries, indent=2, default=str))