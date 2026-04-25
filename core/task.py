"""
task representation. the unit of work that flows through the network.
"""

from dataclasses import dataclass, field
import time


@dataclass
class Task:
    task_id: str
    task_type: str
    description: str
    expected_answer: str = ""
    context: str = ""
    result: str = ""
    state: str = "pending"  # pending, completed, failed
    forward_history: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    @property
    def is_complete(self):
        return self.state in ("completed", "failed")

    @property
    def execution_time(self):
        if self.completed_at > 0:
            return self.completed_at - self.created_at
        return time.time() - self.created_at

    def add_to_history(self, agent_id):
        self.forward_history.append(agent_id)

    def has_visited(self, agent_id):
        return agent_id in self.forward_history

    def complete(self, result):
        self.result = result
        self.state = "completed"
        self.completed_at = time.time()

    def fail(self):
        self.state = "failed"
        self.completed_at = time.time()

    def check_answer(self):
        """check if result matches expected answer. flexible matching."""
        if not self.expected_answer:
            return None
        result = self._normalize(self.result)
        expected = self._normalize(self.expected_answer)
        if not result:
            return False
        # exact match after normalization
        if result == expected:
            return True
        # check if expected is contained in result (handles "The answer is X")
        if expected in result:
            return True
        # check if result is contained in expected
        if result in expected:
            return True
        return False

    @staticmethod
    def _normalize(text):
        """strip whitespace, parentheses, quotes, lowercase."""
        text = text.strip().lower()
        text = text.strip("()[]\"'")
        text = text.strip()
        return text
