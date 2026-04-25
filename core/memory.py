"""
RAG memory pools. agents remember past tasks and use them to inform decisions.

two pools per agent:
  - router pool: past routing decisions and outcomes
  - executor pool: past execution results and approaches
"""

from . import embedding


class Experience:
    """a single memory entry."""

    def __init__(self, task_type, description, result, success, execution_time=0, extra=None):
        self.task_type = task_type
        self.description = description
        self.result = result
        self.success = success
        self.execution_time = execution_time
        self.extra = extra or {}
        self.text = f"[{task_type}] {description}"

    def to_dict(self):
        return {
            "task_type": self.task_type,
            "description": self.description,
            "result": self.result,
            "success": self.success,
            "execution_time": self.execution_time,
            "extra": self.extra,
        }


class MemoryPool:
    """experience pool with embedding-based retrieval and eviction."""

    def __init__(self, limit=40, retrieval_top_k=3, retrieval_threshold=0.7):
        self.limit = limit
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_threshold = retrieval_threshold
        self.experiences = []

    @property
    def is_full(self):
        return self.limit > 0 and len(self.experiences) >= self.limit

    def add(self, experience):
        """add an experience. evict if at capacity."""
        if self.is_full:
            self._evict(experience)
        else:
            self.experiences.append(experience)

    def retrieve(self, query_text):
        """find similar past experiences by embedding similarity."""
        if not self.experiences:
            return []

        candidates = [{"text": e.text, "experience": e} for e in self.experiences]
        results = embedding.find_similar(
            query_text, candidates,
            top_k=self.retrieval_top_k,
            threshold=self.retrieval_threshold,
        )
        return [(r[0]["experience"], r[1]) for r in results]

    def _evict(self, new_experience):
        """evict the least diverse entry to make room.

        removes the experience whose embedding is most similar to another
        existing experience (least unique). keeps memory diverse.
        """
        if not self.experiences:
            self.experiences.append(new_experience)
            return

        # find the pair with highest similarity (least diverse)
        all_entries = self.experiences + [new_experience]
        max_sim = -1
        evict_idx = 0

        for i, entry in enumerate(all_entries):
            for j, other in enumerate(all_entries):
                if i >= j:
                    continue
                sim = embedding.similarity(entry.text, other.text)
                if sim > max_sim:
                    max_sim = sim
                    # evict the one from the most-similar pair that has worse success rate
                    if entry.success and not other.success:
                        evict_idx = j
                    elif other.success and not entry.success:
                        evict_idx = i
                    else:
                        # both same success status: evict the older one (lower index)
                        evict_idx = i

        if evict_idx < len(self.experiences):
            self.experiences.pop(evict_idx)
            self.experiences.append(new_experience)
        # else: new_experience is the eviction target, don't add it

    def format_for_prompt(self, query_text):
        """retrieve similar experiences and format them for injection into prompts."""
        similar = self.retrieve(query_text)
        if not similar:
            return ""

        lines = ["SIMILAR PAST EXPERIENCES:"]
        for exp, score in similar:
            status = "succeeded" if exp.success else "failed"
            lines.append(f"  [{exp.task_type}] {exp.description[:80]} -> {status}, answer: {exp.result[:50]}")

        return "\n".join(lines)

    def summary(self):
        return {
            "size": len(self.experiences),
            "limit": self.limit,
            "success_count": sum(1 for e in self.experiences if e.success),
            "failure_count": sum(1 for e in self.experiences if not e.success),
        }
