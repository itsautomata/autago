"""
the agent. capability vectors, routing, execution, evolution.
"""

import re

from . import llm
from . import prompts


class Agent:
    def __init__(self, agent_id, config):
        self.id = agent_id
        self.config = config

        agent_cfg = config["agents"]
        self.abilities = {a: agent_cfg["initial_ability"] for a in agent_cfg["abilities"]}
        self.ability_bounds = agent_cfg["ability_bounds"]
        self.decay_rate = agent_cfg["decay_rate"]
        self.decay_interval = agent_cfg["decay_interval"]

        self.success_rates = {}  # {target_agent_id: float}
        self.load = 0
        self.tasks_processed = 0

        # history for metrics
        self.ability_history = []
        self.decisions = []

    def _clamp_ability(self, value):
        return max(self.ability_bounds[0], min(self.ability_bounds[1], value))

    def ability_score(self, task_type):
        """average ability across dimensions relevant to this task type."""
        relevant = TASK_ABILITY_MAP.get(task_type, list(self.abilities.keys()))
        if not relevant:
            return sum(self.abilities.values()) / len(self.abilities)
        return sum(self.abilities.get(a, 0.6) for a in relevant) / len(relevant)

    def capabilities_text(self):
        """format capabilities for prompt injection."""
        lines = [f"  {k}: {v:.2f}" for k, v in sorted(self.abilities.items())]
        return "\n".join(lines)

    def decide_action(self, task, max_forwards):
        """router: decide EXECUTE, FORWARD, or SPLIT."""
        self.load += 1
        forwards_remaining = max_forwards - len(task.forward_history)

        system = prompts.ROUTER_SYSTEM.format(max_forwards=max_forwards)
        user = prompts.ROUTER_DECIDE.format(
            task_type=task.task_type,
            description=task.description,
            context=task.context or "(none)",
            capabilities=self.capabilities_text(),
            forward_history=", ".join(str(a) for a in task.forward_history) or "(none)",
            forwards_remaining=forwards_remaining,
            similar_experiences="",  # TODO: RAG memory
        )

        response = llm.call(system, [{"role": "user", "content": user}])
        decision = self._parse_decision(response)

        # can't forward if no forwards remaining
        if decision == "FORWARD" and forwards_remaining <= 0:
            decision = "EXECUTE"

        self.decisions.append({
            "task_id": task.task_id,
            "decision": decision,
            "response": response,
        })
        return decision

    def pick_next_agent(self, task, candidates):
        """router: pick which agent to forward to."""
        candidates_text = "\n".join(
            f"  agent {a.id}: abilities = {a.capabilities_text()}, "
            f"load = {a.load}, success_rate = {self.success_rates.get(a.id, 0.0):.2f}"
            for a in candidates
        )
        system = "you are a routing agent. pick the best agent for this task."
        user = prompts.ROUTER_PICK_AGENT.format(
            task_type=task.task_type,
            description=task.description,
            candidates=candidates_text,
        )

        response = llm.call(system, [{"role": "user", "content": user}])
        agent_id = self._parse_agent_id(response)

        # fallback: if parsing fails, pick first candidate
        if agent_id is None or not any(a.id == agent_id for a in candidates):
            agent_id = candidates[0].id

        return agent_id

    def execute(self, task):
        """executor: solve the task."""
        system = prompts.EXECUTOR_SYSTEM
        user = prompts.EXECUTOR_SOLVE.format(
            task_type=task.task_type,
            description=task.description,
            context=task.context or "(none)",
            similar_experiences="",  # TODO: RAG memory
        )

        response = llm.call(system, [{"role": "user", "content": user}])
        answer = self._parse_answer(response)

        self.load = max(0, self.load - 1)
        return answer

    def update_abilities(self, task_type, success):
        """update capability vector after a task."""
        self.tasks_processed += 1
        relevant = TASK_ABILITY_MAP.get(task_type, list(self.abilities.keys()))

        if success:
            for ability in relevant:
                if ability in self.abilities:
                    self.abilities[ability] = self._clamp_ability(
                        self.abilities[ability] + 0.1
                    )
            # correlated abilities get a smaller boost
            for ability in self.abilities:
                if ability not in relevant:
                    for rel in relevant:
                        corr = ABILITY_CORRELATIONS.get((rel, ability), 0.0)
                        if corr > 0.3:
                            self.abilities[ability] = self._clamp_ability(
                                self.abilities[ability] + 0.1 * corr * 0.5
                            )

        # periodic decay
        if self.tasks_processed % self.decay_interval == 0:
            for ability in self.abilities:
                self.abilities[ability] = self._clamp_ability(
                    self.abilities[ability] * (1 - self.decay_rate)
                )

        self.ability_history.append(dict(self.abilities))

    def update_success_rate(self, target_agent_id, success):
        """update EMA success rate for a target agent."""
        current = self.success_rates.get(target_agent_id, 0.0)
        self.success_rates[target_agent_id] = current * 0.9 + (1.0 if success else 0.0) * 0.1

    def summary(self):
        return {
            "id": self.id,
            "abilities": dict(self.abilities),
            "tasks_processed": self.tasks_processed,
            "load": self.load,
            "success_rates": dict(self.success_rates),
        }

    def _parse_decision(self, response):
        for line in response.strip().split("\n"):
            upper = line.strip().upper()
            if upper.startswith("DECISION:"):
                val = upper.split(":", 1)[1].strip()
                if val in ("EXECUTE", "FORWARD", "SPLIT"):
                    return val
        return "EXECUTE"  # safe fallback

    def _parse_agent_id(self, response):
        for line in response.strip().split("\n"):
            if line.strip().upper().startswith("AGENT:"):
                val = line.split(":", 1)[1].strip()
                match = re.search(r"\d+", val)
                if match:
                    return int(match.group())
        return None

    def _parse_answer(self, response):
        for line in response.strip().split("\n"):
            if line.strip().upper().startswith("ANSWER:"):
                return line.split(":", 1)[1].strip()
        return response.strip()


# task type to ability dimension mapping (BBH)
TASK_ABILITY_MAP = {
    "boolean_expressions": ["reasoning"],
    "causal_judgement": ["reasoning", "inference"],
    "date_understanding": ["reasoning", "knowledge"],
    "disambiguation_qa": ["language", "reasoning"],
    "dyck_languages": ["sequence", "reasoning"],
    "formal_fallacies": ["reasoning", "inference"],
    "geometric_shapes": ["spatial", "reasoning"],
    "hyperbaton": ["language"],
    "logical_deduction_five_objects": ["reasoning", "inference"],
    "logical_deduction_seven_objects": ["reasoning", "inference"],
    "logical_deduction_three_objects": ["reasoning", "inference"],
    "movie_recommendation": ["knowledge", "language"],
    "multistep_arithmetic_two": ["mathematical"],
    "navigate": ["spatial", "sequence"],
    "object_counting": ["mathematical", "reasoning"],
    "penguins_in_a_table": ["reasoning", "knowledge"],
    "reasoning_about_colored_objects": ["reasoning", "spatial"],
    "ruin_names": ["language", "knowledge"],
    "salient_translation_error_detection": ["language", "inference"],
    "snarks": ["language", "inference"],
    "sports_understanding": ["knowledge", "reasoning"],
    "temporal_sequences": ["sequence", "reasoning"],
    "tracking_shuffled_objects_five_objects": ["sequence", "spatial"],
    "tracking_shuffled_objects_seven_objects": ["sequence", "spatial"],
    "tracking_shuffled_objects_three_objects": ["sequence", "spatial"],
    "web_of_lies": ["reasoning", "inference"],
    "word_sorting": ["language", "sequence"],
}

# ability correlations (simplified)
ABILITY_CORRELATIONS = {
    ("reasoning", "inference"): 0.6,
    ("inference", "reasoning"): 0.6,
    ("reasoning", "mathematical"): 0.4,
    ("mathematical", "reasoning"): 0.4,
    ("language", "knowledge"): 0.4,
    ("knowledge", "language"): 0.4,
    ("sequence", "spatial"): 0.3,
    ("spatial", "sequence"): 0.3,
    ("sequence", "reasoning"): 0.3,
    ("reasoning", "sequence"): 0.3,
}