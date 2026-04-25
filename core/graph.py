"""
agent graph. topology adaptation, edge weights, routing.
"""

from .agent import Agent


class AgentGraph:
    def __init__(self, config, logger=None):
        self.config = config
        graph_cfg = config["graph"]
        routing_cfg = config["routing"]["weights"]

        self.initial_weight = graph_cfg["initial_edge_weight"]
        self.edge_bounds = graph_cfg["edge_bounds"]
        self.prune_threshold = graph_cfg["prune_threshold"]
        self.success_factor = graph_cfg["success_factor"]
        self.failure_factor = graph_cfg["failure_factor"]

        self.routing_weights = routing_cfg
        self.max_forwards = config["experiment"]["forward_path_max_length"]

        # create agents
        n = config["experiment"]["agent_count"]
        self.agents = {i: Agent(i, config) for i in range(n)}

        # complete graph: all-to-all, asymmetric weights
        self.edges = {}
        for i in self.agents:
            for j in self.agents:
                if i != j:
                    self.edges[(i, j)] = self.initial_weight

        self.pruned_edges = []  # history of pruned connections
        self.logger = logger
        self.routing_mode = config.get("routing", {}).get("mode", "score")

    def get_agent(self, agent_id):
        return self.agents[agent_id]

    def get_neighbors(self, agent_id):
        """agents reachable from this agent (edge weight above threshold)."""
        return [
            self.agents[j]
            for (i, j) in self.edges
            if i == agent_id and self.edges[(i, j)] > self.prune_threshold
        ]

    def select_initial_agent(self, task):
        """pick the best starting agent for a task based on ability scores."""
        scores = {}
        for agent_id, agent in self.agents.items():
            scores[agent_id] = agent.ability_score(task.task_type)

        best_score = max(scores.values())
        best_agents = [aid for aid, s in scores.items() if s == best_score]

        # tiebreak: lowest load
        import random
        best_agents.sort(key=lambda aid: (self.agents[aid].load, random.random()))
        return self.agents[best_agents[0]]

    def score_candidates(self, source_agent, task, candidates):
        """score candidate agents for forwarding using routing weights."""
        w = self.routing_weights
        scored = []

        for agent in candidates:
            if task.has_visited(agent.id):
                continue

            ability = agent.ability_score(task.task_type)
            load_factor = 1.0 / (1.0 + agent.load)
            success = source_agent.success_rates.get(agent.id, 0.0)
            connectivity = 1.0 if len(self.get_neighbors(agent.id)) > 0 else 0.0

            score = (
                w["ability"] * ability
                + w["load"] * load_factor
                + w["success_rate"] * success
                + w["connectivity"] * connectivity
            )
            scored.append((agent, score))

        scored.sort(key=lambda x: -x[1])
        return [agent for agent, _ in scored]

    def update_edge(self, source_id, target_id, success, execution_time=1.0):
        """update edge weight after a routing result."""
        key = (source_id, target_id)
        if key not in self.edges:
            return

        factor = self.success_factor if success else self.failure_factor
        time_factor = min(1.0, 1.0 / (execution_time * 0.1)) if execution_time > 0 else 1.0

        new_weight = self.edges[key] * factor * time_factor
        self.edges[key] = max(self.edge_bounds[0], min(self.edge_bounds[1], new_weight))

        # prune if below threshold
        if self.edges[key] <= self.prune_threshold:
            del self.edges[key]
            self.pruned_edges.append({
                "source": source_id,
                "target": target_id,
                "reason": f"weight dropped to {new_weight:.3f}",
            })

    def process_task(self, task):
        """route a task through the network. returns (result, path)."""
        agent = self.select_initial_agent(task)
        task.add_to_history(agent.id)
        path = [agent.id]

        if self.logger:
            self.logger.task_start(task, agent.id)

        for _ in range(self.max_forwards + 1):
            # compute best other agent's score for this task type
            best_other_score = 0.0
            for other in self.agents.values():
                if other.id != agent.id and not task.has_visited(other.id):
                    s = other.ability_score(task.task_type)
                    if s > best_other_score:
                        best_other_score = s

            decision = agent.decide_action(
                task, self.max_forwards,
                routing_mode=self.routing_mode,
                best_other_score=best_other_score,
            )

            # extract reasoning from the last decision record
            reasoning = agent.decisions[-1].get("reasoning", "") if agent.decisions else ""

            if self.logger:
                self.logger.routing_decision(
                    agent.id, task, decision, reasoning, dict(agent.abilities)
                )

            if decision == "EXECUTE":
                result = agent.execute(task)
                task.complete(result)
                if self.logger:
                    self.logger.execution(agent.id, task, result, reasoning)
                return task, path

            elif decision == "FORWARD":
                neighbors = self.get_neighbors(agent.id)
                candidates = self.score_candidates(agent, task, neighbors)

                if not candidates:
                    result = agent.execute(task)
                    task.complete(result)
                    if self.logger:
                        self.logger.execution(agent.id, task, result, "no candidates, forced execute")
                    return task, path

                next_id = agent.pick_next_agent(task, candidates[:3], routing_mode=self.routing_mode)
                next_agent = self.agents[next_id]

                if self.logger:
                    self.logger.forward(agent.id, next_id, task)

                task.add_to_history(next_id)
                path.append(next_id)
                agent = next_agent

            elif decision == "SPLIT":
                result = agent.execute(task)
                task.complete(result)
                if self.logger:
                    self.logger.execution(agent.id, task, result, "split (treated as execute)")
                return task, path

        # max forwards reached, force execute
        result = agent.execute(task)
        task.complete(result)
        if self.logger:
            self.logger.execution(agent.id, task, result, "max forwards reached")
        return task, path

    def update_after_task(self, task, path, success):
        """update graph after a task completes."""
        if self.logger:
            self.logger.task_result(task, path, success)

        # update edges along the path
        for i in range(len(path) - 1):
            source_id = path[i]
            target_id = path[i + 1]
            old_weight = self.edges.get((source_id, target_id), 0)
            self.update_edge(source_id, target_id, success, task.execution_time)
            new_weight = self.edges.get((source_id, target_id), 0)
            pruned = (source_id, target_id) not in self.edges

            if self.logger:
                self.logger.edge_update(source_id, target_id, old_weight, new_weight, pruned)

            self.agents[source_id].update_success_rate(target_id, success)

        # update abilities for the executing agent (last in path)
        executor_id = path[-1]
        executor = self.agents[executor_id]
        old_abilities = dict(executor.abilities)
        executor.update_abilities(task.task_type, success)

        if self.logger:
            self.logger.ability_update(
                executor_id, task.task_type, old_abilities, dict(executor.abilities), success
            )

    def topology_summary(self):
        """current state of the graph."""
        n = len(self.agents)
        total_possible = n * (n - 1)
        active_edges = len(self.edges)
        pruned = len(self.pruned_edges)

        return {
            "agents": n,
            "active_edges": active_edges,
            "pruned_edges": pruned,
            "density": active_edges / total_possible if total_possible > 0 else 0,
            "edge_weights": {f"{i}->{j}": w for (i, j), w in sorted(self.edges.items())},
            "agent_summaries": {aid: a.summary() for aid, a in self.agents.items()},
        }
