"""
cost estimator. predicts token usage and cost before running.
"""

# average tokens per component (estimated from prompt templates + typical responses)
TOKENS_PER_CALL = {
    "router_decide": {"input": 400, "output": 50},
    "router_pick": {"input": 300, "output": 30},
    "executor": {"input": 350, "output": 150},
    "identity": {"input": 300, "output": 400},
    "verify": {"input": 20, "output": 10},
    "embedding_load": {"input": 0, "output": 0},  # local, no API cost
}



def estimate(config):
    """estimate token usage and cost for a run."""
    exp = config["experiment"]
    routing_mode = config.get("routing", {}).get("mode", "score")
    agent_count = exp["agent_count"]
    warmup = exp.get("warmup_tasks", 100)
    test = exp.get("test_tasks", 50)
    total_tasks = warmup + test
    identity_interval = config.get("agents", {}).get(
        "identity_interval", 0
    )
    model = config["llm"]["model"]
    provider = config["llm"]["provider"]

    # calls per task by routing mode
    if routing_mode == "score":
        calls_per_task = {
            "executor": 1,
            "router_decide": 0,
            "router_pick": 0,
        }
    elif routing_mode == "hybrid":
        calls_per_task = {
            "executor": 1,
            "router_decide": 0.2,  # ~20% ambiguous
            "router_pick": 0.1,
        }
    else:  # llm
        calls_per_task = {
            "executor": 1,
            "router_decide": 1.5,  # avg including forwards
            "router_pick": 0.5,
        }

    # identity calls
    identity_calls = 0
    if identity_interval > 0:
        snapshots_per_agent = total_tasks // identity_interval
        identity_calls = snapshots_per_agent * agent_count

    # compute totals
    total_input = 0
    total_output = 0
    total_calls = 1  # verify call

    for call_type, count_per_task in calls_per_task.items():
        count = int(count_per_task * total_tasks)
        tokens = TOKENS_PER_CALL[call_type]
        total_input += tokens["input"] * count
        total_output += tokens["output"] * count
        total_calls += count

    # identity
    total_input += TOKENS_PER_CALL["identity"]["input"] * identity_calls
    total_output += TOKENS_PER_CALL["identity"]["output"] * identity_calls
    total_calls += identity_calls

    # cost from config pricing
    pricing_cfg = config.get("pricing", {})
    pricing = pricing_cfg.get(model, None)
    unknown_model = pricing is None
    if unknown_model:
        pricing = [0.0, 0.0]
    cost_input = (total_input / 1_000_000) * pricing[0]
    cost_output = (total_output / 1_000_000) * pricing[1]
    total_cost = cost_input + cost_output

    return {
        "total_tasks": total_tasks,
        "total_calls": total_calls,
        "identity_snapshots": identity_calls,
        "tokens": {
            "input": total_input,
            "output": total_output,
            "total": total_input + total_output,
        },
        "cost": {
            "input": f"${cost_input:.4f}",
            "output": f"${cost_output:.4f}",
            "total": f"${total_cost:.4f}",
            "free": total_cost == 0,
        },
        "unknown_model": unknown_model,
        "model": model,
        "provider": provider,
        "routing_mode": routing_mode,
    }


def print_estimate(config):
    """print a formatted cost estimate."""
    est = estimate(config)

    print("cost estimate:")
    print(f"  model: {est['model']} ({est['provider']})")
    print(f"  routing: {est['routing_mode']}")
    print(f"  tasks: {est['total_tasks']}")
    print(f"  LLM calls: {est['total_calls']}")
    if est["identity_snapshots"]:
        print(f"  identity snapshots: {est['identity_snapshots']}")
    print(
        f"  tokens: ~{est['tokens']['total']:,} "
        f"({est['tokens']['input']:,} in, "
        f"{est['tokens']['output']:,} out)"
    )
    if est["unknown_model"]:
        print(
            f"  cost: unknown (model '{est['model']}' "
            "not in pricing config)"
        )
        print("  add it to config/default.yaml under 'pricing:'")
    elif est["cost"]["free"]:
        print("  cost: free")
    else:
        print(f"  cost: {est['cost']['total']}")
