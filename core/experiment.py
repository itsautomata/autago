"""
experiment runner. feeds tasks to the graph, measures what happens.
"""

import json
import time
from datetime import datetime
from pathlib import Path

from . import config as cfg
from . import embedding
from . import llm
from .dataset import load_bbh, split_dataset
from .graph import AgentGraph
from .logger import Logger
from .metrics import Metrics
from .task import Task

RESULTS_DIR = Path(__file__).parent.parent / "results"


def run_experiment(config, run_name=None):
    """run a full experiment: warmup + test."""
    # init LLM and verify connection before anything else
    provider_name, kwargs = cfg.get_llm_kwargs(config)
    llm.init(provider_name, **kwargs)

    print(f"verifying {provider_name} connection...")
    try:
        llm.verify()
    except Exception as e:
        print(f"connection failed: {e}")
        if provider_name == "ollama":
            print("  start ollama: ollama serve")
        return
    print("connection ok.\n")

    # init embedding model for memory
    embed_cfg = config.get("embedding", {})
    embedding.init(
        model_name=embed_cfg.get("model", "all-MiniLM-L6-v2"),
        max_cache=embed_cfg.get("cache_limit", 1000),
    )

    exp_cfg = config["experiment"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{timestamp}_{run_name}" if run_name else timestamp
    run_dir = RESULTS_DIR / dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"{'='*50}")
    print("  autago experiment")
    print(f"  provider: {provider_name}")
    print(f"  model: {config['llm']['model']}")
    print(f"  agents: {exp_cfg['agent_count']}")
    print(f"  routing: {config.get('routing', {}).get('mode', 'score')}")
    print(f"{'='*50}\n")

    # load data
    print("loading BBH dataset...")
    examples = load_bbh(max_per_task=exp_cfg.get("max_per_task"))
    train, test = split_dataset(examples, train_ratio=0.8)

    warmup_count = min(exp_cfg.get("warmup_tasks", len(train)), len(train))
    test_count = min(exp_cfg.get("test_tasks", len(test)), len(test))
    train = train[:warmup_count]
    test = test[:test_count]

    print(f"  warmup tasks: {len(train)}")
    print(f"  test tasks: {len(test)}")
    print()

    # create logger and graph
    log_cfg = config.get("logging", {})
    log_enabled = log_cfg.get("enabled", True)
    log_verbose = log_cfg.get("verbose", True)
    logger = Logger(
        log_dir=run_dir if log_enabled else None,
        verbose=log_verbose,
    ) if log_enabled else None
    graph = AgentGraph(config, logger=logger)

    # phase 1: warmup (train with updates)
    print("--- WARMUP ---\n")
    warmup_metrics = Metrics()
    run_phase(graph, train, warmup_metrics, "warmup", run_dir, update_graph=True)

    print(f"\n  warmup accuracy: {warmup_metrics.accuracy:.1%}")
    print(f"  warmup avg hops: {warmup_metrics.avg_hops:.2f}")

    # save post-warmup state
    save_graph_state(graph, run_dir / "post_warmup_state.json")

    # phase 2: test (evaluate without updates)
    print("\n--- TEST ---\n")
    test_metrics = Metrics()
    run_phase(graph, test, test_metrics, "test", run_dir, update_graph=False)

    print(f"\n  test accuracy: {test_metrics.accuracy:.1%}")
    print(f"  test avg hops: {test_metrics.avg_hops:.2f}")

    # save results
    if logger:
        logger.phase_summary("warmup", warmup_metrics, graph.agents)
        logger.phase_summary("test", test_metrics, graph.agents)
        logger.save_entries(run_dir / "log_entries.json")
        logger.close()
    save_results(graph, warmup_metrics, test_metrics, run_dir)

    print(f"\n{'='*50}")
    print("  EXPERIMENT COMPLETE")
    print(f"  warmup accuracy: {warmup_metrics.accuracy:.1%}")
    print(f"  test accuracy: {test_metrics.accuracy:.1%}")
    spec = test_metrics.specialization_depth(graph.agents)
    print(f"  specialization depth: {spec:.4f}")
    topo = graph.topology_summary()
    active = topo['active_edges']
    possible = topo['agents'] * (topo['agents'] - 1)
    print(f"  edges remaining: {active}/{possible}")
    print(f"  memory usage: {warmup_metrics.memory_usage:.1%}")
    print(f"  results: {run_dir}")
    print(f"{'='*50}\n")


def run_phase(graph, examples, metrics, phase_name, run_dir, update_graph=True):
    """run a set of tasks through the graph."""
    phase_dir = run_dir / phase_name
    phase_dir.mkdir(exist_ok=True)

    for i, example in enumerate(examples):
        task = Task(
            task_id=f"{phase_name}-{i}",
            task_type=example["task_type"],
            description=example["input"],
            expected_answer=example["target"],
        )

        try:
            completed_task, path = graph.process_task(task)
            success = completed_task.check_answer()
            if success is None:
                success = False

            # check if memory was used BEFORE storing this task
            executor_id = path[-1] if path else None
            memory_used = False
            if executor_id is not None:
                executor = graph.agents[executor_id]
                query = f"[{task.task_type}] {task.description}"
                memory_used = len(executor.executor_memory.retrieve(query)) > 0

            if update_graph:
                graph.update_after_task(completed_task, path, success)

            metrics.record(completed_task, path, success, memory_used=memory_used)

            status = "ok" if success else "xx"
            print(f"  [{status}] {i+1}/{len(examples)} {task.task_type} "
                  f"path={path} answer={completed_task.result[:40]}")

        except Exception as e:
            task.fail()
            metrics.record(task, [], False)
            print(f"  [!!] {i+1}/{len(examples)} {task.task_type} error: {e}")

        # save progress every 10 tasks
        if (i + 1) % 10 == 0:
            metrics.save(phase_dir / "metrics.json")
            save_graph_state(graph, phase_dir / "graph_state.json")

        time.sleep(0.2)  # rate limit buffer

    metrics.save(phase_dir / "metrics.json")


def save_graph_state(graph, path):
    """save current graph topology and agent states."""
    state = graph.topology_summary()
    Path(path).write_text(json.dumps(state, indent=2, default=str))


def save_results(graph, warmup_metrics, test_metrics, run_dir):
    """save final results."""
    results = {
        "warmup": warmup_metrics.full_report(graph.agents),
        "test": test_metrics.full_report(graph.agents),
        "topology": graph.topology_summary(),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))