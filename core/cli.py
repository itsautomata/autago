#!/usr/bin/env python3
"""
autago: self-specializing agent network.

usage:
    autago --test                             # test LLM connection
    autago run                                # run experiment
    autago run --name "baseline"                           # named run
    autago run --provider openrouter --model google/gemma-4-31b-it
    autago run --provider ollama --model qwen3:8b
    autago run experiment.agent_count=5                    # override any config value
"""

import sys

from core import config as cfg
from core import llm


def parse_args(argv):
    config_path = None
    provider = None
    model = None
    test_mode = False
    run_mode = False
    estimate_mode = False
    models_mode = False
    run_name = None
    overrides = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
            i += 1
        elif arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        elif arg.startswith("--provider="):
            provider = arg.split("=", 1)[1]
            i += 1
        elif arg == "--provider" and i + 1 < len(argv):
            provider = argv[i + 1]
            i += 2
        elif arg.startswith("--model="):
            model = arg.split("=", 1)[1]
            i += 1
        elif arg == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        elif arg.startswith("--name="):
            run_name = arg.split("=", 1)[1]
            i += 1
        elif arg == "--name" and i + 1 < len(argv):
            run_name = argv[i + 1]
            i += 2
        elif arg == "--test":
            test_mode = True
            i += 1
        elif arg == "--estimate":
            estimate_mode = True
            i += 1
        elif arg == "--models":
            models_mode = True
            i += 1
        elif arg == "run":
            run_mode = True
            i += 1
        elif "=" in arg:
            overrides.append(arg)
            i += 1
        else:
            i += 1

    return (config_path, provider, model, test_mode,
            run_mode, estimate_mode, models_mode,
            run_name, overrides)


def test_llm(config):
    """quick test: one LLM call to verify the provider works."""
    provider_name, kwargs = cfg.get_llm_kwargs(config)
    print(f"provider: {provider_name}")
    print(f"model: {kwargs.get('model')}")
    print("testing connection...\n")

    try:
        llm.init(provider_name, **kwargs)
        llm.verify()
        print("connection ok.")
    except Exception as e:
        print(f"failed: {e}")
        if provider_name == "ollama":
            print("  start ollama: ollama serve")


def main():
    args = sys.argv[1:]
    (config_path, provider_override, model_override,
     test_mode, run_mode, estimate_mode, models_mode,
     run_name, overrides) = parse_args(args)

    # load config
    config = cfg.load(config_path)

    # apply CLI overrides
    if provider_override:
        overrides.append(f"llm.provider={provider_override}")
    if model_override:
        overrides.append(f"llm.model={model_override}")
    if overrides:
        config = cfg.override(config, overrides)

    if models_mode:
        pricing = config.get("pricing", {})
        print("available models (pricing per M tokens):\n")
        print(f"  {'model':<45} {'input':>7} {'output':>7}")
        print(f"  {'-'*45} {'-'*7} {'-'*7}")
        for model_name, prices in sorted(pricing.items()):
            inp = "free" if prices[0] == 0 else f"${prices[0]}"
            out = "free" if prices[1] == 0 else f"${prices[1]}"
            print(f"  {model_name:<45} {inp:>7} {out:>7}")
        print(
            "\nadd models in config/default.yaml "
            "under 'pricing:'"
        )
        return

    if estimate_mode:
        from core.cost import print_estimate
        print_estimate(config)
        return

    if test_mode:
        test_llm(config)
        return

    if run_mode:
        from core.experiment import run_experiment
        run_experiment(config, run_name=run_name)
        return

    # no command: show usage
    print("autago: self-specializing agent network\n")
    print("commands:")
    print("  autago --test                              test LLM connection")
    print("  autago --estimate                          estimate cost")
    print("  autago --models                            list models and pricing")
    print("  autago run                                 run experiment")
    print("  autago run --name baseline                 named run")
    print("  autago run --config config/custom.yaml     use custom config")
    print()
    print("provider overrides:")
    print("  --provider ollama|gemini|openrouter")
    print("  --model qwen3:8b|gemma-4-31b-it|google/gemma-4-31b-it")
    print()
    print("config overrides (key=value):")
    print("  llm.provider=ollama              LLM provider")
    print("  llm.model=qwen3:8b               model name")
    print("  llm.temperature=0.0              sampling temperature")
    print("  llm.max_tokens=2048              max output tokens")
    print("  experiment.agent_count=3         number of agents")
    print("  experiment.warmup_tasks=100      training tasks")
    print("  experiment.test_tasks=50         evaluation tasks")
    print("  experiment.max_per_task=10       examples per task type")
    print("  experiment.forward_path_max_length=3  max forward hops")
    print("  routing.mode=score               score|hybrid|llm")
    print("  agents.initial_ability=0.6       starting ability score")
    print("  agents.decay_rate=0.1            ability decay rate")
    print("  agents.decay_interval=10         decay every N tasks")
    print("  graph.prune_threshold=0.3        edge pruning threshold")
    print("  graph.success_factor=1.1         edge weight on success")
    print("  graph.failure_factor=0.9         edge weight on failure")
    print("  memory.executor_limit=40         executor memory pool size")
    print("  memory.retrieval_top_k=3         similar experiences to retrieve")
    print("  logging.enabled=true             detailed logging on/off")
    print("  logging.verbose=true             terminal output on/off")


if __name__ == "__main__":
    main()
