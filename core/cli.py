#!/usr/bin/env python3
"""
autago: self-specializing agent network.

usage:
    autago --test                                          # test LLM connection
    autago run                                             # run experiment with default config
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
    run_name = None
    overrides = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            i += 2
        elif arg == "--provider" and i + 1 < len(argv):
            provider = argv[i + 1]
            i += 2
        elif arg == "--model" and i + 1 < len(argv):
            model = argv[i + 1]
            i += 2
        elif arg == "--name" and i + 1 < len(argv):
            run_name = argv[i + 1]
            i += 2
        elif arg == "--test":
            test_mode = True
            i += 1
        elif arg == "run":
            run_mode = True
            i += 1
        elif "=" in arg:
            overrides.append(arg)
            i += 1
        else:
            i += 1

    return config_path, provider, model, test_mode, run_mode, run_name, overrides


def test_llm(config):
    """quick test: one LLM call to verify the provider works."""
    provider_name, kwargs = cfg.get_llm_kwargs(config)
    print(f"provider: {provider_name}")
    print(f"model: {kwargs.get('model')}")
    print(f"testing connection...\n")

    provider = llm.init(provider_name, **kwargs)

    response = llm.call(
        "you are a helpful assistant. respond in one sentence.",
        [{"role": "user", "content": "what is 2 + 2?"}],
        temperature=0.0,
        max_tokens=50,
    )
    print(f"response: {response}")
    print("\nconnection ok.")


def main():
    args = sys.argv[1:]
    config_path, provider_override, model_override, test_mode, run_mode, run_name, overrides = parse_args(args)

    # load config
    config = cfg.load(config_path)

    # apply CLI overrides
    if provider_override:
        overrides.append(f"llm.provider={provider_override}")
    if model_override:
        overrides.append(f"llm.model={model_override}")
    if overrides:
        config = cfg.override(config, overrides)

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
    print("  autago --test                  test LLM connection")
    print("  autago run                     run experiment")
    print("  autago run --name baseline     named run")
    print("  autago run --provider ollama   use specific provider")
    print("  autago run experiment.agent_count=5  override config")


if __name__ == "__main__":
    main()
