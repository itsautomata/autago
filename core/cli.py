#!/usr/bin/env python3
"""
autago: self-specializing agent network.

usage:
    autago --test                                          # test LLM connection
    autago --config config/bbh.yaml                        # specific config
    autago --provider openrouter --model google/gemma-4-31b-it
    autago --provider ollama --model qwen3:8b
    autago --provider gemini --model gemma-4-31b-it
    autago llm.temperature=0.5 experiment.agent_count=5    # override any config value
"""

import sys

from core import config as cfg
from core import llm


def parse_args(argv):
    config_path = None
    provider = None
    model = None
    test_mode = False
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
        elif arg == "--test":
            test_mode = True
            i += 1
        elif "=" in arg:
            overrides.append(arg)
            i += 1
        else:
            i += 1

    return config_path, provider, model, test_mode, overrides


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
    config_path, provider_override, model_override, test_mode, overrides = parse_args(args)

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

    # initialize LLM
    provider_name, kwargs = cfg.get_llm_kwargs(config)
    llm.init(provider_name, **kwargs)

    print(f"autago initialized")
    print(f"  provider: {provider_name}")
    print(f"  model: {config['llm']['model']}")
    print(f"  agents: {config['experiment']['agent_count']}")
    print()

    # TODO: experiment runner goes here
    print("experiment runner not built yet. use --test to verify LLM connection.")


if __name__ == "__main__":
    main()
