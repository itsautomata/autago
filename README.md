# autago

a self-specializing, self-evolving agent network.

agents start identical, become different and evolve through task pressure. topology, capability scores and memory. no central orchestration.

based on [AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems](https://arxiv.org/abs/2504.00587) (NeurIPS 2025) framework. reproducing it from scratch with major refinements and more configuration control.

## setup

```bash
# install
uv pip install -e .

# set up provider (pick one)
# option 1: local (ollama)
ollama serve                  # in another terminal
ollama pull qwen3:8b

# option 2: gemini 
echo "GEMINI_API_KEY=your_key" > .env

# option 3: openrouter
echo "OPENROUTER_API_KEY=your_key" > .env

# test connection
autago --test

# run experiment
autago run

# run with different provider
autago run --provider gemini --model gemma-4-31b-it 

# quick test run with a name
autago run experiment.warmup_tasks=10 experiment.test_tasks=5 --name quick_test
```

## datasets

### BBH (first dataset)

tasks from [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) (BBH). 27 task types across 7 ability dimensions.

| task | ability dimensions |
|---|---|
| boolean_expressions | reasoning |
| causal_judgement | reasoning, inference |
| date_understanding | reasoning, knowledge |
| disambiguation_qa | language, reasoning |
| dyck_languages | sequence, reasoning |
| formal_fallacies | reasoning, inference |
| geometric_shapes | spatial, reasoning |
| hyperbaton | language |
| logical_deduction_five_objects | reasoning, inference |
| logical_deduction_seven_objects | reasoning, inference |
| logical_deduction_three_objects | reasoning, inference |
| movie_recommendation | knowledge, language |
| multistep_arithmetic_two | mathematical |
| navigate | spatial, sequence |
| object_counting | mathematical, reasoning |
| penguins_in_a_table | reasoning, knowledge |
| reasoning_about_colored_objects | reasoning, spatial |
| ruin_names | language, knowledge |
| salient_translation_error_detection | language, inference |
| snarks | language, inference |
| sports_understanding | knowledge, reasoning |
| temporal_sequences | sequence, reasoning |
| tracking_shuffled_objects_five_objects | sequence, spatial |
| tracking_shuffled_objects_seven_objects | sequence, spatial |
| tracking_shuffled_objects_three_objects | sequence, spatial |
| web_of_lies | reasoning, inference |
| word_sorting | language, sequence |

## configuration

all settings live in `config/default.yaml` and can be overridden from the CLI with `key=value`.

### LLM provider

| key | default | options |
|---|---|---|
| llm.provider | ollama | ollama, gemini, openrouter |
| llm.model | qwen3:8b | any model supported by the provider |
| llm.temperature | 0.0 | 0.0 - 1.0 |
| llm.max_tokens | 2048 | max output tokens per call |

### experiment

| key | default | description |
|---|---|---|
| experiment.agent_count | 3 | number of agents in the network |
| experiment.warmup_tasks | 100 | training tasks (graph evolves) |
| experiment.test_tasks | 50 | evaluation tasks (graph frozen) |
| experiment.max_per_task | 10 | examples loaded per task type |
| experiment.forward_path_max_length | 3 | max forward hops per task |

### routing

| key | default | description |
|---|---|---|
| routing.mode | score | score (pure math, fast), hybrid (LLM on ambiguous), llm (original) |
| routing.weights.ability | 0.4 | weight for ability match in agent scoring |
| routing.weights.load | 0.3 | weight for agent availability |
| routing.weights.success_rate | 0.2 | weight for historical success |
| routing.weights.connectivity | 0.1 | weight for outgoing connections |

### agents

| key | default | description |
|---|---|---|
| agents.initial_ability | 0.6 | starting score for all ability dimensions |
| agents.ability_bounds | [0.1, 2.0] | min/max ability scores |
| agents.decay_rate | 0.1 | ability decay multiplier |
| agents.decay_interval | 10 | decay triggers every N tasks |

### graph

| key | default | description |
|---|---|---|
| graph.initial_edge_weight | 1.0 | starting weight for all connections |
| graph.edge_bounds | [0.1, 2.0] | min/max edge weights |
| graph.prune_threshold | 0.3 | edges below this are removed |
| graph.success_factor | 1.1 | edge weight multiplier on success |
| graph.failure_factor | 0.9 | edge weight multiplier on failure |

### embedding

| key | default | description |
|---|---|---|
| embedding.model | all-MiniLM-L6-v2 | local embedding model for memory retrieval |
| embedding.cache_limit | 1000 | max cached embeddings |

model options: `all-MiniLM-L6-v2` (80MB, fast), `BAAI/bge-small-en-v1.5` (130MB), `BAAI/bge-large-en-v1.5` (1.3GB, original AgentNet)

### memory

| key | default | description |
|---|---|---|
| memory.executor_limit | 40 | max experiences per agent executor pool |
| memory.router_limit | -1 | max experiences per agent router pool (-1 = unlimited) |
| memory.retrieval_top_k | 3 | similar experiences retrieved per query |
| memory.retrieval_threshold | 0.7 | minimum similarity to retrieve |

### logging

| key | default | description |
|---|---|---|
| logging.enabled | true | write detailed logs to file |
| logging.verbose | true | show per-task output in terminal |

when enabled, logs are saved to `results/<run_id>/`:
- `detailed.log`: human-readable, full decision trace
- `log_entries.json`: structured, machine-readable
- `results.json`: final metrics and agent states
