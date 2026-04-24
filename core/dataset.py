"""
dataset loader. downloads and parses BBH (BIG-Bench Hard) tasks.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent.parent / "data"
BBH_BASE_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/bbh"

BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


def download_bbh(task_name):
    """download a single BBH task file."""
    bbh_dir = DATA_DIR / "bbh"
    bbh_dir.mkdir(parents=True, exist_ok=True)

    filepath = bbh_dir / f"{task_name}.json"
    if filepath.exists():
        return filepath

    url = f"{BBH_BASE_URL}/{task_name}.json"
    print(f"  downloading {task_name}...")
    with httpx.Client(timeout=30) as client:
        resp = client.get(url)
        resp.raise_for_status()
        filepath.write_text(resp.text)

    return filepath


def load_bbh_task(task_name):
    """load a BBH task, downloading if needed. returns list of (input, target) pairs."""
    filepath = download_bbh(task_name)
    data = json.loads(filepath.read_text())

    examples = []
    for ex in data.get("examples", []):
        examples.append({
            "input": ex["input"],
            "target": ex["target"].strip(),
            "task_type": task_name,
        })
    return examples


def load_bbh(task_names=None, max_per_task=None, shuffle=True):
    """load multiple BBH tasks. returns flat list of examples, shuffled across types."""
    tasks = task_names or BBH_TASKS
    all_examples = []

    for task_name in tasks:
        examples = load_bbh_task(task_name)
        if max_per_task:
            examples = examples[:max_per_task]
        all_examples.extend(examples)

    if shuffle:
        random.shuffle(all_examples)

    return all_examples


def split_dataset(examples, train_ratio=0.8):
    """split examples into train (warmup) and test sets.
    uses stratified split: each task type is split proportionally
    so both sets contain a mix of all types."""
    by_type = defaultdict(list)
    for ex in examples:
        by_type[ex["task_type"]].append(ex)

    train, test = [], []
    for task_type, type_examples in by_type.items():
        split_idx = max(1, int(len(type_examples) * train_ratio))
        train.extend(type_examples[:split_idx])
        test.extend(type_examples[split_idx:])

    random.shuffle(train)
    random.shuffle(test)

    return train, test