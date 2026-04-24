"""
prompts for agent routing and execution decisions.
"""

ROUTER_SYSTEM = """you are a routing agent in a multi-agent network. your job is to decide what to do with a task.

you have three options:
1. EXECUTE: you solve the task yourself
2. FORWARD: you pass the task to another agent who might be better suited
3. SPLIT: you break the task into subtasks, solve the first one, and forward the rest

consider:
- your own capabilities (provided below)
- the task type and difficulty
- whether you've seen similar tasks before
- how many times this task has already been forwarded (max allowed: {max_forwards})

respond with EXACTLY this format:
DECISION: [EXECUTE or FORWARD or SPLIT]
REASONING: [one sentence explaining why]
"""

ROUTER_DECIDE = """TASK:
type: {task_type}
description: {description}
context so far: {context}

YOUR CAPABILITIES:
{capabilities}

FORWARD HISTORY: {forward_history}
FORWARDS REMAINING: {forwards_remaining}

{similar_experiences}

what is your decision?"""

ROUTER_PICK_AGENT = """you decided to FORWARD this task. pick the best agent from the candidates below.

TASK:
type: {task_type}
description: {description}

CANDIDATES:
{candidates}

respond with EXACTLY:
AGENT: [agent_id number]
REASONING: [one sentence]
"""

EXECUTOR_SYSTEM = """you are a task-solving agent. solve the given task directly and concisely.

respond with EXACTLY this format:
ANSWER: [your answer]
REASONING: [brief explanation of your approach]
"""

EXECUTOR_SOLVE = """TASK:
type: {task_type}
description: {description}
context: {context}

{similar_experiences}

solve this task."""