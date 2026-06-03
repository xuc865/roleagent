"""
Prompt templates aligned with supplementary Figures 7–9 of
"Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution" (ACL anonymous submission).

Task-specific agent templates (e.g. search) live under ``agent_system/environments/prompts/``.
"""

# Figure 7 — predicate / judgment reward for one (obs, action, predicted_next, actual_next) tuple.
PROMPT_COMPARE_PREDICTED_VS_ACTUAL = """You are an objective evaluator assessing how accurately an agent predicted the outcome of its
action in an interactive environment.
## Current Observation
{current_obs_text}
##Action Taken
{action_text}
## Agent's Prediction of Next Observation
{predicted_next_text}
## Actual Next Observation
{actual_next_text}
## Your Task
Evaluate how well the agent's prediction matches the actual next observation. Focus on whether the
key facts, objects, and state changes described in the prediction align with what actually happened.
Minor wording differences are acceptable; what matters is semantic correctness.
Respond with a single score between 0.0 and 1.0. The metrics are:
1.0: The prediction is fully correct — all key elements match.
0.5: The prediction is partially correct — some elements match.
0.0: The prediction is incorrect — the actual outcome differs substantially.
Output your score in the following format and nothing else:
YOUR_SCORE"""


# Figure 8 — Agent-In-World failure abstraction (per failed trajectory).
PROMPT_FAILURE_MODE_FROM_TRAJECTORY = """You are an expert AI trainer specializing in diagnosing why AI agents fail at multi-step reasoning
tasks.
## Task Context
{task_description}
## Failed Trajectory
The agent attempted the task above but failed. Here are the steps it took:
{trajectory_description}
## Your Analysis Task Carefully examine the trajectory and produce a structured failure analysis.
**Step 1 – Root Cause Identification**
Identify the PRIMARY failure modes and describe briefly:
**Step 2 – Critical Step Identification**
Identify the SINGLE step where the failure became irreversible (the "point of no return").
**Step 3 – Transferable Lesson**
State a concise, generalizable lesson (1-2 sentences) that would help an agent avoid this class of
mistake on SIMILAR tasks in the future. Focus on the decision rule, not the specific content.
## Output Format
Wrap your entire analysis in <reflection> tags using this exact structure:
<reflection>
ROOT_CAUSE_TYPE: [category from Step 1]
ROOT_CAUSE_DETAIL: [1-2 sentences explaining why this root cause applies]
CRITICAL_STEP: [step number and brief description of what went wrong]
TRANSFERABLE_LESSON: [the generalizable rule an agent should follow]
RETRIEVAL_QUERY: [the query for the retrieval stage]
</reflection>"""


# Figure 9 — retrieve historical tasks with similar failure modes (LLM-as-environment curriculum).
PROMPT_RETRIEVE_SIMILAR_FAILURES = """You are an expert AI curriculum designer. Your job is to identify which historical training tasks are
most relevant for helping an agent overcome a specific failure pattern.
## Current Failure Pattern
The agent is currently struggling with the following error pattern:
{error_pattern}
## Historical Task Candidates
Below are historical tasks where the agent previously failed. Each entry shows the task description
and a brief failure analysis.
{candidates_text}
## Your Task
Select tasks from the list above that are MOST SIMILAR to the current failure pattern.
Similarity means:
1. The task requires the same type of reasoning or skill that the agent is currently failing at.
2. The task's failure analysis describes a similar root cause or mistake.
3. Re-training on this task would most directly help the agent overcome the current pattern.
## Output Format
Output ONLY the following structured block, with no additional text:
<selected_tasks>
INDEX/TASK/REFLECTIONS: <index, task and reflections from the candidate list>
REASON: <one sentence explaining why this task matches the current failure pattern>
INDEX/TASK/REFLECTIONS: <index, task and reflections from the candidate list>
REASON: <one sentence explaining why this task matches the current failure pattern>
</selected_tasks>"""
