# Test-Time Scaling

Inference-time evaluation of code-interpreter reasoning. A model is prompted on a task, the Python it emits is executed, the output is fed back, and the loop repeats until the model stops calling the interpreter or the round budget runs out.

We follow the implementation from [R1-Code-Interpreter](https://github.com/yongchao98/R1-Code-Interpreter), which is vendored here under `R1-Code-Interpreter/` together with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) as the local-model inference backend. Refer to the upstream repo for task definitions, dataset construction and the original baselines; this page covers only what differs here and what you need to change to run it.

## Benchmarks

`benchmark_inference_test_generic.py` sweeps the **33 SymBench tasks** (26 `train` + 7 `test`) defined in `env_name_list1`–`env_name_list4`. **BIG-Bench-Hard** (27 tasks, `big_bench_hard:<task>`) and **Reasoning Gym** (84 tasks, `reasoning_gym_<task>`) are defined but commented out — swap in the commented `task_list_train` / `task_list_test` assignments in `__main__` to include them, taking the sweep to 144 tasks.

Data lives in `dataset_gather/`; loaders and verifiers for every task are in `Logic_Game_func.py` (`load_task_dataset`, `verify_solution_func_gather`).

## Baseline methods

Pass one as `--baseline_method_name`. Each is a branch in `run_logic_game_baselines`:

| Method | Notes |
|:---|:---|
| `1_only_ques`, `All_text`, `All_code_CoT` | Single-turn upstream baselines; no code execution |
| `Code_Interpreter` | OpenAI Assistants API hosted `code_interpreter`; API models only |
| `R1_code_interpreter_test` | Upstream multi-turn code-interpreter loop |
| `R1_code_interpreter_data_syn_{1,code,hint}` | Upstream rollout collection for data synthesis |
| `CodeSteer`, `CodeSteer_wo_self-answer_checker` | Upstream guidance-model loops |
| `All_code_CoT_retool` | **Added here** — single-turn with ReTool `<code>` framing |
| `beh_R1_code_interpreter_test` | **Added here** — multi-turn with the cognitive-behavior prompt |
| `eval_fix_retool` | **Added here** — ReTool `<tool_call>` / `<tool_response>` loop on the Qwen chat template, matching `train-time/evaluation` |
| `eval-only` | **Added here** — re-scores saved transcripts without generating |

## Setup

### 1. Provide the `retool.executor` module

Both runners do a module-level `from retool.executor import PythonExecutor`. That package is **not** in this repo, so without it every baseline fails at import. [`train-time/evaluation/executor.py`](../train-time/evaluation/executor.py) is the same class:

```bash
cd test-time-scaling/R1-Code-Interpreter
mkdir -p retool && touch retool/__init__.py
ln -s ../../../train-time/evaluation/executor.py retool/executor.py
```

### 2. Inference YAML (LLaMA-Factory route)

Downloaded weights are served in-process by LLaMA-Factory, pointed at by `model_name_or_path`. `--LLM_path` is a **file name** under `LLaMA_Factory/examples/inference/`:

```yaml
model_name_or_path: /path/to/checkpoint/model
template: qwen
infer_backend: huggingface  # choices: [huggingface, vllm]
trust_remote_code: true
temperature: 0.6
```

### 3. Hard-coded paths to change

| File | Variable | Current value |
|:---|:---|:---|
| `benchmark_inference_test_generic.py` | `infer_base_path` | `/examples/inference/` |
| `benchmark/run_R1_code_interpreter_data_syn.py` | `result_path` (`eval-only` branch) | `/result-table3/...` |
| `dataset_creation_1.py` | `base_path` | `/R1-Code-Interpreter/results_gather` |

### 4. Model routing

`answer_sampling_func` picks the backend from `--model_name`: ids on its allow-list go to `GPT_response` in `generation_models.py`, everything else to LLaMA-Factory's `run_response`. There are therefore two ways to run downloaded weights:

- **In-process** — a LLaMA-Factory YAML whose `model_name_or_path` is the local path (step 2). This is the default route, since any unrecognised `--model_name` lands here.
- **Over an OpenAI-compatible server** — `generation_models.py` talks to whatever `LLM_BASE_URL` points at, so weights served by vLLM/SGLang work by passing the served id or path as `--model_name`.

Going the second route means adding your id to **both** allow-lists, or it silently falls through to LLaMA-Factory or raises `Model name is wrong!`:

| Location | Role |
|:---|:---|
| `answer_sampling_func` (`benchmark/run_R1_code_interpreter_data_syn.py`) | Chooses `GPT_response` over `run_response` |
| `GPT_response_once` (`generation_models.py`) | Validates the id, then selects the chat-completions path |

The two `/opt/models/Qwen/...` entries are in the second list only, so under `benchmark_inference_test_generic.py` they still route to LLaMA-Factory. They are reachable as the verifier judge, which reads `LLM_MODEL_NAME` (8 call sites in `Logic_Game_func.py`) and calls `GPT_response` directly.

Environment variables read by `generation_models.py`:

| Variable | Default | Used for |
|:---|:---|:---|
| `LLM_BASE_URL` | - | OpenAI-compatible endpoint |
| `LLM_API_KEY` | `sk-` | Key for that endpoint |
| `LLM_MODEL_NAME` | `gpt-4o` | Judge model used by the verifiers |

Anthropic, Mistral and DeepSeek keys are still inline placeholders in `generation_models.py`.

## Running

Run from `R1-Code-Interpreter/` — all imports and data paths are relative to it.

```bash
cd test-time-scaling/R1-Code-Interpreter
python benchmark_inference_test_generic.py \
  --model_name multiturn-bsft-qwen-2.5-coder-7b-instruct \
  --LLM_path my_model.yaml \
  --baseline_method_name eval_fix_retool
```

| Flag | Default | Description |
|:---|:---|:---|
| `--model_name` | `None` | An allow-listed API or served-model id, or any label for a LLaMA-Factory checkpoint (see step 4) |
| `--LLM_path` | `None` | YAML file name under `infer_base_path` (required even for API models) |
| `--baseline_method_name` | `None` | One method from the table above |
| `--max_n_tasks` | `10000` | Cap the task sweep to the first *n* tasks |

## Output

```
results_gather/
├── acc_result_log_{model}_together.txt        # One line per finished task
├── Cost_runtime_gather_{model}_{method}.txt   # Mean seconds per sample
└── {task}/result_{task}_{method}_{model}/
    └── Test_sample_{i}/
        ├── conversation_data.json             # question, user_prompt_list, response_list
        └── success_failure.txt                # "True" or "False"
```

## Related

- [Training-time fine-tuning](../train-time/train/README.md) · [Training-time evaluation](../train-time/evaluation/README.md)
- Upstream: [R1-Code-Interpreter](https://github.com/yongchao98/R1-Code-Interpreter) · [CodeSteer / SymBench](https://github.com/yongchao98/CodeSteer-v1.0) · [Reasoning Gym](https://github.com/open-thought/reasoning-gym) · [BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard)
