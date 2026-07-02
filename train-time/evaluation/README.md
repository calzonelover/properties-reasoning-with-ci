# Evaluation

Multi-turn math evaluation with optional Python code execution (ReTool-style tool calling). Models are run with [vLLM](https://github.com/vllm-project/vllm); generated code is executed in a sandbox and fed back to the model until it finishes or hits a stop token.

Supported benchmarks:

| Dataset   | File                         | Description                          |
|-----------|------------------------------|--------------------------------------|
| `AIME24`  | `dataset/AIME24.jsonl`       | AIME 2024 problems                   |
| `AIME25`  | `dataset/AIME25.jsonl`       | AIME 2025 problems                   |
| `MATH500` | `dataset/MATH500.jsonl`      | 500-problem MATH subset              |

## Directory layout

```
evaluation/
├── eval.py              # Main evaluation entry point
├── executor.py          # Sandboxed Python code executor
├── prompt_template.json # Prompt templates (`retool`, `retool_fix`, …)
├── dataset/             # Benchmark JSONL files
├── container/           # Apptainer/Singularity image definition
│   ├── env.def
│   └── build.sh
└── script/              # SLURM batch eval scripts (sft / bsft only)
    └── eval_all_{model}_{sft|bsft}.sh
```

## Requirements

- **GPU cluster** with NVIDIA GPUs (scripts assume A100, 1–8 GPUs depending on model size)
- **Apptainer/Singularity** (for containerized runs)
- **SLURM** (optional; for `eval_all_*.sh` scripts)
- Python packages (installed in the container or locally):
  - `vllm`, `transformers`, `datasets`, `symeval`, `timeout_decorator`, `pebble`, `multiprocess`, `tqdm`, `numpy`

## Setup

### 1. Evaluator module

`eval.py` imports graders from `evaluator/`:

```python
from evaluator.MC_evaluator_list import MCEvaluator
from evaluator.MATH_evaluator_list import MATHEvaluator
```

That directory is not included in this repo. Copy or symlink it from the ReTool evaluation tree:

```bash
cd test-time-scaling/evaluation
cp -r /path/to/ReTool/evaluation/evaluator .
```

Alternatively, add the ReTool evaluation path to `PYTHONPATH`:

```bash
export PYTHONPATH="/path/to/ReTool/evaluation:${PYTHONPATH}"
```

### 2. Build the Apptainer image

```bash
cd container
bash build.sh   # produces env.sif
```

SLURM scripts reference `custom_img_eval_retool.sif`. Either rename the built image or update the script paths:

```bash
cp env.sif custom_img_eval_retool.sif
```

The image is based on `verlai/verl` and installs vLLM, flash-attn, symeval, and related dependencies (see `container/env.def`).

### 3. Model and HuggingFace cache

Set a shared cache directory for model weights:

```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
export VLLM_CACHE_ROOT=/path/to/cache
```

Pass the same path to `--cache_dir` when running `eval.py`.

## Quick start

From the `evaluation/` directory, with 4 GPUs available:

```bash
python eval.py \
  --data_name MATH500 \
  --target_path ./results \
  --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
  --paralle_size 4 \
  --prompt_template ./prompt_template.json \
  --prompt retool \
  --max_tokens 16384 \
  --cache_dir /path/to/cache \
  --exe_code
```

- Use `--exe_code` to enable multi-turn code interpreter tool calling.
- Set `--paralle_size` equal to the number of GPUs (tensor parallelism).
- Use `--prompt retool_fix` for the shorter prompt variant in `prompt_template.json`.

### Run inside Apptainer

```bash
apptainer exec --nv --bind /dev/shm:/dev/shm --writable-tmpfs \
  ./container/env.sif /bin/bash -c \
  "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/compat:\${LD_LIBRARY_PATH} \
   && cd /path/to/evaluation \
   && python eval.py \
        --data_name MATH500 \
        --target_path ./results \
        --model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
        --paralle_size 4 \
        --prompt_template ./prompt_template.json \
        --prompt retool_fix \
        --max_tokens 16384 \
        --cache_dir /path/to/cache \
        --exe_code"
```

### Submit a SLURM job

1. Pick a script under `script/` matching your model and training stage (`sft` or `bsft`).
2. Edit `MODEL_PATH`, `CACHE_PATH`, partition, and GPU count if needed.
3. Submit from the evaluation directory:

```bash
cd test-time-scaling/evaluation
mkdir -p log
sbatch script/eval_all_qwen2_5_coder_7b_bsft.sh
```

Each script loops over **AIME24, AIME25, and MATH500** and writes results under `results/{model}_{stage}/`.

## CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--data_name` | (required) | `AIME24`, `AIME25`, or `MATH500` |
| `--target_path` | (required) | Output directory for result JSONL files |
| `--model_name_or_path` | (required) | HuggingFace model ID or local checkpoint path |
| `--paralle_size` | `8` | vLLM tensor parallel size (number of GPUs) |
| `--max_tokens` | `10000` | Max tokens per generation step |
| `--prompt_template` | `None` | Path to prompt template JSON |
| `--prompt` | `retool` | Template key inside the JSON file |
| `--n` | `8` | Number of samples per problem (pass@k) |
| `--bs` | `100` | Batch size for prompt processing |
| `--exe_code` | off | Enable code interpreter multi-turn loop |
| `--enable_lora` | off | Load LoRA adapter from `{model}/lora_adapter` |
| `--cache_dir` | `None` | HuggingFace / vLLM download cache |
| `--use_slice` | off | Split dataset across slices (for multi-job runs) |
| `--slice_id` | `0` | Slice index when `--use_slice` is set |

## How evaluation works

1. **Prompting** — Each problem is wrapped with a chat template and an optional `code_interpreter` tool definition.
2. **Generation** — vLLM generates a response; generation stops at `</tool_call>` when `--exe_code` is set.
3. **Code execution** — Python code inside the tool call is extracted and run in `executor.py` (timeout-protected sandbox).
4. **Multi-turn loop** — Execution output is appended as a tool response and generation continues until no more tool calls.
5. **Scoring** — Answers are graded with `MATHEvaluator` (symbolic math equivalence via `symeval`). A fallback checks the last number in the final line against the reference.

## Output format

Results are written to:

```
{target_path}/{exe|no_exe}-{model_tag}-{dataset}-{template}-{n}.jsonl
```

Each line (except the last) is a JSON object:

```json
{
  "input": "problem text",
  "output": "reference answer",
  "prediction": [
    {
      "solution": "full model output",
      "correctness": true,
      "stop_reason": null,
      "token_len": 1234,
      "code_num": 2
    }
  ]
}
```

The final line contains aggregate accuracy:

```json
{"accuracy": 42.5}
```

## Script naming guide

### `eval_all_{model}_{stage}.sh`

Every script runs `eval.py` with `--prompt retool_fix` and `--exe_code` over **AIME24, AIME25, and MATH500**.

| Stage | Meaning |
|-------|---------|
| **`sft`** | SFT checkpoint — **no** cognitive behavior injection |
| **`bsft`** | BSFT checkpoint — **injects** cognitive behavior |

| Script | Model checkpoint |
|--------|------------------|
| `eval_all_qwen2_5_3b_sft.sh` | `multiturn-sft-qwen-2.5-3b-instruct` |
| `eval_all_qwen2_5_3b_bsft.sh` | `multiturn-bsft-qwen-2.5-3b-instruct` |
| `eval_all_qwen2_5_7b_sft.sh` | `multiturn-sft-qwen-2.5-7b-instruct` |
| `eval_all_qwen2_5_7b_bsft.sh` | `multiturn-bsft-nt-qwen-2.5-7b-instruct` |
| `eval_all_qwen2_5_coder_7b_sft.sh` | `multiturn-sft-qwen-2.5-coder-7b-instruct` |
| `eval_all_qwen2_5_coder_7b_bsft.sh` | `multiturn-bsft-qwen-2.5-coder-7b-instruct` |
| `eval_all_qwen3_8b_sft.sh` | `multiturn-sft-qwen-3-8b-instruct` |
| `eval_all_qwen3_8b_bsft.sh` | `multiturn-bsft-qwen-3-8b-instruct` |
| `eval_all_qwen3_8b_base_sft.sh` | `multiturn-sft-qwen-3-8b-base` |
| `eval_all_qwen3_8b_base_bsft.sh` | `multiturn-bsft-qwen-3-8b-base` |
| `eval_all_retool_32b_sft.sh` | `JoeYing/ReTool-Qwen-32B-SFT` |

Before submitting, edit the script to set `MODEL_PATH`, `CACHE_PATH`, `TARGET_PATH`, `PARALLEL_SIZE`, and SLURM partition/QoS for your cluster.

## Dataset format

Each JSONL row:

```json
{
  "input": "question text",
  "output": "reference answer (often \\boxed{...} LaTeX)"
}
```

`MATH500.jsonl` may include extra fields (`subject`, `level`, `solution`, …); only `input` and `output` are used at eval time.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: evaluator` | Copy or symlink `evaluator/` (see Setup above) |
| CUDA OOM | Lower `--paralle_size`, reduce `--bs`, or lower `--max_tokens` |
| Container not found | Build `container/env.sif` and update script SIF paths |
| Wrong paths in scripts | Replace hard-coded `/ist-project/...` paths with your checkout and cache locations |
| `HF_HOME` / download errors | Set `--cache_dir` and ensure the cache directory is writable |

## Related code

- **Upstream ReTool eval:** `/path/to/ReTool/evaluation/` (`evaluator/` module)
- **Training (separate from this folder):** `test-time-scaling/R1-Code-Interpreter/` (VERL SFT / RL pipelines)
