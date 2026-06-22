"""HuggingFace Transformers retool evaluation for Symbench tasks.

Runs the same multi-turn tool-call inference loop as retool/eval_fix.py
(stop=["</tool_call>"] with raw-text continuation) but uses the
transformers library directly instead of vLLM. Questions are sourced
via Symbench's Logic_Game_func.load_task_dataset and scored via
Logic_Game_func.verify_solution_func_gather. Output jsonl matches
eval_fix.py's schema:

    {"input", "output", "prediction": [{"solution", "correctness",
     "stop_reason", "token_len", "code_num"}, ...]}

Entry point: run_retool_eval(...). Callable via inference.py through the
small dispatch in symbench_eval.run_logic_game_baselines.

NOTE: `max_sample_num` from inference.py is interpreted as `n`
(samples per question) per eval_fix.py semantics. All questions in the
symbench task dataset are processed.
"""

import os

# Silence HF tokenizers fork-after-parallelism warnings triggered by
# PythonExecutor's subprocess pool. Must be set before tokenizers is imported.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import time
import logging
import yaml

from prompt import retool_eval_prompt_upper, retool_eval_prompt_lower
from Logic_Game_func import load_task_dataset, verify_solution_func_gather
from retool.executor import PythonExecutor
from generation_models import log_run_info


logger = logging.getLogger("symbench_retool_eval")


def _ensure_logging_configured():
    """Attach a stderr handler with timestamps if the root logger has none.

    Level defaults to INFO; override with SYMBENCH_LOG_LEVEL=DEBUG etc.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(handler)
    level = os.environ.get("SYMBENCH_LOG_LEVEL", "INFO").upper()
    root.setLevel(getattr(logging, level, logging.INFO))


def _fmt_secs(s):
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{sec:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{sec:04.1f}s"


_CODE_INTERPRETER_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "A tool for executing code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]


_HF_BUNDLE_CACHE = {}


def _load_hf_bundle(args_path):
    """Load HF model + tokenizer from a YAML config. Cached per args_path."""
    if args_path in _HF_BUNDLE_CACHE:
        logger.debug("Reusing cached HF bundle for %s", args_path)
        return _HF_BUNDLE_CACHE[args_path]

    with open(args_path, "r") as f:
        cfg = yaml.safe_load(f)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = cfg["model_name_or_path"]
    logger.info("Loading model from %s", model_path)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    bundle = {
        "tokenizer": tokenizer,
        "model": model,
        "max_new_tokens": int(cfg.get("max_new_tokens", 10000)),
        "temperature": float(cfg.get("temperature", 0.6)),
        "top_p": float(cfg.get("top_p", 0.7)),
    }
    _HF_BUNDLE_CACHE[args_path] = bundle

    try:
        n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        n_gpu = 0
    logger.info(
        "Model loaded in %s  (dtype=bf16, device_map=auto, gpus=%d, max_new_tokens=%d, "
        "temperature=%.2f, top_p=%.2f)",
        _fmt_secs(time.time() - t0), n_gpu,
        bundle["max_new_tokens"], bundle["temperature"], bundle["top_p"],
    )
    return bundle


def _hf_generate(model, tokenizer, prompts, *, max_new_tokens, temperature,
                 top_p, stop_strings, n):
    """Generate completions for a list of prompts using transformers.

    Returns a list (one element per prompt) of lists of (text, stop_reason)
    tuples of length n. stop_reason semantics match vLLM's: one of the
    stop_strings (e.g. "</tool_call>"), "stop" (EOS), or "length" (hit
    max_new_tokens). When a stop string is matched, it is stripped from
    the returned text to mirror vLLM behavior.
    """
    import torch

    if not prompts:
        return []

    # Repeat each prompt n times; HF doesn't cleanly support per-sequence
    # num_return_sequences with stop_strings, so flatten to independent rows.
    input_texts = []
    for p in prompts:
        for _ in range(n):
            input_texts.append(p)

    device = next(model.parameters()).device
    enc = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    ).to(device)
    input_len = enc["input_ids"].shape[1]

    do_sample = temperature is not None and float(temperature) > 0.0
    gen_kwargs = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = float(temperature)
        gen_kwargs["top_p"] = float(top_p)
    if stop_strings:
        gen_kwargs["stop_strings"] = list(stop_strings)
        gen_kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        output_ids = model.generate(**gen_kwargs)

    gen_ids = output_ids[:, input_len:]
    pad_id = tokenizer.pad_token_id

    flat = []
    for row in gen_ids:
        row_list = row.tolist()
        full_len = len(row_list)
        end = full_len
        while end > 0 and row_list[end - 1] == pad_id:
            end -= 1
        trimmed = row_list[:end]
        text = tokenizer.decode(trimmed, skip_special_tokens=True)

        if end >= full_len:
            stop_reason = "length"
        else:
            stop_reason = "stop"

        for s in (stop_strings or []):
            if text.endswith(s):
                text = text[: -len(s)]
                stop_reason = s
                break

        flat.append((text, stop_reason))

    grouped = []
    for p_idx in range(len(prompts)):
        grouped.append(flat[p_idx * n:(p_idx + 1) * n])
    return grouped


def _extract_tool_call_code(tool_call_text):
    """Parse <tool_call>...</tool_call> JSON and return the inner code string (or None)."""
    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", tool_call_text, re.S)
    if not match:
        return None
    try:
        tool_call_dict = json.loads(match.group(1).replace("\n", "\\n"))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(tool_call_dict, dict):
        return None
    args = tool_call_dict.get("arguments")
    if not isinstance(args, dict):
        return None
    code = args.get("code")
    if not isinstance(code, str):
        return None
    return code


def _execute_tool_calls_batch(tool_call_texts, executor):
    """Batch-execute tool-call strings. Returns a list of excu_content strings (one per input)."""
    codes = []
    idx_with_code = []
    for i, text in enumerate(tool_call_texts):
        code = _extract_tool_call_code(text)
        if code is None:
            continue
        codes.append(code)
        idx_with_code.append(i)

    results = ["None"] * len(tool_call_texts)
    if not codes:
        return results
    batch = executor.batch_apply(codes)
    for i, (out, report) in zip(idx_with_code, batch):
        content = out if report == "Done" else report
        content = "None" if not str(content).strip() else str(content)
        results[i] = content
    return results


def _build_prompt(tokenizer, question):
    initial = retool_eval_prompt_upper + question + retool_eval_prompt_lower
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": initial}],
        tools=_CODE_INTERPRETER_TOOL,
        tokenize=False,
        add_generation_prompt=True,
    )


def _run_loop_for_question(model, tokenizer, executor, prompt,
                           gen_params, n, max_round_num):
    """Mirrors eval_fix.py's per-question tool-call loop for n samples.

    Stops a sample once it has executed `max_round_num` tool calls, marking
    it as finished with stop_reason="max_rounds" even if the model wanted
    to keep going.
    """
    initial = _hf_generate(
        model, tokenizer, [prompt],
        max_new_tokens=gen_params["max_new_tokens"],
        temperature=gen_params["temperature"],
        top_p=gen_params["top_p"],
        stop_strings=gen_params["stop_strings"],
        n=n,
    )[0]

    pred_stop_reason_lst = [[text, reason] for text, reason in initial]
    while len(pred_stop_reason_lst) < n:
        pred_stop_reason_lst.append(list(initial[0]))
    pred_stop_reason_lst = pred_stop_reason_lst[:n]

    intermediate_responses = [prompt for _ in range(n)]
    code_num_lst = [0] * n
    fini_responses = [None] * n
    fini_stop = [None] * n

    max_rounds = max(1, int(max_round_num))
    turn_idx = 0
    logger.debug("Initial generation: %d samples, %d finished with tool_call",
                 n, sum(1 for p in pred_stop_reason_lst if p and p[1] == "</tool_call>"))

    while any(p is not None for p in pred_stop_reason_lst):
        code_to_execute_lst = []
        exec_indices = []
        for res_idx in range(n):
            pred_stop_reason = pred_stop_reason_lst[res_idx]
            inter_response = intermediate_responses[res_idx]
            if pred_stop_reason is None or inter_response is None:
                continue
            pred, stop_reason = pred_stop_reason
            if stop_reason != "</tool_call>":
                fini_responses[res_idx] = inter_response + pred
                fini_stop[res_idx] = stop_reason
                pred_stop_reason_lst[res_idx] = None
                intermediate_responses[res_idx] = None
                continue
            if code_num_lst[res_idx] >= max_rounds:
                fini_responses[res_idx] = inter_response + pred
                fini_stop[res_idx] = "max_rounds"
                pred_stop_reason_lst[res_idx] = None
                intermediate_responses[res_idx] = None
                continue
            content = pred.split("<tool_call>")[-1].replace("</tool_call>", "").strip()
            content = "<tool_call>" + content + "</tool_call>"
            code_to_execute_lst.append(content)
            exec_indices.append(res_idx)
            intermediate_responses[res_idx] = inter_response + pred

        if not code_to_execute_lst:
            break

        exec_outputs = _execute_tool_calls_batch(code_to_execute_lst, executor)

        cont_prompts = []
        cont_indices = []
        for exec_out, idx in zip(exec_outputs, exec_indices):
            intermediate_responses[idx] += (
                "\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n"
                + exec_out
                + "\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
            )
            cont_prompts.append(intermediate_responses[idx])
            cont_indices.append(idx)
            code_num_lst[idx] += 1

        turn_idx += 1
        logger.debug("Turn %d: executing %d tool calls, continuing %d samples",
                     turn_idx, len(code_to_execute_lst), len(cont_prompts))
        new_outputs = _hf_generate(
            model, tokenizer, cont_prompts,
            max_new_tokens=gen_params["max_new_tokens"],
            temperature=gen_params["temperature"],
            top_p=gen_params["top_p"],
            stop_strings=gen_params["stop_strings"],
            n=1,
        )
        for new_i, idx in enumerate(cont_indices):
            text, reason = new_outputs[new_i][0]
            pred_stop_reason_lst[idx] = [text, reason]

    for i in range(n):
        if fini_responses[i] is None:
            base = intermediate_responses[i] or ""
            last = pred_stop_reason_lst[i]
            if last is not None:
                fini_responses[i] = base + last[0]
                fini_stop[i] = last[1]
            else:
                fini_responses[i] = base
                fini_stop[i] = "stop"

    predictions = []
    for i in range(n):
        sol = fini_responses[i]
        predictions.append({
            "solution": sol,
            "stop_reason": fini_stop[i],
            "token_len": len(tokenizer.encode(sol)),
            "code_num": code_num_lst[i],
        })
    return predictions


def run_retool_eval(task_name, gather_save_input_dir, model_name, baseline_method_name,
                    args_path, start_index, max_sample_num, max_round_num,
                    train_test="together"):
    """Symbench retool eval. Dispatch target for symbench_eval.run_logic_game_baselines.

    max_sample_num is interpreted as n (samples per question). All questions
    in the dataset are processed starting from start_index.
    """
    _ensure_logging_configured()

    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}_{train_test}.txt")

    task_exists = False
    try:
        with open(log_file_result, "r") as f:
            content = f.read()
        task_exists = bool(content.strip()) and (task_name in content)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("Error reading task log %s: %s", log_file_result, e)

    if task_exists:
        logger.info("Task '%s' already completed, skipping", task_name)
        return

    bundle = _load_hf_bundle(args_path)
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    n = max(1, int(max_sample_num))

    gen_params = {
        "temperature": bundle["temperature"],
        "top_p": bundle["top_p"],
        "max_new_tokens": bundle["max_new_tokens"],
        "stop_strings": ["</tool_call>"],
    }

    executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=10)

    (solution_list, question_list, target_list, puzzles, solution_data_list,
     question_constrained_list, question_matrix_list, number_list, word_list,
     letter_list, save_input_dir) = load_task_dataset(task_name, model_name)

    base_save_code_dir = save_input_dir + f"/result_{task_name}_{baseline_method_name}_{model_name}"
    os.makedirs(base_save_code_dir, exist_ok=True)

    safe_model = str(model_name).replace("/", "_")
    safe_task = str(task_name).replace("/", "_")
    tgt_jsonl = os.path.join(
        base_save_code_dir,
        f"{baseline_method_name}-{safe_model}-{safe_task}-n{n}.jsonl",
    )
    fout = open(tgt_jsonl, "w")

    total_problem = 0
    total_correct = 0

    question_num_total = len(question_list)
    start = max(0, int(start_index))
    end = question_num_total

    logger.info(
        "Task start: %s | questions=%d (start=%d) | n=%d | max_rounds=%d | out=%s",
        task_name, end - start, start, n, int(max_round_num), tgt_jsonl,
    )
    task_t0 = time.time()

    for i in range(start, end):
        question = question_list[i]
        solution = solution_list[i] if len(solution_list) > 0 else ""
        target = target_list[i] if len(target_list) > 0 else ""
        number_list_item = number_list[i] if len(number_list) > 0 else ""
        word = word_list[i] if len(word_list) > 0 else ""
        letter = letter_list[i] if len(letter_list) > 0 else ""

        save_code_dir = os.path.join(base_save_code_dir, f"Test_sample_{i}/")
        os.makedirs(save_code_dir, exist_ok=True)

        q_t0 = time.time()
        prompt = _build_prompt(tokenizer, question)
        prompt_tokens = len(tokenizer.encode(prompt))
        logger.info(
            "[%s] q %d/%d starting (prompt_tokens=%d, n=%d)",
            task_name, i + 1, end, prompt_tokens, n,
        )

        predictions = _run_loop_for_question(
            model, tokenizer, executor, prompt, gen_params, n, max_round_num,
        )

        any_correct = False
        q_correct = 0
        for pred in predictions:
            try:
                r1, r2 = verify_solution_func_gather(
                    i, task_name, pred["solution"], save_code_dir, question, solution,
                    target, puzzles, solution_data_list, solution_list,
                    question_constrained_list, question_matrix_list,
                    number_list_item, word, letter,
                )
            except Exception as e:
                logger.warning("[%s] q %d verify error: %s", task_name, i + 1, e)
                r1, r2 = False, False
            correct = bool(r1) or bool(r2)
            pred["correctness"] = correct
            any_correct = any_correct or correct
            total_problem += 1
            if correct:
                total_correct += 1
                q_correct += 1

        q_elapsed = time.time() - q_t0
        done_questions = i + 1 - start
        remaining = end - (i + 1)
        avg_q = (time.time() - task_t0) / max(1, done_questions)
        eta = avg_q * remaining
        stop_reasons = [p.get("stop_reason") for p in predictions]
        code_nums = [p.get("code_num") for p in predictions]
        token_lens = [p.get("token_len") for p in predictions]
        logger.info(
            "[%s] q %d/%d done in %s | correct=%d/%d | stop=%s code=%s tok=%s "
            "| running acc=%d/%d (%.1f%%) | ETA %s",
            task_name, i + 1, end, _fmt_secs(q_elapsed),
            q_correct, len(predictions),
            stop_reasons, code_nums, token_lens,
            total_correct, total_problem,
            100.0 * total_correct / max(1, total_problem),
            _fmt_secs(eta),
        )

        with open(os.path.join(save_code_dir, "success_failure.txt"), "w") as f:
            f.write("True" if any_correct else "False")

        new_data = {
            "input": question,
            "output": solution,
            "prediction": predictions,
        }
        fout.write(json.dumps(new_data) + "\n")
        fout.flush()

        with open(os.path.join(
            base_save_code_dir,
            f"acc_result_log_{model_name}_{baseline_method_name}_{train_test}.txt",
        ), "w") as f:
            f.write(f"correct/all:{total_correct}/{total_problem}\n")

    if total_problem > 0:
        results = {"accuracy": round(total_correct / total_problem * 100, 2)}
        fout.write(json.dumps(results) + "\n")
        fout.flush()

    fout.close()

    run_info = f"retool_eval, {task_name}, {baseline_method_name}, {model_name}\n"
    run_info_result = f"correct/all:{total_correct}/{total_problem}\n"
    log_run_info(log_file_result, run_info + run_info_result)

    task_elapsed = time.time() - task_t0
    if total_problem > 0:
        acc = round(total_correct / total_problem * 100, 2)
        logger.info(
            "Task done: %s | acc=%.2f%% (%d/%d) | elapsed=%s",
            task_name, acc, total_correct, total_problem, _fmt_secs(task_elapsed),
        )
    else:
        logger.warning("Task done: %s | no problems scored | elapsed=%s",
                       task_name, _fmt_secs(task_elapsed))
