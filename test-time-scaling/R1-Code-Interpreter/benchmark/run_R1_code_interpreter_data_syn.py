import json
import re
import pandas as pd
import os
import subprocess
import sys
from openai import OpenAI
from generation_models import message_construct_func, message_construct_llama_func, GPT_response, count_total_tokens, \
    extract_code, extract_and_check, LLM_answer_code_checker, paraphrase_with_GPT4, log_run_info
import random
import math
import json
from typing import List, Tuple, Dict
import time
import numpy as np
import ast
import glob
from prompt import *
from argparse import ArgumentParser
from symbolic_code_check import analyze_code_and_explain
import random
import yaml
import string
from LLaMA_Factory.src.llamafactory.chat.chat_model import run_response
from retool.executor import PythonExecutor
from transformers import AutoTokenizer

### To do, add tasks to import related functions
from Logic_Game_func import load_task_dataset, verify_solution_func_gather, multi_round_answer_sampling

_RETOOL_TOKENIZER_CACHE = {}


def _get_retool_tokenizer(args_path):
    """Load and cache the tokenizer for the model specified in the LLaMA-Factory YAML.

    Used by the retool_eval_reformat branch to build the same chat-template
    prefix as retool/eval_fix.py (with the code_interpreter tool registered).
    """
    global _RETOOL_TOKENIZER_CACHE
    if args_path in _RETOOL_TOKENIZER_CACHE:
        return _RETOOL_TOKENIZER_CACHE[args_path]
    with open(args_path, "r") as _f:
        _yaml_args = yaml.safe_load(_f)
    model_path = _yaml_args.get("model_name_or_path")
    if model_path is None:
        raise ValueError(
            f"model_name_or_path not found in YAML config: {args_path}"
        )
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _RETOOL_TOKENIZER_CACHE[args_path] = tok
    return tok


_RETOOL_CODE_INTERPRETER_TOOL = [
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

def save_file_func_CodeSteer(save_code_dir, response_list, user_prompt_list, question, user_prompt_list_save, response_list_save):
    data = {
        'question': question,
        'response_list': response_list,
        'user_prompt_list': user_prompt_list,
        'user_prompt_list_save': user_prompt_list_save,
        'response_list_save': response_list_save
    }

    output_file = os.path.join(save_code_dir, 'conversation_data.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        f.close()
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def save_file_func_baselines(save_code_dir, response_list, user_prompt_list, question, system_message):
    data = {
        'question': question,
        'response_list': response_list,
        'user_prompt_list': user_prompt_list,
        'system_message': system_message
    }

    output_file = os.path.join(save_code_dir, 'conversation_data.json')

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        f.close()
        print(f"Data successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving data: {str(e)}")

def answer_sampling_func(model_name, args_path, user_prompt_list, response_list, code_interpreter=False):
    if model_name in ['o3-mini-2025-01-31', 'o1', "o1-preview", 'o1-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo',
                        "claude-3-5-sonnet-20241022", "claude-3-sonnet-20240229",
                        "claude-3-opus-20240229", "claude-3-haiku-20240307", 'open-mixtral-8x7b', "mistral-large-latest",
                        'DeepSeek-R1', 'DeepSeek-V3',
                        # our custom
                        'deepseek-ai/DeepSeek-V3',
                        ]:
        response = GPT_response('', user_prompt_list[-1], model_name=model_name, code_interpreter=code_interpreter,
                                user_prompt_list=user_prompt_list, response_total_list=response_list, logprobs=False)
    else:
        messages = message_construct_llama_func(user_prompt_list, response_list)
        response = run_response(messages, args_path)
    return response

def code_interpreter_func(save_code_dir, response, round_num):
    code_block_list = extract_code(response)
    if not code_block_list:
        return '', ''

    code = code_block_list[0]
    try:
        # Launch Python, read the script from stdin ("-")
        result = subprocess.run(
            ["python3", "-"],
            input=code,
            capture_output=True,
            text=True,
            timeout=60
        )
        output = result.stdout
        errors = result.stderr
    except subprocess.TimeoutExpired as e:
        try:
            output = e.stdout or ""
            errors = (e.stderr or "") + (
                f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"
            )
        except:
            output = ""
            errors = f"TimeoutExpired: Command timed out after 60 seconds"
    return output, errors

def run_logic_game_baselines(task_name, gather_save_input_dir, model_name, baseline_method_name, args_path, start_index, max_sample_num, max_round_num, train_test = 'together'):
    print("model name:", model_name)
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}_{train_test}.txt")

    task_exists = False
    # Open and read the log file
    try:
        with open(log_file_result, 'r') as file:
            content = file.read()
        f.close()

        # Check if the file is empty
        if not content.strip():
            print(f"The file {log_file_result} is empty.")
            task_exists = False
        else:
            # Check if task_name is in the content
            task_exists = task_name in content

            if task_exists:
                print(f"Task '{task_name}' found in the log file.")
            else:
                print(f"Task '{task_name}' not found in the log file.")

    except FileNotFoundError:
        print(f"Error: File {log_file_result} not found.")
    except Exception as e:
        print(f"Error reading file: {e}")

    if task_exists:
        print(f"Task '{task_name}' has already been completed. Skipping...")
        return

    print('\n' + '*'*30)
    total_sample_num = 0
    total_correct_num = 0

    # Load dataset
    solution_list, question_list, target_list, puzzles, solution_data_list, question_constrained_list, question_matrix_list, number_list, word_list, letter_list, save_input_dir = load_task_dataset(
        task_name, model_name)

    print(f'length of number_list: {len(number_list)}')
    base_save_code_dir = save_input_dir + f'/result_{task_name}_{baseline_method_name}_{model_name}'
    if not os.path.exists(base_save_code_dir):
        os.makedirs(base_save_code_dir)

    ### Remain unchanged
    question_num_total = len(question_list)

    for i in range(min(start_index, question_num_total-max_sample_num), max_sample_num + min(start_index, question_num_total-max_sample_num)):
        solution = ''; question = ''; number_list_item = ''; target = ''; word = ''; letter = ''
        question = question_list[i]
        if len(solution_list) > 0:
            solution = solution_list[i]
        if len(number_list) > 0:
            number_list_item = number_list[i]
        if len(target_list) > 0:
            target = target_list[i]
        if len(word_list) > 0:
            word = word_list[i]
        if len(letter_list) > 0:
            letter = letter_list[i]

        total_sample_num += 1

        save_code_dir = os.path.join(base_save_code_dir, f"Test_sample_{i}/")
        if not os.path.exists(save_code_dir):
            os.makedirs(save_code_dir)

        print('-------###-------###-------###-------')
        print(f"\nTest_sample_{i}, total: {min(len(solution_list), max_sample_num)}/")

        system_message = ''
        response_list = []
        code_usage_round_num = 0
        code_interpreter = False
        if baseline_method_name in ['1_only_ques', 'All_code_CoT', 'All_text', 'Code_Interpreter', 'All_code_CoT_retool']:
            print("all code cot selected")
            if baseline_method_name == '1_only_ques':
                user_prompt_list = [question]
            elif baseline_method_name == 'All_code_CoT':
                user_prompt_list = [with_COT_code_output_prompt + question]
            elif baseline_method_name == 'All_text':
                user_prompt_list = [text_output_prompt + question]
            elif baseline_method_name == 'Code_Interpreter':
                user_prompt_list = [question]
                code_interpreter = True
            elif baseline_method_name == 'All_code_CoT_retool':
                user_prompt_list = [with_COT_code_output_prompt_retool + question]

            response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list, code_interpreter)
            response_list.append(response)

        elif baseline_method_name == 'eval-only':
            result_path = "/ist-project/scads/jenta/result-table3/multiturn_sft_qwen2_5_coder_7b_instruct"
            content_path = glob.glob(f"{result_path}/{task_name}/*")
            final_path = content_path[0]
            sample_path = f"{final_path}/Test_sample_{i}/conversation_data.json"
            with open(sample_path, 'r') as file:
                response = json.load(file)
            response_list.extend(response["response_list"])
            user_prompt_list = response["user_prompt_list"]

        elif baseline_method_name == 'R1_code_interpreter_test':
            for round_num in range(max_round_num):
                if round_num == 0:
                    question = R1_code_interpreter_data_syn_prompt1 + 'question: ' + question + '\n'
                    user_prompt_list = [question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break
        
        elif baseline_method_name == 'R1_code_interpreter_test':
            for round_num in range(max_round_num):
                if round_num == 0:
                    question = R1_code_interpreter_data_syn_prompt1 + 'question: ' + question + '\n'
                    user_prompt_list = [question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break

        elif baseline_method_name == 'eval_fix_retool':
            # Inline eval_fix.py-style tool-call loop using answer_sampling_func.
            # Mirrors retool/eval_fix.py.main(): apply the Qwen chat template with
            # the code_interpreter tool registered, parse <tool_call>...</tool_call>
            # JSON, batch-execute via PythonExecutor.apply, then continue with a
            # <tool_response>...</tool_response> turn until the model stops emitting
            # tool calls or we hit max_round_num.
            executor = PythonExecutor(get_answer_from_stdout=True, timeout_length=10)

            tokenizer = _get_retool_tokenizer(args_path)

            # Prompt templates for eval_fix_retool. Mirrors the
            # `--prompt_template`/`--prompt` JSON-dict pattern used by
            # retool/eval_fix.py. `{query}` is Python str.format() placeholder;
            # literal braces use `{{`/`}}`.
            #   retool:     original upper+lower wording that embeds a
            #               `<code>```python\ncode snippet\n```\n</code>` example
            #               (conflicts with the Qwen tool template's <tool_call>
            #               schema and leaks a literal "code snippet" placeholder
            #               that extract_code() can accidentally grab).
            #   retool_fix: clean variant matching the working
            #               `exe-multiturn-...AIME24-prompt_template-8.jsonl`
            #               sample -- just the question plus a final-answer
            #               reminder. Lets the Qwen tools template be the sole
            #               source of tool-call-format instructions.
            PROMPT_TEMPLATES = {
                "retool": (
                    "Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox, and the output (wrapped in `<interpreter>output_str</interpreter>`) can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports. \n"
                    "Each code snippet is wrapped with `<code>\n```python\ncode snippet\n```\n</code>`.\n"
                    "The last part of your response should be in the following format:\n"
                    "<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>\n\n"
                    "*user question:*\n"
                    "Answer the following Math Problem and put the answer in the format of \\boxed{{answer}}\n\n"
                    "{query}\n\n\n"
                    "Remember to place the final answer in the last part using the format: \n"
                    "<answer>\n\\boxed{{'The final answer goes here.'}}\n</answer>"
                ),
                "retool_fix": (
                    "\n{query}\n\n\n"
                    "Remember to place the final answer in the last part using the format: \n\n"
                    "\\boxed{{'The final answer goes here.'}}"
                ),
            }
            PROMPT_KEY = "retool_fix"
            initial_prompt = PROMPT_TEMPLATES[PROMPT_KEY].format(query=question)

            # Strip Qwen's default "You are Qwen, created by Alibaba Cloud..."
            # system line via a scoped chat_template override. The tools block
            # (`# Tools`, `<tools>...</tools>`, `<tool_call>` framing) is emitted
            # by other Jinja expressions and is preserved. Only the `{{- '...' }}`
            # literal is blanked out, leaving valid Jinja (`{{- '' }}`).
            _qwen_default_system_literal = (
                "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
            )
            _original_chat_template = getattr(tokenizer, "chat_template", None)
            if (
                isinstance(_original_chat_template, str)
                and _qwen_default_system_literal in _original_chat_template
            ):
                tokenizer.chat_template = _original_chat_template.replace(
                    _qwen_default_system_literal, ""
                )
                try:
                    templated_prefix = tokenizer.apply_chat_template(
                        [{"role": "user", "content": initial_prompt}],
                        tools=_RETOOL_CODE_INTERPRETER_TOOL,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                finally:
                    tokenizer.chat_template = _original_chat_template
            else:
                templated_prefix = tokenizer.apply_chat_template(
                    [{"role": "user", "content": initial_prompt}],
                    tools=_RETOOL_CODE_INTERPRETER_TOOL,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            full_solution_trace = templated_prefix
            user_prompt_list = [initial_prompt]
            code_usage_round_num = 0
            stop_reason = "stop"

            for round_num in range(max_round_num):
                response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                response_list.append(response)

                print(f"\n########## Round {round_num} ##########\n response: {response}")

                # Dual-format detection so this works with both BSFT-tuned
                # models (emit `<tool_call>` JSON, like the sample jsonl from
                # eval_fix.py) and vanilla instruct models (emit ```python
                # markdown fences, like Qwen2.5-Coder-7B-Instruct defaults).
                # `<tool_call>` is preferred when both are present.
                tool_end = response.find("</tool_call>")
                md_match = (
                    re.search(r"```python\n(.*?)\n```", response, re.S)
                    if tool_end == -1 else None
                )

                if tool_end == -1 and md_match is None:
                    # No tool-call and no markdown fence -> model's done talking.
                    full_solution_trace += response
                    stop_reason = "stop"
                    break

                code_to_run = None

                if tool_end != -1:
                    # answer_sampling_func cannot stop on "</tool_call>" the way
                    # vLLM does in eval_fix.py, so we truncate at the first
                    # close-tag and discard anything the model generated past it.
                    pre_close = response[:tool_end]
                    full_solution_trace += pre_close

                    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", response, re.S)
                    if match:
                        try:
                            tool_call_dict = json.loads(match.group(1).replace("\n", "\\n"))
                            code_to_run = tool_call_dict.get("arguments", {}).get("code")
                        except Exception:
                            code_to_run = None
                else:
                    # Markdown-fence fallback: keep the model's reasoning text
                    # before the fence, then rewrite the fence itself as a
                    # synthetic `<tool_call>` block so the `solution` trace
                    # stays byte-compatible with eval_fix.py's output format
                    # (uniform `<tool_call>...</tool_call>` + `<tool_response>`
                    # framing regardless of which input format the model chose).
                    # Text AFTER the fence is discarded, matching how the
                    # `<tool_call>` path truncates at `</tool_call>`.
                    fence_start = md_match.start()
                    full_solution_trace += response[:fence_start]

                    code_to_run = md_match.group(1)
                    synthetic_tool_call = json.dumps({
                        "name": "code_interpreter",
                        "arguments": {"code": code_to_run},
                    })
                    full_solution_trace += (
                        "<tool_call>\n" + synthetic_tool_call + "\n"
                    )

                if code_to_run is not None and isinstance(code_to_run, str):
                    # Mirror retool/eval_fix.py lines 292-296 exactly: "Done" -> output,
                    # otherwise the error report. Only fall back to "None" when we
                    # couldn't extract code at all (equivalent of eval_fix.py's
                    # `no_code_idx` path, batch_results_include_none = None).
                    code_usage_round_num += 1
                    res, report = executor.apply(code_to_run)
                    excu_content = res if report == "Done" else report
                else:
                    excu_content = "None"

                feedback = (
                    "\n</tool_call><|im_end|>\n"
                    "<|im_start|>user\n<tool_response>\n" + str(excu_content) +
                    "\n</tool_response><|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                full_solution_trace += feedback

                user_prompt_list.append(f"<tool_response>\n{excu_content}\n</tool_response>")

                if round_num == max_round_num - 1:
                    stop_reason = "length"

            # Put the full accumulated trace into response_list so the outer
            # verify_solution_func_gather call (same pipeline as other baselines)
            # sees every tool-response turn, not just the model's final turn.
            if response_list:
                response_list[-1] = full_solution_trace
            else:
                response_list.append(full_solution_trace)

            token_len = len(tokenizer.encode(full_solution_trace))

            # Stash the partial prediction so we can write the final jsonl
            # *after* the outer verify_solution_func_gather call runs, which
            # lets us populate the `correctness` field the same way
            # retool/eval_fix.py does (lines 395-402).
            _eval_fix_retool_pending = {
                "full_solution_trace": full_solution_trace,
                "stop_reason": stop_reason,
                "token_len": token_len,
                "code_num": code_usage_round_num,
            }
            print("+"*27)
            print(_eval_fix_retool_pending)
            print("+"*27)

        elif baseline_method_name == 'beh_R1_code_interpreter_test':
            for round_num in range(max_round_num):
                if round_num == 0:
                    question = beh_R1_code_interpreter_data_syn_prompt1 + 'question: ' + question + '\n'
                    user_prompt_list = [question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break

        elif baseline_method_name == 'R1_code_interpreter_data_syn_1':
            for round_num in range(max_round_num):
                if round_num == 0:
                    user_prompt_list = [R1_code_interpreter_data_syn_prompt1 + question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break
        elif baseline_method_name == 'R1_code_interpreter_data_syn_code':
            for round_num in range(max_round_num):
                if round_num == 0:
                    user_prompt_list = [R1_code_interpreter_data_syn_prompt1_code + question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break
        elif baseline_method_name == 'R1_code_interpreter_data_syn_hint':
            for round_num in range(max_round_num):
                if round_num == 0:
                    user_prompt_list = [question]
                    user_prompt_list_input = [R1_code_interpreter_data_syn_prompt2 + question]
                    response = answer_sampling_func(model_name, args_path, user_prompt_list_input, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")
                else:
                    response = answer_sampling_func(model_name, args_path, user_prompt_list_input, response_list)
                    response_list.append(response)
                    print(f"\n########## Round {round_num} ##########\n response: {response}")

                output, errors = code_interpreter_func(save_code_dir, response, round_num)
                if len(output) > 0 or len(errors) > 0:
                    code_usage_round_num += 1
                    user_prompt_list_input.append(f'{R1_code_interpreter_data_syn_intermediate_step}Code output: {output}\nErrors: {errors}')
                    user_prompt_list.append(f'Code output: {output}\nErrors: {errors}')
                else:
                    break
        elif baseline_method_name == 'CodeSteer':
            CodeSteer_output_prompt_guidance_list = [];
            CodeSteer_input_prompt_list = [code_text_choice_prompt + question];
            CodeSteer_input_prompt_training_list = [code_text_choice_prompt + question]

            starting_prompt_choice = with_COT_code_output_prompt

            print(f'Starting prompt choice: {starting_prompt_choice}')
            user_prompt_list = [starting_prompt_choice + question]
            CodeSteer_output_prompt_guidance_list.append(starting_prompt_choice)
            response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
            response_list.append(response)

            ############ Further rounds of guidance ############
            for round_num in range(max_round_num):
                code_block_list = extract_code(response)
                if len(code_block_list) > 0:
                    code_complexity_summary, code_complexity_score = analyze_code_and_explain(code_block_list[0])
                    if code_complexity_score <= 2:
                        code_complexity_summary += '\nThe generated code may not be complex enough to carry out symbolic computing for solving the task.'
                    with open(save_code_dir + f"/code_1_{round_num}.py", "w") as f:
                        f.write(code_block_list[0])
                    f.close()

                    try:
                        result = subprocess.run(
                            ["python3", save_code_dir + f"/code_1_{round_num}.py"],
                            capture_output=True, text=True, timeout=60
                        )
                        output = result.stdout
                        errors = result.stderr
                    except subprocess.TimeoutExpired as e:
                        output = e.stdout if e.stdout else ""
                        errors = e.stderr if e.stderr else ""
                        errors += f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"
                    output = output[:1000]
                    errors = errors[:1000]
                    response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'

                    if isinstance(output, str):
                        if count_total_tokens([output + errors], []) > 8000:
                            response = response + f'\nThe execution result from the generated code is too long to be displayed.'
                        else:
                            response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'
                    else:
                        response = response + f'\nThe execution result from the generated code is:\nerrors: {errors}'

                check_code_saving_path = save_code_dir + f"/check_code_1_{round_num}.py"
                check_result = LLM_answer_code_checker(question, response, check_code_saving_path)

                CodeSteer_input_prompt_head = f'''{decision_prompt_complex_code} {question}\n'''
                if len(code_block_list) > 0:
                    print('\n############True#############\n')
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\nThe summary of generated code complexity is: {code_complexity_summary}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                else:
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe feedback from the checking agent is:\n{check_result}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                CodeSteer_input_prompt_total = CodeSteer_input_prompt_head + CodeSteer_input_prompt
                CodeSteer_input_prompt_list.append(CodeSteer_input_prompt_total)
                CodeSteer_input_prompt_training_list.append(CodeSteer_input_prompt)
                response_text = GPT_response("", '', model_name=model_name, code_interpreter=False,
                                             user_prompt_list=CodeSteer_input_prompt_list,
                                             response_total_list=CodeSteer_output_prompt_guidance_list,
                                             logprobs=False)
                matches = re.findall(r'<<<(.*?)>>>', response_text, re.DOTALL)
                guidance_prompt = matches[-1] if matches else response_text

                print(f'\nGuidance prompt_{round_num + 1}: {guidance_prompt}\n')

                CodeSteer_output_prompt_guidance_list.append(guidance_prompt)
                if '<<<Code>>>' in guidance_prompt:
                    guidance_prompt = with_COT_code_output_prompt
                elif '<<<Text>>>' in guidance_prompt:
                    guidance_prompt = text_output_prompt
                elif '<<<Return Answer>>>' in guidance_prompt or 'Return Answer' in guidance_prompt or '<<<Terminate>>>' in guidance_prompt or 'Terminate' in guidance_prompt:
                    break
                user_prompt_list.append(guidance_prompt)
                response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)

                response_list.append(response)
        elif baseline_method_name == 'CodeSteer_wo_self-answer_checker':
            CodeSteer_output_prompt_guidance_list = [];
            CodeSteer_input_prompt_list = [code_text_choice_prompt + question];
            CodeSteer_input_prompt_training_list = [code_text_choice_prompt + question]

            starting_prompt_choice = with_COT_code_output_prompt

            print(f'Starting prompt choice: {starting_prompt_choice}')
            user_prompt_list = [starting_prompt_choice + question]
            user_prompt_list_save = [question]
            CodeSteer_output_prompt_guidance_list.append(starting_prompt_choice)
            response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
            response_list.append(response)
            response_list_save = [response]
            response_save = ''

            ############ Further rounds of guidance ############
            for round_num in range(max_round_num):
                code_block_list = extract_code(response)
                if len(code_block_list) > 0:
                    code_complexity_summary, code_complexity_score = analyze_code_and_explain(code_block_list[0])
                    if code_complexity_score <= 2:
                        code_complexity_summary += '\nThe generated code may not be complex enough to carry out symbolic computing for solving the task.'
                    with open(save_code_dir + f"/code_1_{round_num}.py", "w") as f:
                        f.write(code_block_list[0])
                    f.close()

                    try:
                        result = subprocess.run(
                            ["python3", save_code_dir + f"/code_1_{round_num}.py"],
                            capture_output=True, text=True, timeout=60
                        )
                        output = result.stdout
                        errors = result.stderr
                    except subprocess.TimeoutExpired as e:
                        output = e.stdout if e.stdout else ""
                        errors = e.stderr if e.stderr else ""
                        errors += f"\nTimeoutExpired: Command '{e.cmd}' timed out after {e.timeout} seconds"
                    output = output[:1000]
                    errors = errors[:1000]
                    response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'

                    if isinstance(output, str):
                        if count_total_tokens([output + errors], []) > 8000:
                            response = response + f'\nThe execution result from the generated code is too long to be displayed.'
                        else:
                            response = response + f'\nThe execution result from the generated code is:\noutput: {output}, errors: {errors}'
                    else:
                        response = response + f'\nThe execution result from the generated code is:\nerrors: {errors}'

                CodeSteer_input_prompt_head = f'''{decision_prompt_complex_code} {question}\n'''
                if len(code_block_list) > 0:
                    print('\n############True#############\n')
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\nThe summary of generated code complexity is: {code_complexity_summary}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                else:
                    CodeSteer_input_prompt = f'''The response from TaskLLM is: {response}\n\n''' + \
                                             f'''The final returned guidance prompt should be of the format <<<guidance prompt content>>>.'''

                CodeSteer_input_prompt_total = CodeSteer_input_prompt_head + CodeSteer_input_prompt
                CodeSteer_input_prompt_list.append(CodeSteer_input_prompt_total)
                CodeSteer_input_prompt_training_list.append(CodeSteer_input_prompt)
                response_text = GPT_response("", '', model_name=model_name, code_interpreter=False,
                                             user_prompt_list=CodeSteer_input_prompt_list,
                                             response_total_list=CodeSteer_output_prompt_guidance_list,
                                             logprobs=False)
                matches = re.findall(r'<<<(.*?)>>>', response_text, re.DOTALL)
                guidance_prompt = matches[-1] if matches else response_text

                print(f'\nGuidance prompt_{round_num + 1}: {guidance_prompt}\n')

                CodeSteer_output_prompt_guidance_list.append(guidance_prompt)
                if '<<<Code>>>' in guidance_prompt:
                    guidance_prompt = with_COT_code_output_prompt
                elif '<<<Text>>>' in guidance_prompt:
                    guidance_prompt = text_output_prompt
                elif '<<<Return Answer>>>' in guidance_prompt or 'Return Answer' in guidance_prompt or '<<<Terminate>>>' in guidance_prompt or 'Terminate' in guidance_prompt:
                    break
                user_prompt_list.append(guidance_prompt)
                response = answer_sampling_func(model_name, args_path, user_prompt_list, response_list)
                if len(code_block_list) > 0:
                    user_prompt_list_save.append(f'Code output: {output}\nErrors: {errors}')
                    response_list_save.append(response_save + f'{guidance_prompt}\n\n{response}\n\n')
                    response_save = ''
                else:
                    response_save += f'{guidance_prompt}\n\n{response}\n\n'

                response_list.append(response)


        if 'CodeSteer' in baseline_method_name:
            save_file_func_CodeSteer(save_code_dir, response_list, user_prompt_list, question, user_prompt_list_save, response_list_save)
        else:
            save_file_func_baselines(save_code_dir, response_list, user_prompt_list, question, system_message)

        response = response_list[-1]
        #response = response[:10000]
        token_len_response = count_total_tokens([response], [])
        print(f'token_len_response: {token_len_response}')
        ## fix edge case where judger has not enough context length
        try:
            True_false_result_1, True_false_result_2 = verify_solution_func_gather(i, task_name, response,
                                                                                save_code_dir, question, solution,
                                                                                target, puzzles, solution_data_list,
                                                                                solution_list, question_constrained_list,
                                                                                question_matrix_list, number_list_item, word, letter)
        except Exception as e:
            print(str(e))
            True_false_result_1, True_false_result_2 = False, False

        ### Remain unchanged
        if True_false_result_1 == False and True_false_result_2 == False:
            print('False')
            with open(save_code_dir + f"/success_failure.txt", "w") as f:
                f.write('False')
            f.close()
        else:
            print('True')
            with open(save_code_dir + f"/success_failure.txt", "w") as f:
                f.write('True')
            f.close()
            total_correct_num += 1

        print(f'\ntotal_sample_num: {total_sample_num}')
        print(f'total_correct_num: {total_correct_num}\n')
        print(f'Code usage round number: {code_usage_round_num}\n')
        path = base_save_code_dir + f"/acc_result_log_{model_name}_{baseline_method_name}_{train_test}.txt"
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        with open(path, "w") as f:
            f.write(f"correct/all:{total_correct_num}/{total_sample_num}\n")
        f.close()

    run_info = f"CodeSteer, {task_name}, {baseline_method_name}, {model_name}\n"
    run_info_result = f'correct/all:{total_correct_num}/{total_sample_num}\n'
    log_file_result = os.path.join(gather_save_input_dir, f"acc_result_log_{model_name}_{train_test}.txt")
    log_run_info(log_file_result, run_info + run_info_result)
