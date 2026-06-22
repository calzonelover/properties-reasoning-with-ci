import argparse

def arg_convert():
    parser = argparse.ArgumentParser(description='pog')
    parser.add_argument('--start_task', type=int, default=None)
    parser.add_argument('--end_task', type=int, default=None)
    parser.add_argument('--name_path', type=str, default="check_token")
    parser.add_argument('--save_file_name', type=str, default="qwen3-32b-check-5-token-full")
    # additional
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--LLM_path', type=str, default=None)
    parser.add_argument('--baseline_method_name', type=str, default=None)
    parser.add_argument('--max_n_tasks', type=int, default=None)
    
    args = parser.parse_args()
    return args