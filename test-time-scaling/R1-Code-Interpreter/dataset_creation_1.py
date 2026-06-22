import os
import json
import random
from generation_models import count_total_tokens, extract_code
from prompt import *

def load_conversation_data(file_path):
    #Load conversation data from a JSON file and return the original list formats.
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return (
            data['response_list'],
            data['user_prompt_list'],
            data['question']
        )
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def load_conversation_data_CodeSteer(file_path):
    #Load conversation data from a JSON file and return the original list formats.
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return (
            data['response_list_save'],
            data['user_prompt_list_save'],
            data['question']
        )
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def create_formatted_dataset(data_dir_list):
    """
    Process directories to create a formatted dataset from conversation data
    where success_failure.txt indicates 'True'.
    """
    dataset = []
    max_token_len = 600000
    response_len_dict = {}
    for data_path in data_dir_list:
        if not os.path.exists(data_path):
            print(f'Directory does not exist: {data_path}')
            continue
        # Walk through all subdirectories
        per_dir_count = 0
        for root, _, files in os.walk(data_path):
            conv_path = os.path.join(root, 'conversation_data.json')
            status_path = os.path.join(root, 'success_failure.txt')
            # Check if both required files exist
            if not (os.path.isfile(conv_path) and os.path.isfile(status_path)):
                continue
            try:
                # Read success_failure.txt first
                with open(status_path, 'r') as f:
                    status = f.read().strip()
                if status == 'True':  # Changed from 'Success' to 'True'
                    try:
                        if 'CodeSteer' in root:
                            response_list, user_prompt_list, question = load_conversation_data_CodeSteer(conv_path)
                        else:
                            response_list, user_prompt_list, question = load_conversation_data(conv_path)
                        question = R1_code_interpreter_data_syn_prompt1 + 'question: ' + question + '\n'

                        user_prompt_list[0] = question
                        if count_total_tokens(user_prompt_list, response_list) > max_token_len:
                            continue
                        for response in response_list:
                            code_block_list = extract_code(response)
                            if len(code_block_list) > 1:
                                print(f'Error: More than one code block in response: {response}')
                                continue
                        if len(response_list) == 10:
                            continue
                        if r'''<<<'answer'>>>''' in response_list[-1] or r'''<<<answer>>>''' in response_list[-1] or r'''<<<answer content>>>''' in response_list[-1]:
                            continue
                        #if len(response_list) == 1 and random.random() < 0.6:
                        #    continue
                        #if len(response_list) == 2 and random.random() < 0.7:
                        #    continue
                        if len(response_list) not in response_len_dict:
                            response_len_dict[len(response_list)] = 1
                        else:
                            response_len_dict[len(response_list)] += 1

                        # Verify we have matching data
                        if (len(user_prompt_list) > 0 and
                                len(user_prompt_list) == len(response_list)):
                            # Create history list from previous interactions
                            history = []
                            for i in range(len(user_prompt_list) - 1):
                                history.append([
                                    user_prompt_list[i],
                                    response_list[i]
                                ])
                            # Create dataset item with the final interaction
                            dataset_item = {
                                "instruction": user_prompt_list[-1],
                                "output": response_list[-1],
                                "history": history
                            }
                            dataset.append(dataset_item)
                            if per_dir_count < 60:
                                per_dir_count += 1
                            else:
                                break

                    except Exception as e:
                        print(f'Error processing conversation data in {root}: {str(e)}')
            except Exception as e:
                print(f'Error reading status file in {root}: {str(e)}')
    if not dataset:
        print('\nWarning: No valid data items were found!')
        print('Please verify:')
        print('1. The directory paths are correct')
        print('2. The directories contain the required files')
        print('3. The success_failure.txt files contain "True"')
        print('4. The conversation_data.json files are properly formatted')
    else:
        pass
    print(f'\nNumber of items in dataset: {len(dataset)}')
    response_len_dict = dict(sorted(response_len_dict.items()))
    print(f'\nResponse length distribution: {response_len_dict}')
    random.shuffle(dataset)
    return dataset

if __name__ == '__main__':

    ### 33 tasks from SymBench
    env_name_list1 = ['eight_queens', 'pooling', 'reversi', 'light_puzzles', 'new_operator', 'mahjong_pattern',
                      'statistical_counting', 'synthesis_decomposition', '2048', 'matrix_transformation']

    env_name_list2 = ['pattern_recognition', 'constrained_linear_arrangement', 'string_synthesis', 'logic_puzzle',
                      'string_insertion', 'letter_logic_diagram', 'standard_sudoku', 'string_deletion_and_modification']

    env_name_list3 = ['string_splitting', 'permutations_and_combinations', 'logical_equation',
                      'combinatorial_calculation', 'cryptanalysis', 'game24', 'BoxLift', 'Blocksworld', 'gsm',
                      'math_geometry']
    ### 27 tasks from BigBench Hard
    env_name_big_bench_hard = ['big_bench_hard:' + task for task in
                               ['date_understanding', 'web_of_lies', 'disambiguation_qa', 'formal_fallacies',
                                'geometric_shapes', 'logical_deduction_seven_objects', 'navigate', 'dyck_languages', 'boolean_expressions',
                                'causal_judgement', 'hyperbaton', 'logical_deduction_five_objects', 'logical_deduction_three_objects',
                                'movie_recommendation', 'multistep_arithmetic_two', 'object_counting', 'penguins_in_a_table', 'word_sorting',
                                'tracking_shuffled_objects_three_objects',
                                'tracking_shuffled_objects_seven_objects', 'tracking_shuffled_objects_five_objects',
                                'temporal_sequences',
                                'sports_understanding', 'snarks', 'salient_translation_error_detection', 'ruin_names',
                                'reasoning_about_colored_objects']]
    ### 100 Reasoning Gym tasks
    reasoning_gym_available_datasets = [
        'ab', 'acre', 'advanced_geometry', 'aiw', 'arc_1d', 'arc_agi', 'base_conversion', 'basic_arithmetic', 'bf',
        'binary_alternation', 'binary_matrix', 'bitwise_arithmetic', 'caesar_cipher', 'calendar_arithmetic',
        'chain_sum', 'circuit_logic',
    ]
    list_1 = ['codeio', 'color_cube_rotation', 'complex_arithmetic', 'count_primes', 'countdown', 'course_schedule',
              'dice', 'emoji_mystery', 'family_relationships', 'fraction_simplification', 'futoshiki', 'game_of_life',
              'gcd', 'graph_color', 'group_anagrams', 'isomorphic_strings', 'jugs', 'knight_swap', 'largest_island',
              'lcm', 'leg_counting', 'list_functions', 'manipulate_matrix', 'maze', 'needle_haystack',
              'number_filtering', 'number_format', 'number_sorting', 'palindrome_generation', 'palindrome_partitioning',
              'polynomial_multiplication', 'pool_matrix', 'propositional_logic', 'quantum_lock', 'ransom_note',
              'rectangle_count', 'rotate_matrix', 'rotten_oranges', 'rush_hour', 'self_reference', 'shortest_path',
              'simple_geometry', 'simple_integration', 'sokoban', 'spiral_matrix', 'string_manipulation', 'syllogism',
              'tower_of_hanoi', 'tsumego', 'word_ladder', 'zebra_puzzles']

    reasoning_gym_datasets = [task for task in reasoning_gym_available_datasets]
    reasoning_gym_datasets_train = ['reasoning_gym_' + task for task in list_1]

    data_dir_list = []
    base_path = '/home/ycchen/R1-Code-Interpreter/results_gather'


    baseline_method_name_list = ['R1_code_interpreter_data_syn_1', 'R1_code_interpreter_data_syn_hint', 'All_text', 'R1_code_interpreter_data_syn_code']
    for model_name in ['gpt-4o', 'claude-3-5-sonnet-20241022']:  # Qwen2.5-14B-1M, Qwen2.5-7B-1M, DeepSeek-R1-7B, DeepSeek-R1-8B, Llama3-8B
        for task_name in env_name_list1 + env_name_list2 + env_name_list3:
            for baseline_method_name in baseline_method_name_list:
                data_dir = os.path.join(base_path, task_name, f'result_{task_name}_{baseline_method_name}_{model_name}')
                if os.path.exists(data_dir):
                    data_dir_list.append(data_dir)
                else:
                    print(f'Directory does not exist: {data_dir}')

        for task_name in reasoning_gym_datasets[:10] + reasoning_gym_datasets_train:
            for baseline_method_name in baseline_method_name_list:
                data_dir = os.path.join(base_path, 'reasoning_gym', task_name, f'result_reasoning_gym_{task_name}_{baseline_method_name}_{model_name}')
                if os.path.exists(data_dir):
                    data_dir_list.append(data_dir)
                else:
                    print(f'Directory does not exist: {data_dir}')

        for task_name in env_name_big_bench_hard[:20]:
            for baseline_method_name in baseline_method_name_list:
                data_dir = os.path.join(base_path, 'big_bench_hard', f'result_{task_name}_{baseline_method_name}_{model_name}')
                if os.path.exists(data_dir):
                    data_dir_list.append(data_dir)
                else:
                    print(f'Directory does not exist: {data_dir}')

    formatted_dataset = create_formatted_dataset(data_dir_list)

    if formatted_dataset:
        output_path = './results_gather/R1_CI_gather_dataset_round4_version1.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(formatted_dataset, f, indent=2)
        print(f'\nDataset saved to: {output_path}')
