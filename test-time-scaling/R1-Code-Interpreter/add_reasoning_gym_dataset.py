import os
import csv
import json
import reasoning_gym
import numpy as np

seed = 32
task_size = 1000

available_datasets = [
    'ab',
    'acre',
    'advanced_geometry',
    'aiw',
    'arc_1d',
    'arc_agi',
    'base_conversion',
    'basic_arithmetic',
    'bf',
    'binary_alternation',
    'binary_matrix',
    'bitwise_arithmetic',
    'caesar_cipher',
    'calendar_arithmetic',
    'chain_sum',
    'circuit_logic',
    'codeio',
    'color_cube_rotation',
    'complex_arithmetic',
    'count_bits',
    'count_primes',
    'countdown',
    'course_schedule',
    'cryptarithm',
    'decimal_arithmetic',
    'dice',
    'emoji_mystery',
    'family_relationships',
    'figlet_font',
    'fraction_simplification',
    'futoshiki',
    'game_of_life',
    'game_of_life_halting',
    'gcd',
    'graph_color',
    'group_anagrams',
    'gsm_symbolic',
    'intermediate_integration',
    'isomorphic_strings',
    'jugs',
    'knight_swap',
    'knights_knaves',
    'largest_island',
    'lcm',
    'leg_counting',
    'letter_counting',
    'letter_jumble',
    'list_functions',
    #'mahjong_puzzle',
    'manipulate_matrix',
    'maze',
    #'mini_sudoku',
    'modulo_grid',
    #'n_queens',
    'needle_haystack',
    'number_filtering',
    'number_format',
    'number_sequence',
    'number_sorting',
    'palindrome_generation',
    'palindrome_partitioning',
    'polynomial_equations',
    'polynomial_multiplication',
    'pool_matrix',
    'prime_factorization',
    'products',
    'propositional_logic',
    #'puzzle24',
    'quantum_lock',
    'ransom_note',
    'rearc',
    'rectangle_count',
    'rotate_matrix',
    'rotten_oranges',
    'rubiks_cube',
    'rush_hour',
    'self_reference',
    'shortest_path',
    'simple_equations',
    'simple_geometry',
    'simple_integration',
    'sokoban',
    'spell_backward',
    'spiral_matrix',
    #'string_insertion',
    'string_manipulation',
    #'string_splitting',
    #'string_synthesis',
    #'sudoku',
    'syllogism',
    'time_intervals',
    'tower_of_hanoi',
    'tsumego',
    'word_ladder',
    'word_sequence_reversal',
    #'word_sorting',
    'zebra_puzzles',
]

output_dir = './dataset_gather/reasoning_gym'
os.makedirs(output_dir, exist_ok=True)

def convert_to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # for torch.Tensor scalar
        return obj.item()
    else:
        return str(obj)  # fallback for unknown types

def convert_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_keys_to_str(item) for item in obj]
    else:
        return obj

for dataset in available_datasets:
    data = reasoning_gym.create_dataset(dataset, size=task_size, seed=seed)
    output_path = os.path.join(output_dir, f'{dataset}.csv')

    with (open(output_path, 'w', newline='', encoding='utf-8') as csvfile):
        writer = csv.DictWriter(csvfile, fieldnames=['ID', 'dataset', 'question', 'answer', 'full_data'])
        writer.writeheader()

        for i, x in enumerate(data):
            # print(f'x: {x}')
            # Optional: validate that the answer is correct
            # assert data.score_answer(answer=x['answer'], entry=x) == 1.0

            question = x['question'] + '\nOutput final answer with the format <<<answer>>>.'

            if dataset not in ['futoshiki']:
                full_data = json.dumps(x, indent=4, ensure_ascii=False, default=convert_to_serializable)  # serialize full x dictionary as string

            if dataset == 'arc_agi':
                question = x['question'].replace('Your final answer should just be the text output grid itself.',
                                                 'Your final answer should be the output grid enclosed in triple angle brackets, like this: <<<output grid>>>.')
            elif dataset == 'bf':
                question = x['question'].replace('Respond only with the exact output of the program.',
                                                 'Respond only with the exact output of the program enclosed in triple angle brackets, like this: <<<output>>>.')
            elif dataset == 'binary_matrix':
                question = x['question'] + '\nYour final answer should be the output matrix enclosed in triple angle brackets, like this <<<output matrix>>>.'
            # elif dataset == 'boxnet':
            #     question = x['question'] + '\nOutput action plan enclosed in triple angle brackets, like this <<<action plan>>>.'
            elif dataset == 'codeio':
                question = x['question'].replace(' without writing any code', '').replace('in the form of a JSON object', 'in the form of a JSON object enclosed in triple angle brackets, like this <<<JSON object>>>')
            elif dataset == 'cryptarithm':
                question = x['question'].replace('Output format: "', 'Output format: <<<'
                                                 ).replace('" (without quotes)', '>>>')
            elif dataset == 'futoshiki':
                question = x['question'].replace('Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell.',
                                                 'Ensure your answer follows the same format as the puzzle above, just replace blanks (_) with the correct value for the cell and put the answer in triple angle brackets, like this <<<answer to the puzzle>>>.')
                full_data = json.dumps(convert_keys_to_str(x), indent=4, ensure_ascii=False, default=convert_to_serializable)  # serialize full x dictionary as string
            elif dataset == 'game_of_life':
                question = x['question'] + '\nOutput grid in JSON format enclosed in triple angle brackets, like this <<<grid in JSON format>>>.'
            elif dataset == 'game_of_life_halting':
                question = x['question'] + '\nOutput final answer with the format <<<answer>>>, i.e. <<<True>>> or <<<False>>>.'
            elif dataset == 'gcd':
                question = x['question'].replace('Give only the GCD as your final answer.',
                                                 'Give only the GCD enclosed in triple angle brackets as your final answer, like <<<GCD>>>.')
            elif dataset == 'graph_color':
                question = x['question'].replace('Return your solution as a JSON map of vertices to colors. (For example: {"0": 1, "1": 2, "2": 3}.)',
                                                 'Return your solution as a JSON map of vertices to colors enclosed in triple angle brackets, like this <<<JSON map of vertices to colors>>>. (For example: <<<{"0": 1, "1": 2, "2": 3}>>>.)')
            elif dataset == 'group_anagrams':
                question = x['question'].replace('The output is a list of lists of strings, where each outer list contains a group of anagrams, e.g. [["eat", "tea"], ["tan", "nat"]].',
                                                 'The output is a list of lists of strings enclosed in triple angle brackets, where each outer list contains a group of anagrams, e.g. <<<[["eat", "tea"], ["tan", "nat"]]>>>.')
            elif dataset == 'jugs':
                question = x['question'].replace('Reply as a JSON-parsable list of moves which result in any of the jugs being filled with the target amount.',
                                                 'Reply as a JSON-parsable list of moves which result in any of the jugs being filled with the target amount, and enclose the JSON-parsable list in triple angle brackets, like <<<JSON-parsable list>>>.')
            elif dataset == 'knight_swap':
                question = x['question'] + '- Output final answer with the format <<<answer>>>'
            elif dataset == 'knights_knaves':
                question = x['question'].replace('(Format your answer like: "',
                                                 '(Enclose your answer in triple angle brackets like: <<<')
                question = question[:-2] + '>>>)'
            elif dataset == 'letter_jumble':
                question = x['question'].replace('Your output should be a sentence with the words unscrambled.',
                                                 'Your output should be a sentence with the words unscrambled, and enclose the sentence in triple angle brackets, like <<<sentence>>>.')
            elif dataset == 'list_functions':
                question = x['question'].replace('Your answer should be a list of element/elements',
                                                 'Your answer should be a list of element/elements enclosed in triple angle brackets, like <<<list of element/elements>>>')
            # elif dataset == 'mahjong_puzzle':
            #     question = x['question'].replace('Your output should be one of the following: "Peng", "Chi", or "Pass" (without quotes).',
            #                                      'Your output should be one of the following: "<<<Peng>>>", "<<<Chi>>>", or "<<<Pass>>>" (without quotes).')
            elif dataset == 'manipulate_matrix':
                question = x['question'] + '\nOutput the final matrix after performing all the operations, ensuring it matches the input matrix format and is enclosed within triple angle brackets, like <<<0 1 2 7\n6 3 5 2\n9 0 2 7>>>.'
            elif dataset == 'maze':
                question = x['question'].replace('Give only the number of steps as your final answer, no other text or formatting.',
                                                 'Give only the number of steps as your final answer, and enclose it within triple angle brackets, like <<<10>>>.')
            elif dataset == 'mini_sudoku':
                question = x['question'] + 'Enclose your response within triple angle brackets, like <<<1 2 3 4\n2 3 4 1\n3 4 1 2\n4 1 2 3>>>.'
            elif dataset == 'modulo_grid':
                question = x['question'].replace('Return the entire completed grid as your answer.',
                                                 'Return the entire completed grid as your answer and enclose it within triple angle brackets, like <<<the entire completed grid>>>.')
            elif dataset == 'n_queens':
                question = x['question'] + '\nEnclose the output board within triple angle brackets, like <<<output board>>>.'
            elif dataset == 'needle_haystack':
                question = x['question'].replace('Reply only with a name.', 'Reply with a name enclosed within triple angle brackets, like <<<name>>>.')
            elif dataset == 'number_filtering':
                question = x['question'].replace('Return the new list in the same format.', 'Return the new list in the same format enclosed within triple angle brackets, like <<<the new list>>>.')
            elif dataset == 'number_format':
                question = x['question'].replace('Your output should be only the number of interest.', 'Your output should be only the number of interest enclosed within triple angle brackets, like <<<the number of interest>>>.')
            elif dataset == 'number_sequence':
                question = x['question'] + '\nOutput the value of "?" enclosed within triple angle brackets, like <<<the value of "?">>>.'
            elif dataset == 'number_sorting':
                question = x['question'].replace("['-69', '-13', '1', '7', '11', '43', '59', '61']", "<<<['-69', '-13', '1', '7', '11', '43', '59', '61']>>>")
            elif dataset == 'palindrome_generation':
                question = x['question'].replace('Your output should be a single string, with no spaces or punctuation.',
                                                 'Your output should be a single string, with no spaces or punctuation, and enclose it within triple angle brackets, like <<<single string>>>.')
            elif dataset == 'palindrome_partitioning':
                question = x['question'].replace('Your output should be a list of lists, where each list represents a palindrome partition, e.g. [["a","a","b"],["aa","b"]].',
                                                 'Your output should be a list of lists enclosed within triple angle brackets, where each list represents a palindrome partition, e.g. <<<[["a","a","b"],["aa","b"]]>>>.')
            elif dataset == 'polynomial_equations':
                question = x['question'].replace('- For 3 or more solutions: all values as comma-separated decimal numbers',
                                                 '- For 3 or more solutions: all values as comma-separated decimal numbers\n   - Enclose the answer within triple angle brackets, like <<<number/numbers>>>.')
            elif dataset == 'pool_matrix':
                question = x['question'].replace('Your output should be a matrix in the same format as the input matrix.',
                                                 'Your output should be a matrix in the same format as the input matrix, enclosed within triple angle brackets, like <<<output matrix>>>.')
            elif dataset == 'prime_factorization':
                question = x['question'].replace('Write the factors separated by × (Example: for 12 the answer would be: 2 × 2 × 3)',
                                                 'Write the factors separated by × and enclose the result within triple angle brackets (Example: for 12 the answer would be: <<<2 × 2 × 3>>>)')
            elif dataset == 'products':
                question = x['question'].replace('Give only the result as your final answer',
                                                 'Give the result enclosed within triple angle brackets as your final answer, like <<<result>>>.')
            elif dataset == 'propositional_logic':
                question = x['question'].replace('Return the conclusion logic statement, as your final answer.',
                                                 'Return the conclusion logic statement enclosed within triple angle brackets, as your final answer, like <<<the conclusion logic statement>>>.')
            elif dataset == 'puzzle24':
                question = x['question'] + '5. Enclose the final answer within triple angle brackets, like <<<the final answer>>>.'
            elif dataset == 'quantum_lock':
                question = x['question'].replace("Your answer should be a sequence of buttons separated by '→', for example: A → B → C",
                                                 "Your answer should be a sequence of buttons separated by '→' enclosed within triple angle brackets, for example: <<<A → B → C>>>")
            elif dataset == 'ransom_note':
                question = x['question'].replace('return True if you can construct the ransom note using the letters in the magazine, and False otherwise.',
                                                 'return True if you can construct the ransom note using the letters in the magazine, and False otherwise. Make sure to enclose the answer within triple angle brackets, i.e. "<<<True>>>"/"<<<False>>>".')
            elif dataset == 'rearc':
                question = x['question'].replace('Your final answer should just be the text output grid itself.',
                                                 'Your final answer should be the text output grid enclosed within triple angle brackets, like <<<the text output grid>>>.')
            elif dataset == 'rectangle_count':
                question = x['question'].replace('Your output should be a single number, representing the total count of rectangles.',
                                                 'Your output should be a single number, representing the total count of rectangles, enclosed within triple angle brackets, like <<<the total count of rectangles>>>.')
            elif dataset == 'rotate_matrix':
                question = x['question'].replace('Your output should be a matrix in the same format as the input.',
                                                 'Your output should be a matrix in the same format as the input, enclosed within triple angle brackets, like <<<the output matrix>>>.')
            elif dataset == 'rotten_oranges':
                question = x['question'] + '\nEnclose final answer within triple angle brackets, i.e. <<<number of minutes>>> or <<<-1>>>.'
            elif dataset == 'rubiks_cube':
                question = x['question'] + '\nEnclose the solution within triple angle brackets, i.e. <<<the solution>>>.'
            elif dataset == 'rush_hour':
                question = x['question'].replace("Specify moves in the format: 'F+1 K+1 M-1 C+3 H+2 ...'",
                                                 "Specify moves in the format: '<<<F+1 K+1 M-1 C+3 H+2 ...>>>'")
            elif dataset == 'self_reference':
                question = x['question'] + '\nEnclose final answer within triple angle brackets, i.e. <<<the number of possible solutions>>>.'
            # elif dataset == 'sentence_reordering':
            #     question = x['question'] + '\nEnclose final answer within triple angle brackets, i.e. <<<the restored sentence>>>.'
            elif dataset == 'shortest_path':
                question = x['question'].replace('Your output should be a sequence of directions that leads from * to #, e.g. right right down down up left',
                                                 'Your output should be a sequence of directions that leads from * to #, enclosed within triple angle brackets, e.g. <<<right right down down up left>>>')
            elif dataset == 'simple_geometry':
                question = x['question'].replace('Return only the angle as your answer.',
                                                 'Return only the angle enclosed within triple angle brackets as your answer, like <<<the angle>>>. ')
            elif dataset == 'simple_integration':
                question = x['question'] + '\nEnclose final answer within triple angle brackets, i.e. <<<the result of calculation>>>'
            elif dataset == 'sokoban':
                question = x['question'].replace('Your solution must be a string of characters',
                                                 'Your solution must be a string of characters enclosed within triple angle brackets, like <<<a string of characters>>>.')
            elif dataset == 'spell_backward':
                question = x['question'] + '\nEnclose final answer within triple angle brackets, i.e. <<<the answer>>>'
            elif dataset == 'spiral_matrix':
                question = x['question'].replace('Your output should be a space-separated list of integers, e.g. 1 2 3 4 5 6',
                                                 'Your output should be a space-separated list of integers enclosed within triple angle brackets, e.g. <<<1 2 3 4 5 6>>>')
            elif dataset == 'string_insertion':
                question = x['question'].replace('Your output should be a string that has been modified according to the pattern.',
                                                 'Your output should be a string that has been modified according to the pattern, enclosed within triple angle brackets, like <<<the string that has been modified>>>.')
            elif dataset == 'string_manipulation':
                question = x['question'].replace('Your output should be the final transformed string after applying all the rules.',
                                                 'Your output should be the final transformed string after applying all the rules, enclosed within triple angle brackets, like <<<the final transformed string>>>.')
            elif dataset == 'string_splitting':
                question = x['question'].replace(
                    """The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
For example 1 0 1 5 4 3 means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z.""",
                    """The output should be the count of each machine and part type after the rules have been exhaustively applied in the following order: A B C X Y Z.
Enclose final answer within triple angle brackets, i.e. <<<the count of each machine and part type>>>.
For example <<<1 0 1 5 4 3>>> means that you have 1 machine A, 0 machine B, 1 machine C, 5 part X, 4 part Y, and 3 part Z."""
                )
            elif dataset == 'string_synthesis':
                question = x['question'].replace(
                    """The output should be the count of each block type after the rules have been applied in the order they are listed above.
For example 1 0 3 0 2 0 0 0 1 means that you have 1 [A] 0 [B] 3 [C] 0 {A} 2 {B} 0 {C} 0 (A) 0 (B) 1 (C).""",
                    """The output should be the count of each block type after the rules have been applied in the order they are listed above.
Enclose final answer within triple angle brackets, i.e. <<<the count of each block type>>>.
For example <<<1 0 3 0 2 0 0 0 1>>> means that you have 1 [A] 0 [B] 3 [C] 0 {A} 2 {B} 0 {C} 0 (A) 0 (B) 1 (C)."""
                )
            elif dataset == 'sudoku':
                question = x['question'] + '\nEnclose final answer within triple angle brackets, i.e. <<<the 9x9 grid>>>.'
            elif dataset == 'syllogism':
                question = x['question'] + '\n\nEnclose final answer within triple angle brackets, i.e. <<<Yes>>> or <<<No>>>.'
            elif dataset == 'time_intervals':
                question = x['question'] + '\nEnclose final answer within triple angle brackets, like <<<the answer>>>.'
            elif dataset == 'tower_of_hanoi':
                question = x['question'].replace('Do not include any other text or formatting.',
                                                 'Enclose final answer within triple angle brackets, like <<<the sequence of moves>>>.')
            elif dataset == 'tsumego':
                question = x['question'].replace("Specify your move in coordinates (e.g. 'C4' for column C, row 4)",
                                                 "Specify your move in coordinates enclosed within triple angle brackets (e.g. '<<<C4>>>' for column C, row 4)")
            elif dataset == 'word_ladder':
                question = x['question'].replace('Provide your answer as a comma-separated sequence of uppercase letters without spaces.',
                                                 'Provide your answer as a comma-separated sequence of uppercase letters without spaces, enclosed within triple angle brackets, like <<<your answer>>>.')
            elif dataset == 'word_sequence_reversal':
                question = x['question'].replace('Provide you answer as a comma-separated list of words with a space after the comma.',
                                                 'Provide your answer as a comma-separated list of words with a space after the comma, enclosed within triple angle brackets, like <<<your answer>>>.')
            elif dataset == 'word_sorting':
                question = x['question'].replace('Your output should be a comma-separated list of words, e.g. word_1, word_2, word_3',
                                                 'Your output should be a comma-separated list of words, enclosed within triple angle brackets, e.g. <<<word_1, word_2, word_3>>>.')
            elif dataset == 'zebra_puzzles':
                question = x['question'].replace('Provide only the name of the person as your final answer.',
                                                 'Provide the name of the person enclosed within triple angle brackets as your final answer, like <<<the name of the person>>>.')

            writer.writerow({
                'ID': i,
                'dataset': dataset,
                'question': question,
                'answer': x['answer'],
                'full_data': full_data
            })

    print(f"Saved {task_size} tasks from '{dataset}' to '{output_path}'")
# print(reasoning_gym.factory.DATASETS)