def analyze_computational_approach(code_string):
    """
    Analyzes Python code to determine if it uses searching, symbolic computing,
    or numerical computing approaches.

    Returns a dictionary with analysis results and explanations.
    """
    import ast
    import re

    analysis = {
        'searching_patterns': False,
        'symbolic_computing': False,
        'numerical_computing': False,
        'explanations': []
    }

    # Parse the code into an AST
    try:
        tree = ast.parse(code_string)
    except Exception as e:
        return {
            'error': True,
            'message': f"Invalid Python code: {str(e)}",
            'searching_patterns': False,
            'symbolic_computing': False,
            'numerical_computing': False,
            'explanations': [],
            'complexity_score': 0
        }

    # Searching patterns indicators
    searching_indicators = {
        'itertools': ['permutations', 'combinations', 'product'],
        'loops': ['for', 'while'],
        'comprehensive_search': ['all', 'any', 'filter', 'map']
    }

    # Symbolic computing indicators
    symbolic_indicators = {
        'string_expressions': ['eval', 'exec'],
        'dynamic_expression_building': ['+', '-', '*', '/', '(', ')'],
        'string_formatting': ['format', 'f"', "f'"]
    }

    # Numerical computing indicators
    numerical_indicators = {
        'math_operations': ['math.', 'numpy.', 'scipy.'],
        'complex_calculations': ['sum', 'abs', 'pow', 'sqrt']
    }

    # Initialize complexity score
    complexity_score = 0

    # Check for searching patterns
    for node in ast.walk(tree):
        # Check for iterations and loops
        if isinstance(node, (ast.For, ast.While)):
            analysis['searching_patterns'] = True
            analysis['explanations'].append("Contains loops for systematic search")

        # Check for itertools usage
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            module_name = node.names[0].name
            if 'itertools' in module_name:
                analysis['searching_patterns'] = True
                analysis['explanations'].append("Uses itertools for combinatorial search")

    # Check for symbolic computing
    if 'eval(' in code_string or 'exec(' in code_string:
        analysis['symbolic_computing'] = True
        analysis['explanations'].append("Uses eval/exec for symbolic expression evaluation")

    # Check for f-strings and string formatting
    for format_indicator in symbolic_indicators['string_formatting']:
        if format_indicator in code_string:
            analysis['symbolic_computing'] = True
            analysis['explanations'].append("Uses string formatting for expression building")
            break

    # Check for numerical computing and count mathematical operations
    math_ops_count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            analysis['numerical_computing'] = True
            analysis['explanations'].append("Contains direct mathematical operations")
            math_ops_count += 1

    # Count complexity indicators
    complexity_score += len(re.findall(r'for|while', code_string))  # Full weight for loops
    complexity_score += len(re.findall(r'if|elif|else', code_string))  # Full weight for conditionals
    complexity_score += len(re.findall(r'try|except', code_string))  # Full weight for error handling
    complexity_score += math_ops_count * 0.25  # Half weight for mathematical operations

    analysis['complexity_score'] = complexity_score

    return analysis

def analyze_code_and_explain(code):
    """Analyzes code and provides a human-readable explanation."""
    results = analyze_computational_approach(code)

    if 'error' in results:
        return results['message'], results['complexity_score']

    explanation = []
    if results['searching_patterns']:
        explanation.append("SEARCHING APPROACH detected because:")
        explanation.extend([f"- {exp}" for exp in results['explanations'] if 'search' in exp.lower()])

    if results['symbolic_computing']:
        explanation.append("\nSYMBOLIC COMPUTING detected because:")
        explanation.extend([f"- {exp}" for exp in results['explanations'] if 'symbol' in exp.lower() or 'expression' in exp.lower()])

    if results['numerical_computing']:
        explanation.append("\nNUMERICAL COMPUTING detected because:")
        explanation.extend([f"- {exp}" for exp in results['explanations'] if 'mathematical' in exp.lower()])

    explanation.append(f"\nComplexity score: {results['complexity_score']}")

    return "\n".join(explanation), results['complexity_score']

# Now let's test with your actual code examples:
find_24_code = '''
from itertools import permutations, product

def find_24(numbers):
    ops = ['+', '-', '*', '/']
    for num_perm in permutations(numbers):
        for op_perm in product(ops, repeat=3):
            expressions = [
                f"({num_perm[0]} {op_perm[0]} {num_perm[1]}) {op_perm[1]} ({num_perm[2]} {op_perm[2]} {num_perm[3]})",
                f"(({num_perm[0]} {op_perm[0]} {num_perm[1]}) {op_perm[1]} {num_perm[2]}) {op_perm[2]} {num_perm[3]}",
                f"({num_perm[0]} {op_perm[0]} ({num_perm[1]} {op_perm[1]} {num_perm[2]})) {op_perm[2]} {num_perm[3]}",
                f"{num_perm[0]} {op_perm[0]} (({num_perm[1]} {op_perm[1]} {num_perm[2]}) {op_perm[2]} {num_perm[3]})",
                f"{num_perm[0]} {op_perm[0]} ({num_perm[1]} {op_perm[1]} ({num_perm[2]} {op_perm[2]} {num_perm[3]}))"
            ]
            for expr in expressions:
                try:
                    if abs(eval(expr) - 24) < 1e-6:
                        return f"<<<{expr} = 24>>>"
                except ZeroDivisionError:
                    continue
    return "No solution found."
'''

simple_code = '''
a, b, c, d = 1, 3, 4, 6
result = (d - a) * (c - b)
print(f"<<<({d} - {a}) * ({c} - {b}) = {result}>>>")
'''

boxlift_code = '''
boxes = [298, 266, 385, 179, 63, 315, 133, 66, 367, 146, 123, 231, 118, 224, 322, 316, 196, 50, 293, 357, 55, 48, 230, 253, 344]
lifters = [175, 128, 130, 131, 116, 198]

# Sort boxes and lifters in descending order
sorted_boxes = sorted(enumerate(boxes), key=lambda x: -x[1])  # (index, weight)
sorted_lifters = sorted(enumerate(lifters), key=lambda x: -x[1])  # (index, capacity)

from itertools import combinations

total_steps = 9  # Maximum allowed steps
steps = []       # To store the steps with assignments
boxes_lifted = set()

for step_num in range(total_steps):
    lifters_available = set(range(len(lifters)))  # All lifters are available at the start of each step
    boxes_in_step = []
    for box_idx, box_weight in sorted_boxes:
        if box_idx in boxes_lifted:
            continue  # Skip if the box is already lifted
        lifter_indices = list(lifters_available)
        # Try to find a combination of lifters who can lift the box
        lifted = False
        for r in range(1, len(lifter_indices)+1):
            for combo in combinations(lifter_indices, r):
                total_capacity = sum(lifters[i] for i in combo)
                if total_capacity >= box_weight:
                    # Assign this box to these lifters
                    boxes_in_step.append((box_weight, list(combo)))
                    boxes_lifted.add(box_idx)
                    for i in combo:
                        lifters_available.remove(i)  # Mark lifters as used in this step
                    lifted = True
                    break
            if lifted:
                break  # Move to the next box after assigning
        if not lifters_available:
            break  # All lifters are used in this step
    if boxes_in_step:
        steps.append(boxes_in_step)
    else:
        break  # No more boxes can be lifted
    if len(boxes_lifted) == len(boxes):
        break  # All boxes are lifted

# Output the steps in the required format
print("<<<")
for idx, step in enumerate(steps):
    step_output = []
    for box_weight, lifter_idxs in step:
        step_output.append((box_weight, lifter_idxs))
    print(f"Step {idx + 1}: {step_output}")
print(">>>")
'''

number_multiply_code = '''
# Calculation steps:
# 1. Multiply 5 by 911
# 2. Multiply the result from step 1 by 7485
# 3. Print the final result

result = 5 * 911 * 7485
print(result)
'''

letters_code = '''
word = 'antidisestablishmentarian'
count = 0
positions = []

for index, char in enumerate(word):
    if char == 'a':
        count += 1
        positions.append(index + 1)

result = f"<<<Count: {count}, Positions: {positions}>>>"
print(result)
'''

'''
print("Analysis of find_24.py:")
summary, complexity_score = analyze_code_and_explain(find_24_code)
print(summary)
print(f"Complexity score: {complexity_score}")


print("\nAnalysis of simple calculation:")
summary, complexity_score = analyze_code_and_explain(simple_code)
print(summary)
print(f"Complexity score: {complexity_score}")


print("\nAnalysis of boxlift.py:")
summary, complexity_score = analyze_code_and_explain(boxlift_code)
print(summary)
print(f"Complexity score: {complexity_score}")

print("\nAnalysis of number_multiply.py:")
summary, complexity_score = analyze_code_and_explain(number_multiply_code)
print(summary)
print(f"Complexity score: {complexity_score}")

print("\nAnalysis of letters.py:")
summary, complexity_score = analyze_code_and_explain(letters_code)
print(summary)
print(f"Complexity score: {complexity_score}")
'''