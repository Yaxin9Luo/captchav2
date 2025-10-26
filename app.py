import os
import json
import random
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', template_folder='templates')

# Dictionary to track which puzzles have been shown for each CAPTCHA type
seen_puzzles = {}
# List to track recently used CAPTCHA types to avoid repetition
recent_types = []
# How many types to remember before allowing repetition
MAX_RECENT_TYPES = 5

PUZZLE_TYPE_SEQUENCE = [
    'Dice_Count',
    'Bingo',
    'Shadow_Plausible',
    'Mirror',
    'Deformation'
]
sequential_index = 0

# Load ground truth data for a specific type
def load_ground_truth(captcha_type):
    path = os.path.join('captcha_data', captcha_type, 'ground_truth.json')
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

# Get available CAPTCHA types
def get_captcha_types():
    base_dir = 'captcha_data'
    if not os.path.exists(base_dir):
        return []
    return [d for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/captcha_data/<captcha_type>/<filename>')
def serve_captcha(captcha_type, filename):
    return send_from_directory(os.path.join('captcha_data', captcha_type), filename)

@app.route('/captcha_data/<captcha_type>/<subdir>/<filename>')
def serve_captcha_subdir(captcha_type, subdir, filename):
    return send_from_directory(os.path.join('captcha_data', captcha_type, subdir), filename)

@app.route('/api/get_puzzle', methods=['GET'])
def get_puzzle():
    global recent_types
    
    # Check if we should return a random puzzle from any type
    is_random = request.args.get('random', 'false').lower() == 'true'
    
    # Get all available CAPTCHA types
    captcha_types = get_captcha_types()
    if not captcha_types:
        return jsonify({'error': 'No CAPTCHA types found'}), 404
    
    # Check if we're in debug mode for a specific type
    debug_type = request.args.get('debug_type')

    mode = request.args.get('mode', '').lower()

    if debug_type and debug_type in captcha_types:
        puzzle_type = debug_type
    elif not is_random and mode == 'sequential':
        global sequential_index
        puzzle_type = PUZZLE_TYPE_SEQUENCE[sequential_index % len(PUZZLE_TYPE_SEQUENCE)]
        sequential_index += 1
    elif is_random:
        # Select a random CAPTCHA type, avoiding recently used types if possible
        available_types = [t for t in captcha_types if t not in recent_types]
        
        # If all types have been used recently, reset the tracking
        if not available_types:
            recent_types = []
            available_types = captcha_types
        
        puzzle_type = random.choice(available_types)
        
        # Add to recent types and maintain maximum length
        recent_types.append(puzzle_type)
        if len(recent_types) > MAX_RECENT_TYPES:
            recent_types.pop(0)
    else:
        # Get puzzle type from query parameter
        puzzle_type = request.args.get('type', 'Dice_Count')
        # Check if puzzle type exists
        if puzzle_type not in captcha_types:
            return jsonify({'error': f'Invalid puzzle type: {puzzle_type}'}), 400
    
    # Load ground truth for the selected type
    ground_truth = load_ground_truth(puzzle_type)
    if not ground_truth:
        return jsonify({'error': f'No puzzles found for type: {puzzle_type}'}), 404
    
    puzzle_files = list(ground_truth.keys())
    
    # Select a random puzzle, avoiding repetition if possible
    if puzzle_type not in seen_puzzles:
        seen_puzzles[puzzle_type] = set()
    
    # Get unseen puzzles
    unseen_puzzles = [p for p in puzzle_files if p not in seen_puzzles[puzzle_type]]
    
    # If all puzzles have been seen, reset the tracking
    if not unseen_puzzles:
        seen_puzzles[puzzle_type] = set()
        unseen_puzzles = puzzle_files
    
    # Select a random puzzle from unseen ones
    selected_puzzle = random.choice(unseen_puzzles)
    
    # Mark this puzzle as seen
    seen_puzzles[puzzle_type].add(selected_puzzle)
    
    # Get the appropriate question prompt based on puzzle type
    if puzzle_type == "Dice_Count":
        prompt = ground_truth[selected_puzzle].get('prompt', "Sum up the numbers on all the dice")
    elif puzzle_type == "Bingo":
        prompt = ground_truth[selected_puzzle].get("prompt", "Please click two images to exchange their position to line up the same images to a line")
    elif puzzle_type == "Shadow_Plausible":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "Select every image that shows a physically plausible shadow."
        )
    elif puzzle_type == "Mirror":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "Select all mirror images that do not match the reference object."
        )
    elif puzzle_type == "Deformation":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "If I release the objects in the left image, choose the right image that shows the correct deformation."
        )
    else:
        prompt = ground_truth[selected_puzzle].get("prompt", "Solve the CAPTCHA puzzle")
    
    # Add input_type to tell the frontend what kind of input to show
    input_type = "text"
    if puzzle_type == "Dice_Count":
        input_type = "number"
    elif puzzle_type == "Bingo":
        input_type = "bingo_swap"
    elif puzzle_type == "Shadow_Plausible":
        input_type = "shadow_plausible"
    elif puzzle_type == "Mirror":
        input_type = "mirror_select"
    elif puzzle_type == "Deformation":
        input_type = "deformation_select"

    
    # For Rotation_Match, include additional data needed for the interface
    additional_data = {}
    
    # For Bingo, include the grid size
    if puzzle_type == "Bingo":
        # Get grid size from ground truth
        grid_size = ground_truth[selected_puzzle].get("grid_size", [3, 3])  # Default to 3x3 grid if not specified
        
        additional_data = {
            "grid_size": grid_size,
            "solution_line": ground_truth[selected_puzzle].get("solution_line", {}),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Mirror":
        reference_image = ground_truth[selected_puzzle].get("reference")
        option_images = ground_truth[selected_puzzle].get("options", [])
        if not reference_image or not option_images:
            return jsonify({'error': f'Invalid mirror data: {selected_puzzle}'}), 500

        additional_data = {
            "reference_image": f'/captcha_data/{puzzle_type}/{reference_image}',
            "option_images": [f'/captcha_data/{puzzle_type}/{img}' for img in option_images],
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [1, len(option_images)]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Deformation":
        reference_image = ground_truth[selected_puzzle].get("reference")
        option_images = ground_truth[selected_puzzle].get("options", [])
        if not reference_image or not option_images:
            return jsonify({'error': f'Invalid deformation data: {selected_puzzle}'}), 500

        additional_data = {
            "reference_image": f'/captcha_data/{puzzle_type}/{reference_image}',
            "option_images": [f'/captcha_data/{puzzle_type}/{img}' for img in option_images],
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [2, 2]),
            "answer": ground_truth[selected_puzzle].get("answer")
        }
    elif puzzle_type == "Squiggle":
        reference_image = ground_truth[selected_puzzle].get("reference")
        if not reference_image:
            return jsonify({'error': f'Invalid squiggle data: {selected_puzzle}'}), 500

        additional_data = {
            "reference_image": f'/captcha_data/{puzzle_type}/{reference_image}',
            "min_path_length": ground_truth[selected_puzzle].get("min_length", 200),
            "min_accuracy": ground_truth[selected_puzzle].get("min_accuracy", 0.6),
            "min_shape_coverage": ground_truth[selected_puzzle].get("min_shape_coverage", 0.0),
            "ink_threshold": ground_truth[selected_puzzle].get("ink_threshold", 180),
            "sample_step": ground_truth[selected_puzzle].get("sample_step", 3),
            "answer": ground_truth[selected_puzzle].get("answer", {})
        }

    else:
        prompt = ground_truth[selected_puzzle].get("prompt", "Solve the CAPTCHA puzzle")

    image_path = None
    if puzzle_type not in ("Rotation_Match", "Shadow_Plausible", "Mirror", "Deformation", "Squiggle"):
        image_path = f'/captcha_data/{puzzle_type}/{selected_puzzle}'

    response_data = {
        'puzzle_type': puzzle_type,
        'image_path': image_path,
        'puzzle_id': selected_puzzle,
        'prompt': prompt,
        'input_type': input_type,
        'debug_info': f"Type: {puzzle_type}, Input: {input_type}, Puzzle: {selected_puzzle}"
    }
    
    # Add any additional data for specific puzzle types
    if additional_data:
        response_data.update(additional_data)

    return jsonify(response_data)

@app.route('/api/get_ground_truth', methods=['POST'])
def get_ground_truth():
    """Return ground truth data for debugging purposes"""
    data = request.json
    puzzle_type = data.get('puzzle_type')
    puzzle_id = data.get('puzzle_id')
    
    if not puzzle_type or not puzzle_id:
        return jsonify({'error': 'Missing puzzle_type or puzzle_id'}), 400
    
    ground_truth = load_ground_truth(puzzle_type)
    
    if puzzle_id not in ground_truth:
        return jsonify({'error': 'Invalid puzzle ID'}), 400
    
    # Return the ground truth for the specified puzzle
    puzzle_data = ground_truth[puzzle_id]
    
    return jsonify({
        'answer': puzzle_data.get('answer'),
        'question': puzzle_data.get('question'),
        'description': puzzle_data.get('description')
    })

@app.route('/api/check_answer', methods=['POST'])
def check_answer():
    data = request.json
    puzzle_type = data.get('puzzle_type', 'Dice_Count')
    puzzle_id = data.get('puzzle_id')
    user_answer = data.get('answer')
    elapsed_time = float(data.get('elapsed_time', 0))

    
    # Validate input
    if not puzzle_id or user_answer is None:
        return jsonify({'error': 'Missing puzzle_id or answer'}), 400
    
    ground_truth = load_ground_truth(puzzle_type)
    
    if puzzle_id not in ground_truth:
        return jsonify({'error': 'Invalid puzzle ID'}), 400
    
    # Get correct answer based on puzzle type
    is_correct = False
            
    
    if puzzle_type == 'Bingo':
        # For Bingo, check if the swapped positions would create a line of matching images
        try:
            # Get the expected correct swap options from ground truth
            correct_swaps = ground_truth[puzzle_id].get('answer', [])
            
            # User answer should be a list of two indices to swap
            user_swaps = user_answer
            
            # Check if the swaps match any of the possible correct swaps
            # For this puzzle, there can be multiple correct solutions
            is_correct = False
            
            # Go through each possible solution
            for correct_swap in correct_swaps:
                # Check if user's swap matches this solution (order doesn't matter)
                if (set(user_swaps) == set(correct_swap) or 
                    (set(user_swaps) == set(correct_swap[::-1]) if len(correct_swap) == 2 else False)):
                    is_correct = True
                    break
                    
            correct_answer_info = correct_swaps
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Bingo'}), 400
    
    
    
    elif puzzle_type == 'Shadow_Plausible':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Shadow_Plausible'}), 400

    elif puzzle_type == 'Mirror':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Mirror'}), 400

    elif puzzle_type == 'Deformation':
        try:
            correct_index = int(ground_truth[puzzle_id].get('answer'))
            user_index = int(user_answer)
            is_correct = user_index == correct_index
            correct_answer_info = correct_index
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Deformation'}), 400
    else:
        # For other types, compare as strings (case insensitive)
        correct_answer = ground_truth[puzzle_id].get('answer')
        is_correct = str(user_answer).lower() == str(correct_answer).lower()
        correct_answer_info = correct_answer
    
    # Get the appropriate answer field based on puzzle type
    if puzzle_type == 'Dice_Count':
        answer_key = 'sum'
    else:
        answer_key = 'answer'
    
    return jsonify({
        'correct': is_correct,
        'user_answer': user_answer,
        'correct_answer': ground_truth[puzzle_id].get(answer_key)
    })

@app.route('/api/benchmark_results', methods=['POST'])
def record_benchmark():
    data = request.json
    
    # Add timestamp if not provided
    if 'timestamp' not in data:
        from datetime import datetime
        data['timestamp'] = datetime.now().isoformat()
    
    # In a real system, you would save this data to a database
    # For this example, we'll just print it to the console
    print(f"Benchmark results: {data}")
    
    # You could store this in a log file as well
    with open('benchmark_results.json', 'a') as f:
        f.write(json.dumps(data) + '\n')
    
    return jsonify({'status': 'success'})

@app.route('/api/types', methods=['GET'])
def get_types():
    """Get available CAPTCHA types"""
    return jsonify({
        'types': get_captcha_types()
    })

if __name__ == '__main__':
    # For local development
    if os.environ.get('DEVELOPMENT'):
        app.run(debug=True)
    else:
        # For production on Hugging Face Spaces
        app.run(host='0.0.0.0', port=7860) 
