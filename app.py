import os
import json
import random
import uuid
import time
import math
import base64
import io
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

app = Flask(__name__, static_folder='static', template_folder='templates')

# Dictionary to track which puzzles have been shown for each CAPTCHA type
seen_puzzles = {}
# List to track recently used CAPTCHA types to avoid repetition
recent_types = []
# How many types to remember before allowing repetition
MAX_RECENT_TYPES = 5

PUZZLE_TYPE_SEQUENCE = [
    # 'Dice_Count',
    # 'Shadow_Plausible',
    # 'Mirror',
    # 'Squiggle',
    # 'Color_Cipher',
    # 'Color_Counting',
    # 'Trajectory_Recovery',
    # 'Spooky_Size',
    # 'Spooky_Circle',
    # 'Spooky_Circle_Grid',
    # 'Spooky_Shape_Grid',
    # 'Spooky_Text',
    # 'Red_Dot',
    # 'Storyboard_Logic',
    'Jigsaw_Puzzle',
]
sequential_index = 0

active_red_dot_puzzles: dict[str, dict] = {}
active_spooky_size_puzzles: dict[str, dict] = {}
active_jigsaw_puzzles: dict[str, dict] = {}

COLOR_SYMBOL_POOL = [
    ("🟥", "red"),
    ("🟧", "orange"),
    ("🟨", "yellow"),
    ("🟩", "green"),
    ("🟦", "blue"),
    ("🟪", "purple"),
    ("⬛", "black"),
    ("⬜", "white"),
]


def generate_color_cipher(config: dict) -> dict:
    """Create a unique Color Cipher puzzle definition."""
    symbol_count = config.get("symbol_count", 3)
    symbol_count = max(2, min(symbol_count, len(COLOR_SYMBOL_POOL)))

    # Pick distinct symbols and values
    selected_symbols = random.sample(COLOR_SYMBOL_POOL, symbol_count)
    low, high = config.get("value_range", [1, 12])
    value_pool = list(range(int(low), int(high) + 1))
    if len(value_pool) < symbol_count:
        value_pool = list(range(1, symbol_count + 5))
    values = random.sample(value_pool, symbol_count)

    mapping = []
    for (symbol, label), value in zip(selected_symbols, values):
        mapping.append({
            "symbol": symbol,
            "value": value,
            "label": label
        })

    term_count = 2 if symbol_count < 3 else random.choice([2, 3])
    operands = random.sample(mapping, term_count)
    available_ops = config.get("operations", ["+", "-"])
    if not available_ops:
        available_ops = ["+"]
    operations = [random.choice(available_ops) for _ in range(term_count - 1)]

    expression_parts = [operands[0]["symbol"]]
    total = operands[0]["value"]
    for op, operand in zip(operations, operands[1:]):
        expression_parts.extend([op, operand["symbol"]])
        if op == "+":
            total += operand["value"]
        elif op == "-":
            total -= operand["value"]
        elif op == "*":
            total *= operand["value"]
        else:
            total += operand["value"]

    expression = " ".join(expression_parts)
    question_template = config.get("question_template", "What is {expression}?")

    puzzle_id = f"color_cipher_{uuid.uuid4().hex}"
    cipher_state = {
        "mapping": mapping,
        "expression": expression
    }
    return {
        "puzzle_id": puzzle_id,
        "mapping": mapping,
        "question": question_template.format(expression=expression),
        "answer": total,
        "reveal_duration": config.get("reveal_duration", 3),
        "input_mode": config.get("input_type", "number"),
        "prompt": config.get(
            "prompt",
            "Keys flash briefly, then vanish. Remember the mapping before it disappears."
        ),
        "debug_expression": expression,
        "cipher_state": cipher_state
    }


def evaluate_color_cipher(expression: str, mapping: list[dict]) -> float:
    """Compute the numeric result of a color cipher expression."""
    if not expression or not mapping:
        raise ValueError('Missing expression or mapping')

    symbol_map = {}
    for entry in mapping:
        symbol = entry.get('symbol')
        value = entry.get('value')
        if symbol is None or value is None:
            continue
        symbol_map[str(symbol)] = float(value)

    tokens = expression.split()
    if not tokens:
        raise ValueError('Empty expression')

    try:
        result = symbol_map[tokens[0]]
    except KeyError as exc:
        raise ValueError(f'Unknown symbol {tokens[0]}') from exc

    idx = 1
    while idx < len(tokens):
        op = tokens[idx]
        if idx + 1 >= len(tokens):
            raise ValueError('Malformed expression')
        symbol = tokens[idx + 1]
        if symbol not in symbol_map:
            raise ValueError(f'Unknown symbol {symbol}')
        value = symbol_map[symbol]

        if op == '+':
            result += value
        elif op == '-':
            result -= value
        elif op == '*':
            result *= value
        elif op == '/':
            if value == 0:
                raise ValueError('Division by zero')
            result /= value
        else:
            raise ValueError(f'Unsupported operator {op}')

        idx += 2

    return result


def generate_jigsaw_image(width: int, height: int) -> Image.Image:
    """Generate a random image with diverse geometric shapes for jigsaw puzzle."""
    if not PIL_AVAILABLE:
        raise ValueError('PIL/Pillow is required for jigsaw puzzle generation')
    
    from PIL import ImageDraw
    
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Random light background color for contrast
    bg_color = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
    img.paste(bg_color, [0, 0, width, height])
    
    # Generate diverse geometric shapes - 5 to 10 shapes for variety
    num_shapes = random.randint(5, 10)
    shape_types = ['circle', 'rectangle', 'ellipse', 'polygon', 'triangle', 'star', 'rounded_rect', 'diamond', 'hexagon']
    
    # Track used positions to avoid too much overlap
    used_positions = []
    
    for _ in range(num_shapes):
        shape_type = random.choice(shape_types)
        
        # Random vibrant colors
        color = (random.randint(60, 230), random.randint(60, 230), random.randint(60, 230))
        
        # Random size - ensure shapes are visible
        size_factor = random.uniform(0.12, 0.35)  # 12% to 35% of image size
        shape_width = int(width * size_factor)
        shape_height = int(height * size_factor)
        
        # Try to place shape avoiding too much overlap
        max_attempts = 10
        for attempt in range(max_attempts):
            center_x = random.randint(shape_width // 2, width - shape_width // 2)
            center_y = random.randint(shape_height // 2, height - shape_height // 2)
            
            # Check if position is too close to existing shapes
            too_close = False
            for used_pos in used_positions:
                dist = math.sqrt((center_x - used_pos[0])**2 + (center_y - used_pos[1])**2)
                if dist < min(shape_width, shape_height):
                    too_close = True
                    break
            
            if not too_close or attempt == max_attempts - 1:
                used_positions.append((center_x, center_y))
                break
        
        # Draw the shape
        if shape_type == 'circle':
            radius = min(shape_width, shape_height) // 2
            draw.ellipse([center_x - radius, center_y - radius,
                         center_x + radius, center_y + radius],
                        fill=color, outline=None)
        
        elif shape_type == 'rectangle':
            x1 = center_x - shape_width // 2
            y1 = center_y - shape_height // 2
            x2 = center_x + shape_width // 2
            y2 = center_y + shape_height // 2
            # Random rotation angle for some rectangles
            if random.random() < 0.3:  # 30% chance of rotation
                angle = random.uniform(-math.pi/6, math.pi/6)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                w2, h2 = shape_width // 2, shape_height // 2
                points = [
                    (center_x + int(-w2*cos_a - h2*sin_a), center_y + int(-w2*sin_a + h2*cos_a)),
                    (center_x + int(w2*cos_a - h2*sin_a), center_y + int(w2*sin_a + h2*cos_a)),
                    (center_x + int(w2*cos_a + h2*sin_a), center_y + int(w2*sin_a - h2*cos_a)),
                    (center_x + int(-w2*cos_a + h2*sin_a), center_y + int(-w2*sin_a - h2*cos_a))
                ]
                draw.polygon(points, fill=color, outline=None)
            else:
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
        
        elif shape_type == 'ellipse':
            x1 = center_x - shape_width // 2
            y1 = center_y - shape_height // 2
            x2 = center_x + shape_width // 2
            y2 = center_y + shape_height // 2
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
        
        elif shape_type == 'rounded_rect':
            x1 = center_x - shape_width // 2
            y1 = center_y - shape_height // 2
            x2 = center_x + shape_width // 2
            y2 = center_y + shape_height // 2
            corner_radius = min(shape_width, shape_height) // 4
            # Draw rounded rectangle
            draw.rectangle([x1 + corner_radius, y1, x2 - corner_radius, y2], fill=color)
            draw.rectangle([x1, y1 + corner_radius, x2, y2 - corner_radius], fill=color)
            draw.ellipse([x1, y1, x1 + corner_radius * 2, y1 + corner_radius * 2], fill=color)
            draw.ellipse([x2 - corner_radius * 2, y1, x2, y1 + corner_radius * 2], fill=color)
            draw.ellipse([x1, y2 - corner_radius * 2, x1 + corner_radius * 2, y2], fill=color)
            draw.ellipse([x2 - corner_radius * 2, y2 - corner_radius * 2, x2, y2], fill=color)
        
        elif shape_type == 'triangle':
            # Random triangle orientation
            orientation = random.choice(['up', 'down', 'left', 'right'])
            size = min(shape_width, shape_height) // 2
            
            if orientation == 'up':
                points = [
                    (center_x, center_y - size),
                    (center_x - size, center_y + size),
                    (center_x + size, center_y + size)
                ]
            elif orientation == 'down':
                points = [
                    (center_x, center_y + size),
                    (center_x - size, center_y - size),
                    (center_x + size, center_y - size)
                ]
            elif orientation == 'left':
                points = [
                    (center_x - size, center_y),
                    (center_x + size, center_y - size),
                    (center_x + size, center_y + size)
                ]
            else:  # right
                points = [
                    (center_x + size, center_y),
                    (center_x - size, center_y - size),
                    (center_x - size, center_y + size)
                ]
            draw.polygon(points, fill=color, outline=None)
        
        elif shape_type == 'polygon':
            # Draw polygon with 5-8 sides
            num_sides = random.randint(5, 8)
            size = min(shape_width, shape_height) // 2
            points = []
            rotation = random.uniform(0, 2 * math.pi)
            for i in range(num_sides):
                angle = (2 * math.pi * i) / num_sides + rotation
                px = center_x + int(size * math.cos(angle))
                py = center_y + int(size * math.sin(angle))
                points.append((px, py))
            draw.polygon(points, fill=color, outline=None)
        
        elif shape_type == 'star':
            # Draw a star (5-pointed or 6-pointed)
            star_points = random.choice([5, 6])
            outer_radius = min(shape_width, shape_height) // 2
            inner_radius = int(outer_radius * random.uniform(0.35, 0.5))
            points = []
            rotation = random.uniform(0, 2 * math.pi)
            for i in range(star_points * 2):
                angle = (math.pi * i) / star_points + rotation - math.pi / 2
                radius = outer_radius if i % 2 == 0 else inner_radius
                px = center_x + int(radius * math.cos(angle))
                py = center_y + int(radius * math.sin(angle))
                points.append((px, py))
            draw.polygon(points, fill=color, outline=None)
        
        elif shape_type == 'diamond':
            # Draw diamond shape
            size = min(shape_width, shape_height) // 2
            points = [
                (center_x, center_y - size),  # Top
                (center_x + size, center_y),  # Right
                (center_x, center_y + size),  # Bottom
                (center_x - size, center_y)   # Left
            ]
            draw.polygon(points, fill=color, outline=None)
        
        elif shape_type == 'hexagon':
            # Draw hexagon
            size = min(shape_width, shape_height) // 2
            rotation = random.uniform(0, math.pi / 3)
            points = []
            for i in range(6):
                angle = (math.pi * i) / 3 + rotation
                px = center_x + int(size * math.cos(angle))
                py = center_y + int(size * math.sin(angle))
                points.append((px, py))
            draw.polygon(points, fill=color, outline=None)
    
    return img


def generate_jigsaw_puzzle(config: dict) -> dict:
    """Generate a random jigsaw puzzle by splitting an image into pieces."""
    if not PIL_AVAILABLE:
        raise ValueError('PIL/Pillow is required for jigsaw puzzle generation')
    
    # Get configuration
    source_images_dir = config.get("source_images_dir", "captcha_data/Jigsaw_Puzzle/sources")
    grid_rows = config.get("grid_rows", random.choice([2, 3]))
    grid_cols = config.get("grid_cols", random.choice([2, 3]))
    piece_size = config.get("piece_size", 150)
    generate_image = config.get("generate_image", True)  # Default to generating images
    
    # Ensure we have at least 2x2 and at most 3x3 for easy human solving
    grid_rows = max(2, min(grid_rows, 3))
    grid_cols = max(2, min(grid_cols, 3))
    
    # Calculate target dimensions
    target_width = grid_cols * piece_size
    target_height = grid_rows * piece_size
    
    # Try to load from source images, or generate one
    img = None
    if not generate_image:
        # Try to find source images
        source_path = os.path.join(source_images_dir)
        if not os.path.exists(source_path):
            source_path = "captcha_data/Jigsaw_Puzzle"
        
        source_images = []
        if os.path.exists(source_path):
            for filename in os.listdir(source_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    if 'piece' not in filename.lower() and 'jigsaw' not in filename.lower():
                        source_images.append(os.path.join(source_path, filename))
        
        if source_images:
            source_image_path = random.choice(source_images)
            img = Image.open(source_image_path)
            img = img.convert('RGB')
            img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    # Generate image if no source image was found or if generation is enabled
    if img is None:
        img = generate_jigsaw_image(target_width, target_height)
    
    # Generate unique puzzle ID
    puzzle_id = f"jigsaw_{int(time.time()*1000)}_{random.randint(1000,9999)}"
    
    # Split image into pieces
    pieces_data = []
    correct_positions = []
    
    for row in range(grid_rows):
        for col in range(grid_cols):
            # Calculate crop box
            left = col * piece_size
            top = row * piece_size
            right = left + piece_size
            bottom = top + piece_size
            
            # Crop piece
            piece = img.crop((left, top, right, bottom))
            
            # Convert to base64 for embedding
            buffered = io.BytesIO()
            piece.save(buffered, format="PNG")
            piece_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            piece_data_url = f"data:image/png;base64,{piece_base64}"
            
            pieces_data.append(piece_data_url)
            
            # Store correct position
            correct_positions.append({
                "piece_index": len(pieces_data) - 1,
                "grid_row": row,
                "grid_col": col
            })
    
    # Shuffle pieces for puzzle (pieces are shown in random order)
    shuffled_indices = list(range(len(pieces_data)))
    random.shuffle(shuffled_indices)
    
    # Store puzzle state
    active_jigsaw_puzzles[puzzle_id] = {
        "pieces_data": pieces_data,
        "correct_positions": correct_positions,
        "grid_size": [grid_rows, grid_cols],
        "piece_size": piece_size,
        "shuffled_indices": shuffled_indices,
        "created_at": time.time()
    }
    
    # Create reference image (full image) as base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    reference_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    reference_data_url = f"data:image/png;base64,{reference_base64}"
    
    return {
        "puzzle_id": puzzle_id,
        "pieces": pieces_data,  # Return base64 data URLs
        "grid_size": [grid_rows, grid_cols],
        "piece_size": piece_size,
        "correct_positions": correct_positions,
        "reference_image": reference_data_url,
        "prompt": "Drag the puzzle pieces to complete the jigsaw puzzle"
    }


def generate_red_dot(config: dict) -> dict:
    """Create a red dot reaction puzzle definition."""
    area_size = config.get("area_size", [420, 320])
    if not isinstance(area_size, (list, tuple)) or len(area_size) < 2:
        area_size = [420, 320]
    area_width = max(100, int(area_size[0]))
    area_height = max(100, int(area_size[1]))

    dot_diameter = max(12, int(config.get("dot_diameter", 40)))
    margin = max(0, int(config.get("margin", 24)))

    max_x = area_width - margin - dot_diameter
    max_y = area_height - margin - dot_diameter
    min_x = margin
    min_y = margin

    if max_x <= min_x:
        min_x = margin
        max_x = area_width - dot_diameter
    if max_y <= min_y:
        min_y = margin
        max_y = area_height - dot_diameter

    timeout_ms = int(config.get("timeout_ms", 2000))
    prompt = config.get("prompt", "Click the red dot before it disappears.")
    required_hits = max(1, int(config.get("required_hits", 1)))
    puzzle_id = f"red_dot_{uuid.uuid4().hex}"

    dots: list[dict[str, float]] = []
    for _ in range(required_hits):
        x_pos = random.uniform(min_x, max_x)
        y_pos = random.uniform(min_y, max_y)
        dots.append({"x": x_pos, "y": y_pos})

    first_dot = dots[0]

    return {
        "puzzle_id": puzzle_id,
        "prompt": prompt,
        "area": {
            "width": area_width,
            "height": area_height
        },
        "dot": {
            "x": first_dot["x"],
            "y": first_dot["y"],
            "diameter": dot_diameter
        },
        "dot_sequence": dots,
        "timeout_ms": timeout_ms,
        "input_type": "red_dot_click",
        "required_hits": required_hits,
        "debug_info": (
            f"{required_hits} hit(s). First dot at ({first_dot['x']:.1f}, {first_dot['y']:.1f}) "
            f"within {area_width}x{area_height} area. Timeout {timeout_ms}ms."
        )
    }


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
    if puzzle_type == "Color_Cipher":
        config = ground_truth.get("config", {})
        cipher = generate_color_cipher(config)
        response_data = {
            'puzzle_type': puzzle_type,
            'image_path': None,
            'puzzle_id': cipher["puzzle_id"],
            'prompt': cipher["prompt"],
            'input_type': 'color_cipher',
            'debug_info': f"Type: {puzzle_type}, Input: color_cipher, Expression: {cipher['debug_expression']}",
            'mapping': cipher["mapping"],
            'question': cipher["question"],
            'reveal_duration': cipher["reveal_duration"],
            'input_mode': cipher["input_mode"],
            'cipher_state': cipher["cipher_state"]
        }
        return jsonify(response_data)
    if puzzle_type == "Red_Dot":
        config = ground_truth.get("config", {})
        puzzle = generate_red_dot(config)
        puzzle_id = puzzle["puzzle_id"]
        dot_diameter = puzzle["dot"]["diameter"]
        radius = dot_diameter / 2
        dots = puzzle.get("dot_sequence", [])
        first_dot_center_x = puzzle["dot"]["x"] + radius
        first_dot_center_y = puzzle["dot"]["y"] + radius

        active_red_dot_puzzles[puzzle_id] = {
            "center_x": first_dot_center_x,
            "center_y": first_dot_center_y,
            "radius": radius,
            "timeout_ms": puzzle["timeout_ms"],
            "current_start_time": time.time(),
            "area_width": puzzle["area"]["width"],
            "area_height": puzzle["area"]["height"],
            "required_hits": puzzle.get("required_hits", 1),
            "current_index": 0,
            "dots": dots,
            "dot_diameter": dot_diameter
        }
        response_data = {
            'puzzle_type': puzzle_type,
            'image_path': None,
            'media_path': None,
            'media_type': None,
            'puzzle_id': puzzle_id,
            'prompt': puzzle["prompt"],
            'input_type': puzzle["input_type"],
            'area': puzzle["area"],
            'dot': puzzle["dot"],
            'timeout_ms': puzzle["timeout_ms"],
            'required_hits': puzzle.get("required_hits", 1),
            'hits_completed': 0,
            'debug_info': f"Type: {puzzle_type}, Input: red_dot_click, Puzzle: {puzzle_id}, {puzzle['debug_info']}"
        }
        return jsonify(response_data)

    if puzzle_type == "Spooky_Size":
        if not ground_truth:
            return jsonify({'error': f'No puzzles found for type: {puzzle_type}'}), 404

        # Select random puzzle
        puzzle_files = list(ground_truth.keys())
        selected_puzzle = random.choice(puzzle_files)
        puzzle_data = ground_truth[selected_puzzle]

        # Generate unique puzzle ID
        puzzle_id = f"spooky_size_{int(time.time()*1000)}_{random.randint(1000,9999)}"

        # Store the target position for validation
        # Convert from top-left + diameter to center (like Red Dot does)
        target_pos = puzzle_data["target_position"]
        diameter = target_pos.get("diameter", target_pos.get("radius", 0) * 2)  # Support both old and new format
        radius = diameter / 2
        center_x = target_pos["x"] + radius
        center_y = target_pos["y"] + radius

        active_spooky_size_puzzles[puzzle_id] = {
            "target_x": center_x,
            "target_y": center_y,
            "radius": radius,
            "start_time": time.time()
        }

        response_data = {
            'puzzle_type': puzzle_type,
            'image_path': None,
            'media_path': f'/captcha_data/{puzzle_type}/{selected_puzzle}.gif',
            'media_type': 'gif',
            'puzzle_id': puzzle_id,
            'prompt': puzzle_data["prompt"],
            'input_type': 'spooky_size_click',
            'canvas_width': 600,
            'canvas_height': 400,
            'debug_info': f"Type: {puzzle_type}, Puzzle: {selected_puzzle}"
        }
        return jsonify(response_data)

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
    
    media_type = ground_truth[selected_puzzle].get("media_type")
    media_path = ground_truth[selected_puzzle].get("media_path")

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
    elif puzzle_type == "Spooky_Circle":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "How many circles can you see in this animation?"
        )
        if not media_type:
            media_type = "gif"
    elif puzzle_type == "Spooky_Circle_Grid":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "How many cells contain circles in this grid?"
        )
        if not media_type:
            media_type = "gif"
    elif puzzle_type == "Spooky_Shape_Grid":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "Click all matching shapes rotating in the specified direction"
        )
        if not media_type:
            media_type = "gif"
    elif puzzle_type == "Spooky_Text":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "What text do you see in this animation?"
        )
        if not media_type:
            media_type = "gif"
    elif puzzle_type == "Storyboard_Logic":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "Reorder the images to show the story in the correct causal sequence"
        )
    elif puzzle_type == "Jigsaw_Puzzle":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "Drag the puzzle pieces to complete the jigsaw puzzle"
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
    elif puzzle_type == "Squiggle":
        input_type = "squiggle_select"
    elif puzzle_type == "Spooky_Circle":
        input_type = "number"
    elif puzzle_type == "Spooky_Circle_Grid":
        input_type = "circle_grid_select"
    elif puzzle_type == "Spooky_Circle_Grid_Direction":
        input_type = "circle_grid_direction_select"
    elif puzzle_type == "Spooky_Shape_Grid":
        input_type = "shape_grid_select"
    elif puzzle_type == "Spooky_Text":
        input_type = "text"
    elif puzzle_type == "Color_Cipher":
        input_type = "color_cipher"
    elif puzzle_type == "Color_Counting":
        input_type = "color_counting_select"
    elif puzzle_type == "Trajectory_Recovery":
        input_type = "trajectory_recovery_select"
    elif puzzle_type == "Storyboard_Logic":
        input_type = "storyboard_logic"
    elif puzzle_type == "Jigsaw_Puzzle":
        input_type = "jigsaw_puzzle"
    
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
    elif puzzle_type == "Shadow_Plausible":
        option_images = ground_truth[selected_puzzle].get("options", [])
        if not option_images:
            return jsonify({'error': f'Invalid shadow data: {selected_puzzle}'}), 500

        additional_data = {
            "option_images": [f'/captcha_data/{puzzle_type}/{img}' for img in option_images],
            "grid_size": ground_truth[selected_puzzle].get("grid_size", []),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Squiggle":
        reference_image = ground_truth[selected_puzzle].get("reference")
        option_images = ground_truth[selected_puzzle].get("options", [])
        if not reference_image or not option_images:
            return jsonify({'error': f'Invalid squiggle data: {selected_puzzle}'}), 500

        additional_data = {
            "reference_image": f'/captcha_data/{puzzle_type}/{reference_image}',
            "option_images": [f'/captcha_data/{puzzle_type}/{img}' for img in option_images],
            "answer": ground_truth[selected_puzzle].get("answer"),
            "reveal_duration": ground_truth[selected_puzzle].get("reveal_duration", 3),
            "grid_size": ground_truth[selected_puzzle].get("grid_size")
        }
    elif puzzle_type == "Spooky_Circle_Grid":
        option_gifs = ground_truth[selected_puzzle].get("options", [])
        if not option_gifs:
            return jsonify({'error': f'Invalid Spooky_Circle_Grid data: {selected_puzzle}'}), 500

        additional_data = {
            "option_images": [f'/captcha_data/{puzzle_type}/{gif}' for gif in option_gifs],
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [3, 3]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Spooky_Circle_Grid_Direction":
        option_gifs = ground_truth[selected_puzzle].get("options", [])
        if not option_gifs:
            return jsonify({'error': f'Invalid Spooky_Circle_Grid_Direction data: {selected_puzzle}'}), 500

        additional_data = {
            "option_images": [f'/captcha_data/{puzzle_type}/{gif}' for gif in option_gifs],
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [3, 3]),
            "answer": ground_truth[selected_puzzle].get("answer", []),
            "target_direction": ground_truth[selected_puzzle].get("target_direction")
        }
    elif puzzle_type == "Spooky_Shape_Grid":
        option_gifs = ground_truth[selected_puzzle].get("options", [])
        if not option_gifs:
            return jsonify({'error': f'Invalid Spooky_Shape_Grid data: {selected_puzzle}'}), 500

        additional_data = {
            "option_images": [f'/captcha_data/{puzzle_type}/{gif}' for gif in option_gifs],
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [3, 3]),
            "answer": ground_truth[selected_puzzle].get("answer", []),
            "target_shape": ground_truth[selected_puzzle].get("target_shape"),
            "target_direction": ground_truth[selected_puzzle].get("target_direction")
        }
    elif puzzle_type == "Color_Counting":
        option_images = ground_truth[selected_puzzle].get("options", [])
        if not option_images:
            return jsonify({'error': f'Invalid Color_Counting data: {selected_puzzle}'}), 500

        additional_data = {
            "option_images": [f'/captcha_data/{puzzle_type}/{img}' for img in option_images],
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [3, 3]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Trajectory_Recovery":
        option_images = ground_truth[selected_puzzle].get("options", [])
        movement_gif = ground_truth[selected_puzzle].get("movement_gif")
        if not option_images or not movement_gif:
            return jsonify({'error': f'Invalid Trajectory_Recovery data: {selected_puzzle}'}), 500

        additional_data = {
            "movement_gif": f'/captcha_data/{puzzle_type}/{movement_gif}',
            "option_images": [f'/captcha_data/{puzzle_type}/{img}' for img in option_images],
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [4, 4]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Storyboard_Logic":
        images = ground_truth[selected_puzzle].get("images", [])
        if not images:
            return jsonify({'error': f'Invalid Storyboard_Logic data: {selected_puzzle}'}), 500

        additional_data = {
            "images": [f'/captcha_data/{puzzle_type}/{img}' for img in images],
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Jigsaw_Puzzle":
        # Check if we should generate a random puzzle or use ground truth
        use_generation = ground_truth.get("config", {}).get("generate_random", True)
        
        if use_generation and PIL_AVAILABLE:
            # Generate random puzzle
            try:
                config = ground_truth.get("config", {})
                puzzle = generate_jigsaw_puzzle(config)
                puzzle_id = puzzle["puzzle_id"]
                
                additional_data = {
                    "pieces": puzzle["pieces"],  # Base64 data URLs
                    "grid_size": puzzle["grid_size"],
                    "piece_size": puzzle["piece_size"],
                    "correct_positions": puzzle["correct_positions"],
                    "reference_image": puzzle["reference_image"]
                }
                
                response_data = {
                    'puzzle_type': puzzle_type,
                    'image_path': None,
                    'media_path': None,
                    'media_type': None,
                    'puzzle_id': puzzle_id,
                    'prompt': puzzle["prompt"],
                    'input_type': 'jigsaw_puzzle',
                    'debug_info': f"Type: {puzzle_type}, Generated puzzle: {puzzle_id}"
                }
                response_data.update(additional_data)
                return jsonify(response_data)
            except Exception as e:
                # Fall back to ground truth if generation fails
                print(f"Jigsaw generation failed: {e}")
        
        # Use ground truth puzzles
        pieces = ground_truth[selected_puzzle].get("pieces", [])
        grid_size = ground_truth[selected_puzzle].get("grid_size", [2, 2])
        piece_size = ground_truth[selected_puzzle].get("piece_size", 150)
        correct_positions = ground_truth[selected_puzzle].get("correct_positions", [])
        reference_image = ground_truth[selected_puzzle].get("image")
        
        if not pieces or not correct_positions:
            return jsonify({'error': f'Invalid Jigsaw_Puzzle data: {selected_puzzle}'}), 500

        additional_data = {
            "pieces": [f'/captcha_data/{puzzle_type}/{piece}' for piece in pieces],
            "grid_size": grid_size,
            "piece_size": piece_size,
            "correct_positions": correct_positions,
            "reference_image": f'/captcha_data/{puzzle_type}/{reference_image}' if reference_image else None
        }
    else:
        prompt = ground_truth[selected_puzzle].get("prompt", "Solve the CAPTCHA puzzle")

    image_path = None
    if puzzle_type not in ("Rotation_Match", "Shadow_Plausible", "Mirror",  "Squiggle", "Spooky_Circle_Grid", "Spooky_Circle_Grid_Direction", "Spooky_Shape_Grid", "Color_Cipher", "Color_Counting", "Trajectory_Recovery", "Storyboard_Logic", "Jigsaw_Puzzle"):
        image_path = f'/captcha_data/{puzzle_type}/{selected_puzzle}'
        if not media_type:
            media_type = "image"
        if not media_path:
            media_path = image_path

    if media_path is None and image_path:
        media_path = image_path

    response_data = {
        'puzzle_type': puzzle_type,
        'image_path': image_path,
        'media_path': media_path,
        'media_type': media_type,
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
    
    if puzzle_type == 'Color_Cipher':
        return jsonify({'error': 'Ground truth is generated dynamically for Color_Cipher puzzles'}), 400
    
    if puzzle_id not in ground_truth:
        return jsonify({'error': 'Invalid puzzle ID'}), 400
    
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
    
    if puzzle_type not in ('Color_Cipher', 'Red_Dot', 'Spooky_Size', 'Jigsaw_Puzzle') and puzzle_id not in ground_truth:
        return jsonify({'error': 'Invalid puzzle ID'}), 400
    
    # Get correct answer based on puzzle type
    is_correct = False
    correct_answer_info = None
    status = None
            
    
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

    elif puzzle_type == 'Squiggle':
        try:
            correct_index = int(ground_truth[puzzle_id].get('answer'))
            user_index = int(user_answer)
            is_correct = user_index == correct_index
            correct_answer_info = correct_index
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Squiggle'}), 400
    elif puzzle_type == 'Spooky_Circle':
        try:
            correct_value = int(ground_truth[puzzle_id].get('answer'))
            user_value = int(user_answer)
            is_correct = user_value == correct_value
            correct_answer_info = correct_value
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Spooky_Circle'}), 400
    elif puzzle_type == 'Spooky_Circle_Grid':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Spooky_Circle_Grid'}), 400
    elif puzzle_type == 'Spooky_Circle_Grid_Direction':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Spooky_Circle_Grid_Direction'}), 400
    elif puzzle_type == 'Spooky_Shape_Grid':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Spooky_Shape_Grid'}), 400
    elif puzzle_type == 'Color_Counting':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Color_Counting'}), 400
    elif puzzle_type == 'Trajectory_Recovery':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Trajectory_Recovery'}), 400
    elif puzzle_type == 'Storyboard_Logic':
        try:
            # For Storyboard_Logic, order matters - check exact sequence match
            correct_order = ground_truth[puzzle_id].get('answer', [])
            user_order = [int(idx) for idx in user_answer]
            is_correct = user_order == correct_order
            correct_answer_info = correct_order
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Storyboard_Logic'}), 400
    elif puzzle_type == 'Jigsaw_Puzzle':
        try:
            # Check if this is a generated puzzle (stored in active_jigsaw_puzzles)
            puzzle_state = active_jigsaw_puzzles.get(puzzle_id)
            
            if puzzle_state:
                # Generated puzzle - use stored correct positions
                correct_positions = puzzle_state.get('correct_positions', [])
            else:
                # Ground truth puzzle
                if puzzle_id not in ground_truth:
                    return jsonify({'error': 'Invalid puzzle ID'}), 400
                correct_positions = ground_truth[puzzle_id].get('correct_positions', [])
            
            user_placements = user_answer if isinstance(user_answer, list) else []
            
            # Convert user placements to a dictionary for easy lookup
            user_positions_dict = {}
            for placement in user_placements:
                if isinstance(placement, dict):
                    piece_idx = placement.get('piece_index')
                    grid_row = placement.get('grid_row')
                    grid_col = placement.get('grid_col')
                    # Ensure all values are integers
                    try:
                        piece_idx = int(piece_idx) if piece_idx is not None else None
                        grid_row = int(grid_row) if grid_row is not None else None
                        grid_col = int(grid_col) if grid_col is not None else None
                        if piece_idx is not None and grid_row is not None and grid_col is not None:
                            user_positions_dict[piece_idx] = {'grid_row': grid_row, 'grid_col': grid_col}
                    except (ValueError, TypeError):
                        continue
            
            # Check if all pieces are correctly placed
            is_correct = True
            if len(user_positions_dict) != len(correct_positions):
                is_correct = False
            else:
                for correct_pos in correct_positions:
                    piece_idx = correct_pos.get('piece_index')
                    correct_row = correct_pos.get('grid_row')
                    correct_col = correct_pos.get('grid_col')
                    
                    # Ensure correct_positions values are integers
                    try:
                        piece_idx = int(piece_idx) if piece_idx is not None else None
                        correct_row = int(correct_row) if correct_row is not None else None
                        correct_col = int(correct_col) if correct_col is not None else None
                    except (ValueError, TypeError):
                        is_correct = False
                        break
                    
                    if piece_idx not in user_positions_dict:
                        is_correct = False
                        break
                    
                    user_pos = user_positions_dict[piece_idx]
                    if user_pos['grid_row'] != correct_row or user_pos['grid_col'] != correct_col:
                        is_correct = False
                        break
            
            correct_answer_info = correct_positions
            
            # Debug logging (can be removed in production)
            if not is_correct:
                print(f"Jigsaw validation failed. Puzzle ID: {puzzle_id}")
                print(f"Correct positions: {correct_positions}")
                print(f"User placements: {user_placements}")
                print(f"User positions dict: {user_positions_dict}")
            
            # Clean up generated puzzle state after validation
            if puzzle_state:
                active_jigsaw_puzzles.pop(puzzle_id, None)
        except (ValueError, TypeError, KeyError) as e:
            import traceback
            print(f"Jigsaw validation error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Invalid answer format for Jigsaw_Puzzle: {str(e)}'}), 400
    elif puzzle_type == 'Spooky_Size':
        state = active_spooky_size_puzzles.get(puzzle_id)
        if state is None:
            return jsonify({'error': 'Puzzle state expired'}), 400
        if not isinstance(user_answer, dict):
            return jsonify({'error': 'Invalid answer format for Spooky_Size'}), 400

        # Get click position
        position = user_answer.get('position') or {}
        try:
            click_x = float(position.get('x'))
            click_y = float(position.get('y'))
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid click position for Spooky_Size'}), 400

        # Check if click is within the target radius (with tolerance like Red Dot)
        target_x = state['target_x']
        target_y = state['target_y']
        radius = state['radius']

        # Add tolerance like Red Dot does (15% of radius + minimum 4px)
        dx = click_x - target_x
        dy = click_y - target_y
        distance_sq = dx * dx + dy * dy
        tolerance = max(4.0, radius * 0.15)
        is_correct = distance_sq <= (radius + tolerance) ** 2

        distance = math.sqrt(distance_sq)

        # Clean up
        active_spooky_size_puzzles.pop(puzzle_id, None)

        correct_answer_info = {
            "target_x": target_x,
            "target_y": target_y,
            "radius": radius,
            "click_x": click_x,
            "click_y": click_y,
            "distance": float(distance)
        }
    elif puzzle_type == 'Red_Dot':
        state = active_red_dot_puzzles.get(puzzle_id)
        if state is None:
            return jsonify({'error': 'Puzzle state expired'}), 400
        if not isinstance(user_answer, dict):
            return jsonify({'error': 'Invalid answer format for Red_Dot'}), 400

        try:
            hit_index = int(user_answer.get('hit_index', state.get('current_index', 0)))
        except (TypeError, ValueError):
            return jsonify({'error': 'Invalid hit index for Red_Dot'}), 400

        expected_index = state.get('current_index', 0)
        if hit_index != expected_index:
            active_red_dot_puzzles.pop(puzzle_id, None)
            return jsonify({
                'correct': False,
                'status': 'failed',
                'message': 'Unexpected click order. Puzzle reset.'
            })

        clicked = bool(user_answer.get('clicked'))
        position = user_answer.get('position') or {}
        try:
            attempt_x = float(position.get('x'))
            attempt_y = float(position.get('y'))
        except (TypeError, ValueError):
            attempt_x, attempt_y = None, None

        timeout_ms = state.get('timeout_ms', 2000)
        start_time = state.get('current_start_time') or state.get('start_time') or time.time()
        elapsed_ms = (time.time() - start_time) * 1000
        within_time = elapsed_ms <= timeout_ms + 200

        within_radius = False
        if clicked and attempt_x is not None and attempt_y is not None:
            dx = attempt_x - state['center_x']
            dy = attempt_y - state['center_y']
            distance_sq = dx * dx + dy * dy
            tolerance = max(4.0, state['radius'] * 0.15)
            within_radius = distance_sq <= (state['radius'] + tolerance) ** 2

        if not clicked or attempt_x is None or attempt_y is None or not within_time or not within_radius:
            active_red_dot_puzzles.pop(puzzle_id, None)
            return jsonify({
                'correct': False,
                'status': 'failed',
                'message': 'Missed the red dot in time.'
            })

        hits_completed = expected_index + 1
        state['current_index'] = hits_completed

        required_hits = state.get('required_hits', hits_completed)

        if hits_completed >= required_hits:
            active_red_dot_puzzles.pop(puzzle_id, None)
            is_correct = True
            status = 'completed'
            correct_answer_info = {
                'hits_completed': hits_completed,
                'required_hits': required_hits,
                'last_hit_elapsed_ms': elapsed_ms
            }
        else:
            next_dot = state['dots'][hits_completed]
            state['center_x'] = next_dot['x'] + state['radius']
            state['center_y'] = next_dot['y'] + state['radius']
            state['current_start_time'] = time.time()
            response_payload = {
                'correct': False,
                'status': 'continue',
                'hits_completed': hits_completed,
                'required_hits': state['required_hits'],
                'next_dot': {
                    'x': next_dot['x'],
                    'y': next_dot['y'],
                    'diameter': state['dot_diameter']
                },
                'timeout_ms': state['timeout_ms']
            }
            return jsonify(response_payload)
    elif puzzle_type == 'Color_Cipher':
        cipher_state = data.get('cipher_state') or {}
        mapping = cipher_state.get('mapping') or []
        expression = cipher_state.get('expression')
        if not mapping or not expression:
            return jsonify({'error': 'Missing color cipher state'}), 400
        try:
            correct_value = evaluate_color_cipher(expression, mapping)
            user_value = float(user_answer)
            is_correct = abs(user_value - float(correct_value)) < 1e-6
            correct_answer_info = correct_value
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Color_Cipher'}), 400
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
    if puzzle_type == 'Color_Cipher':
        correct_value = correct_answer_info
        if isinstance(correct_value, float) and abs(correct_value - round(correct_value)) < 1e-6:
            correct_value = int(round(correct_value))
        correct_payload = correct_value
    elif puzzle_type == 'Red_Dot':
        if isinstance(correct_answer_info, dict):
            hits_done = correct_answer_info.get('hits_completed')
            hits_required = correct_answer_info.get('required_hits', hits_done)
            correct_payload = f'Completed {hits_done}/{hits_required} hits.'
        else:
            correct_payload = 'Click the red dot before it disappears.'
    elif puzzle_type == 'Spooky_Size':
        # Spooky_Size uses dynamic puzzle IDs, return validation details
        if isinstance(correct_answer_info, dict):
            correct_payload = f"Target at ({correct_answer_info['target_x']:.0f}, {correct_answer_info['target_y']:.0f}), radius {correct_answer_info['radius']:.0f}px"
        else:
            correct_payload = "Click validation details"
    elif puzzle_type == 'Storyboard_Logic':
        # Format the correct order as a readable sequence
        if isinstance(correct_answer_info, list):
            order_names = [f"Image {i+1}" for i in correct_answer_info]
            correct_payload = f"Correct order: {' → '.join(order_names)}"
        else:
            correct_payload = ground_truth[puzzle_id].get(answer_key)
    elif puzzle_type == 'Jigsaw_Puzzle':
        # Format the correct positions as a readable string
        if isinstance(correct_answer_info, list):
            position_strs = []
            for pos in correct_answer_info:
                piece_idx = pos.get('piece_index', '?')
                row = pos.get('grid_row', '?')
                col = pos.get('grid_col', '?')
                position_strs.append(f"Piece {piece_idx} at ({row}, {col})")
            correct_payload = f"Correct positions: {'; '.join(position_strs)}"
        else:
            correct_payload = "Puzzle completion details"
    else:
        correct_payload = ground_truth[puzzle_id].get(answer_key)

    response_body = {
        'correct': is_correct,
        'user_answer': user_answer,
        'correct_answer': correct_payload
    }
    if status is not None:
        response_body['status'] = status
    if puzzle_type == 'Red_Dot' and isinstance(correct_answer_info, dict):
        response_body['details'] = correct_answer_info
    if puzzle_type == 'Spooky_Size' and isinstance(correct_answer_info, dict):
        response_body['details'] = correct_answer_info
    if puzzle_type == 'Jigsaw_Puzzle' and not is_correct:
        # Include debug details for incorrect jigsaw puzzles
        response_body['details'] = {
            'user_placements': user_answer if puzzle_type == 'Jigsaw_Puzzle' else None,
            'correct_positions': correct_answer_info if puzzle_type == 'Jigsaw_Puzzle' else None
        }

    return jsonify(response_body)

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
        app.run(debug=True, host='127.0.0.1', port=7860)
    else:
        # For production on Hugging Face Spaces
        app.run(host='127.0.0.1', port=7860) 
