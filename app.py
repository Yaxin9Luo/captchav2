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
    # 'Hole_Counting',
    # 'Rotation_Match',
    # 'Rhythm',
    #'Backmost_Layer',
    #'Shadow_Direction',
    'Global_Phase_Drift',
    # 'Trajectory_Recovery',
    # 'Spooky_Size',
    # 'Spooky_Circle',
    # 'Spooky_Circle_Grid',
    # 'Spooky_Shape_Grid',
    # 'Spooky_Text',
    # 'Red_Dot',
    # 'Storyboard_Logic',
    # 'Static_Jigsaw',
    # 'Transform_Pipeline',
    # 'Set_Game',
    # 'Dynamic_Jigsaw',
    # 'Spooky_Jigsaw',
]
sequential_index = 0

active_red_dot_puzzles: dict[str, dict] = {}
active_spooky_size_puzzles: dict[str, dict] = {}
active_jigsaw_puzzles: dict[str, dict] = {}
active_transform_pipeline_puzzles: dict[str, dict] = {}

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

CURRENT_AGENT_METADATA: dict[str, str] = {}

# Benchmark results file configuration
# Can be set via environment variable BENCHMARK_RESULTS_FILE
# Supports placeholders: {model}, {provider}, {framework}, {timestamp}, {date}
# Examples:
#   - "benchmark_results.json" (default)
#   - "results_{model}_{timestamp}.json"
#   - "benchmark_{date}.json"
BENCHMARK_RESULTS_FILE_PATTERN = os.environ.get(
    'BENCHMARK_RESULTS_FILE', 
    'benchmark_results.json'
)

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
    source_images_dir = config.get("source_images_dir", "captcha_data/Static_Jigsaw/sources")
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
            source_path = "captcha_data/Static_Jigsaw"
        
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


def generate_transform_pipeline(config: dict) -> dict:
    """Generate a Transform Pipeline puzzle using SVG templates with randomized variations."""  
    import json as json_module
    
    # Color definitions
    COL = {
        "red": "#E53935",
        "blue": "#1E88E5",
        "green": "#43A047",
        "purple": "#8E24AA",
        "orange": "#FB8C00",
        "cyan": "#26C6DA",
        "pink": "#EC407A",
        "black": "#000000",
        "white": "#FFFFFF",
        "yellow": "#FDD835",
        "brown": "#8D6E63",
        "teal": "#00897B",
        "gray": "#757575"
    }
    
    # Available colors for randomization (excluding black/white for some templates)
    COLOR_POOL = ["red", "blue", "green", "purple", "orange", "cyan", "pink", "yellow", "brown", "teal"]
    
    # Determine if we should use random pipelines or fixed ones
    use_random_pipelines = config.get("use_random_pipelines", True)
    
    # Template types and their initial states
    # Templates with their color capacity (how many colors they can display)
    TEMPLATE_COLOR_CAPACITY = {
        "lshape": 2,  # color, bg
        "arrow_circle": 3,  # arrow, ring, bg
        "pacman": 3,  # body, eye, bg
        "circle": 2,  # fill, bg
        "square": 2,  # fill, bg
        "triangle": 2,  # fill, bg
        "star": 2,  # fill, bg
        "diamond": 2  # fill, bg
    }
    
    available_templates = config.get("templates", list(TEMPLATE_COLOR_CAPACITY.keys()))
    
    # Define generate_random_pipeline function before it's used
    def generate_random_pipeline(color_pool):
        """Generate a random transformation pipeline."""
        pipeline = []
        num_steps = random.randint(2, 4)  # 2-4 steps
        
        # Available operations
        operations = []
        
        # Always include some rotation
        if random.random() < 0.8:  # 80% chance
            rotations = [90, -90, 180, 270, -270]
            operations.append(("rotate", random.choice(rotations)))
        
        # Mirror operations
        if random.random() < 0.7:  # 70% chance
            operations.append(("mirror_h", None))
        if random.random() < 0.7:  # 70% chance
            operations.append(("mirror_v", None))
        
        # Color mapping
        if random.random() < 0.8:  # 80% chance
            num_mappings = random.randint(1, 3)
            color_map = {}
            available_colors = color_pool.copy()
            for _ in range(num_mappings):
                if len(available_colors) < 2:
                    break
                src_color = random.choice(available_colors)
                available_colors.remove(src_color)
                dst_color = random.choice(available_colors)
                color_map[src_color] = dst_color
            if color_map:
                operations.append(("map_colors", color_map))
        
        # Randomly select and shuffle operations
        selected_ops = random.sample(operations, min(num_steps, len(operations))) if operations else []
        random.shuffle(selected_ops)
        
        return selected_ops if selected_ops else [("rotate", 90), ("mirror_h", None)]  # Fallback
    
    # If using random pipelines, check how many color mappings we have
    if use_random_pipelines:
        # Generate pipeline first to see how many color mappings it has
        pipeline = generate_random_pipeline(COLOR_POOL)
        
        # Count unique colors in color mappings
        color_mappings = [arg for op, arg in pipeline if op == "map_colors"]
        num_mapped_colors = 0
        num_pool_colors = 0  # Count non-black/white colors
        if color_mappings:
            source_colors = set()
            for mapping in color_mappings:
                source_colors.update(mapping.keys())
            num_mapped_colors = len(source_colors)
            # Count how many are pool colors (not black/white)
            num_pool_colors = len([c for c in source_colors if c in COLOR_POOL])
        
        # Prefer templates that can accommodate all mapped colors
        # If we have 2+ pool colors (non-black/white), we need 3-color templates
        # because 2-color templates can only show 1 pool color (fill) + 1 black/white (bg)
        if num_pool_colors >= 2:
            # Need at least 3 color properties to show 2+ pool colors
            suitable_templates = [t for t in available_templates 
                                if t in TEMPLATE_COLOR_CAPACITY 
                                and TEMPLATE_COLOR_CAPACITY[t] >= 3]
            if suitable_templates:
                template_name = random.choice(suitable_templates)
            else:
                template_name = random.choice(available_templates)
        elif num_mapped_colors > 2:
            # Need at least 3 color properties
            suitable_templates = [t for t in available_templates 
                                if t in TEMPLATE_COLOR_CAPACITY 
                                and TEMPLATE_COLOR_CAPACITY[t] >= num_mapped_colors]
            if suitable_templates:
                template_name = random.choice(suitable_templates)
            else:
                template_name = random.choice(available_templates)
        else:
            template_name = random.choice(available_templates)
    else:
        template_name = random.choice(available_templates)
    
    # Pipeline definitions (can be customized via config)
    default_pipelines = {
        "lshape": [
            ("rotate", 90),
            ("map_colors", {"red": "blue"}),
            ("mirror_h", None)
        ],
        "arrow_circle": [
            ("mirror_v", None),
            ("map_colors", {"green": "purple", "white": "black", "black": "white"}),
            ("rotate", 180)
        ],
        "pacman": [
            ("rotate", -90),
            ("map_colors", {"yellow": "cyan", "black": "white"}),
            ("mirror_h", None)
        ]
    }
    
    # Generate random pipeline if enabled (pipeline was already generated above if use_random_pipelines)
    if not use_random_pipelines:
        pipelines_config = config.get("pipelines", default_pipelines)
        # Convert JSON arrays to tuples for pipeline operations
        if isinstance(pipelines_config, dict):
            pipeline_raw = pipelines_config.get(template_name, pipelines_config.get("lshape", default_pipelines["lshape"]))
            # Convert list format from JSON to tuple format
            pipeline = []
            for item in pipeline_raw:
                if isinstance(item, list) and len(item) == 2:
                    op, arg = item
                    # Convert null to None
                    if arg is None or (isinstance(arg, str) and arg.lower() == "null"):
                        pipeline.append((op, None))
                    else:
                        pipeline.append((op, arg))
                else:
                    pipeline.append(item)
        else:
            pipeline = pipelines_config.get(template_name, default_pipelines["lshape"])
    
    
    # Helper functions for SVG generation
    def svg_header(w=256, h=256):
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
    
    def svg_bg(color="#FFFFFF", w=256, h=256):
        return f'<rect x="0" y="0" width="{w}" height="{h}" fill="{color}"/>\n'
    
    def svg_footer():
        return "</svg>\n"
    
    # L-shape template functions
    ORIENTS = ["└", "┌", "┐", "┘"]
    
    def lshape_rotate(o, deg):
        idx = ORIENTS.index(o)
        steps = (deg // 90) % 4
        return ORIENTS[(idx + steps) % 4]
    
    def lshape_mirror_h(o):
        return {"└": "┘", "┘": "└", "┌": "┐", "┐": "┌"}[o]
    
    def lshape_mirror_v(o):
        return {"└": "┌", "┌": "└", "┘": "┐", "┐": "┘"}[o]
    
    def draw_lshape(orientation="└", fill=COL["red"], stroke=None):
        t = 56
        shapes = []
        if orientation == "└":
            shapes.append(f'<rect x="60" y="44" width="{t}" height="156" fill="{fill}"/>\n')
            shapes.append(f'<rect x="60" y="{44+156-t}" width="156" height="{t}" fill="{fill}"/>\n')
        elif orientation == "┌":
            shapes.append(f'<rect x="60" y="44" width="{t}" height="156" fill="{fill}"/>\n')
            shapes.append(f'<rect x="60" y="44" width="156" height="{t}" fill="{fill}"/>\n')
        elif orientation == "┐":
            shapes.append(f'<rect x="{60+156-t}" y="44" width="{t}" height="156" fill="{fill}"/>\n')
            shapes.append(f'<rect x="60" y="44" width="156" height="{t}" fill="{fill}"/>\n')
        elif orientation == "┘":
            shapes.append(f'<rect x="{60+156-t}" y="44" width="{t}" height="156" fill="{fill}"/>\n')
            shapes.append(f'<rect x="60" y="{44+156-t}" width="156" height="{t}" fill="{fill}"/>\n')
        return "".join(shapes)
    
    def lshape_make_scene(orientation="└", color="red", bg="white"):
        svg = svg_header()
        svg += svg_bg(COL[bg])
        svg += draw_lshape(orientation, COL[color])
        svg += svg_footer()
        return svg
    
    # Arrow in circle template functions
    def arrow_polygon_points(direction="right"):
        cx, cy = 128, 128
        shaft_len, shaft_w, head_len = 90, 24, 36
        if direction == "right":
            x0, x1, x2 = cx-60, cx+30, cx+30+head_len
            y0, y1 = cy - shaft_w/2, cy + shaft_w/2
            pts = [(x0,y0),(x1,y0),(x1,cy-18),(x2,cy),(x1,cy+18),(x1,y1),(x0,y1)]
        elif direction == "left":
            x0, x1, x2 = cx+60, cx-30, cx-30-head_len
            y0, y1 = cy - shaft_w/2, cy + shaft_w/2
            pts = [(x0,y0),(x1,y0),(x1,cy-18),(x2,cy),(x1,cy+18),(x1,y1),(x0,y1)]
        elif direction == "up":
            y0, y1, y2 = cy+60, cy-30, cy-30-head_len
            x0, x1 = cx - shaft_w/2, cx + shaft_w/2
            pts = [(x0,y0),(x0,y1),(cx-18,y1),(cx,y2),(cx+18,y1),(x1,y1),(x1,y0)]
        elif direction == "down":
            y0, y1, y2 = cy-60, cy+30, cy+30+head_len
            x0, x1 = cx - shaft_w/2, cx + shaft_w/2
            pts = [(x0,y0),(x0,y1),(cx-18,y1),(cx,y2),(cx+18,y1),(x1,y1),(x1,y0)]
        return pts
    
    def points_to_path(pts):
        return " ".join([f"{'L' if i else 'M'}{x:.1f},{y:.1f}" for i,(x,y) in enumerate(pts)]) + " Z"
    
    def draw_arrow_in_circle(direction="right", bg="white", ring_stroke="black", arrow_fill="green"):
        svg = svg_header()
        svg += svg_bg(COL[bg])
        svg += f'<circle cx="128" cy="128" r="92" fill="none" stroke="{COL[ring_stroke]}" stroke-width="4"/>\n'
        pts = arrow_polygon_points(direction)
        path = points_to_path(pts)
        svg += f'<path d="{path}" fill="{COL[arrow_fill]}"/>\n'
        svg += svg_footer()
        return svg
    
    # Pacman template functions
    def draw_pacman(orientation="right", body_fill="yellow", eye_fill="black", bg="white"):
        svg = svg_header()
        svg += svg_bg(COL[bg])
        cx, cy, r = 128, 128, 92
        dir_angle = {"right":0, "up":-90, "left":180, "down":90}[orientation]
        mouth = 60
        a0 = math.radians(dir_angle - mouth/2)
        a1 = math.radians(dir_angle + mouth/2)
        x0, y0 = cx + r*math.cos(a0), cy + r*math.sin(a0)
        x1, y1 = cx + r*math.cos(a1), cy + r*math.sin(a1)
        d = f"M{cx},{cy} L{x0:.1f},{y0:.1f} A{r},{r} 0 0,1 {x1:.1f},{y1:.1f} Z"
        svg += f'<path d="{d}" fill="{COL.get(body_fill, body_fill)}"/>\n'
        eye_angle = math.radians(dir_angle - 60)
        ex, ey = cx + 30*math.cos(eye_angle), cy + 30*math.sin(eye_angle)
        svg += f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="8" fill="{COL.get(eye_fill, eye_fill)}"/>\n'
        svg += svg_footer()
        return svg
    
    # Additional shape drawing functions
    def draw_circle(center_x=128, center_y=128, radius=80, fill=COL["blue"], bg="white"):
        svg = svg_header()
        svg += svg_bg(COL[bg])
        svg += f'<circle cx="{center_x}" cy="{center_y}" r="{radius}" fill="{fill}"/>\n'
        svg += svg_footer()
        return svg
    
    def draw_square(center_x=128, center_y=128, size=120, fill=COL["green"], bg="white", rotation=0):
        svg = svg_header()
        svg += svg_bg(COL[bg])
        half = size / 2
        if rotation != 0:
            svg += f'<g transform="rotate({rotation} {center_x} {center_y})">\n'
            svg += f'<rect x="{center_x - half}" y="{center_y - half}" width="{size}" height="{size}" fill="{fill}"/>\n'
            svg += '</g>\n'
        else:
            svg += f'<rect x="{center_x - half}" y="{center_y - half}" width="{size}" height="{size}" fill="{fill}"/>\n'
        svg += svg_footer()
        return svg
    
    def draw_triangle(center_x=128, center_y=128, size=100, fill=COL["purple"], bg="white", orientation="up"):
        svg = svg_header()
        svg += svg_bg(COL[bg])
        h = size * math.sqrt(3) / 2
        if orientation == "up":
            pts = [(center_x, center_y - h*2/3), (center_x - size/2, center_y + h/3), (center_x + size/2, center_y + h/3)]
        elif orientation == "down":
            pts = [(center_x, center_y + h*2/3), (center_x - size/2, center_y - h/3), (center_x + size/2, center_y - h/3)]
        elif orientation == "left":
            pts = [(center_x - h*2/3, center_y), (center_x + h/3, center_y - size/2), (center_x + h/3, center_y + size/2)]
        else:  # right
            pts = [(center_x + h*2/3, center_y), (center_x - h/3, center_y - size/2), (center_x - h/3, center_y + size/2)]
        path = points_to_path(pts)
        svg += f'<path d="{path}" fill="{fill}"/>\n'
        svg += svg_footer()
        return svg
    
    def draw_star(center_x=128, center_y=128, outer_radius=70, inner_radius=35, fill=COL["orange"], bg="white", rotation=0):
        svg = svg_header()
        svg += svg_bg(COL[bg])
        points = []
        for i in range(10):
            angle = (math.pi * i) / 5 + math.radians(rotation) - math.pi / 2
            radius = outer_radius if i % 2 == 0 else inner_radius
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))
        path = points_to_path(points)
        svg += f'<path d="{path}" fill="{fill}"/>\n'
        svg += svg_footer()
        return svg
    
    def draw_diamond(center_x=128, center_y=128, width=100, height=100, fill=COL["pink"], bg="white", rotation=0):
        svg = svg_header()
        svg += svg_bg(COL[bg])
        w2, h2 = width / 2, height / 2
        pts = [(center_x, center_y - h2), (center_x + w2, center_y), (center_x, center_y + h2), (center_x - w2, center_y)]
        if rotation != 0:
            svg += f'<g transform="rotate({rotation} {center_x} {center_y})">\n'
            path = points_to_path(pts)
            svg += f'<path d="{path}" fill="{fill}"/>\n'
            svg += '</g>\n'
        else:
            path = points_to_path(pts)
            svg += f'<path d="{path}" fill="{fill}"/>\n'
        svg += svg_footer()
        return svg
    
    # State management functions with randomization
    use_random_initial_state = config.get("use_random_initial_state", True)
    
    def lshape_init():
        if use_random_initial_state:
            orientation = random.choice(ORIENTS)
            color = random.choice(COLOR_POOL)
            bg = random.choice(["white", "black"])
        else:
            orientation = "└"
            color = "red"
            bg = "white"
        return {"type": "lshape", "orientation": orientation, "color": color, "bg": bg}
    
    def lshape_apply(state, op, arg):
        s = state.copy()
        if op == "rotate":
            deg = arg if arg >= 0 else (360+arg)
            s["orientation"] = lshape_rotate(s["orientation"], deg)
        elif op == "mirror_h":
            s["orientation"] = lshape_mirror_h(s["orientation"])
        elif op == "mirror_v":
            s["orientation"] = lshape_mirror_v(s["orientation"])
        elif op == "map_colors":
            s["color"] = arg.get(s["color"], s["color"])
            s["bg"] = arg.get(s["bg"], s["bg"])
        return s
    
    def lshape_render(state):
        return lshape_make_scene(state["orientation"], state["color"], state["bg"])
    
    def arrow_init():
        if use_random_initial_state:
            dir = random.choice(["right", "down", "left", "up"])
            bg = random.choice(["white", "black"])
            ring = random.choice(["black", "white"])
            arrow = random.choice(COLOR_POOL)
        else:
            dir = "right"
            bg = "white"
            ring = "black"
            arrow = "green"
        return {"type": "arrow", "dir": dir, "bg": bg, "ring": ring, "arrow": arrow}
    
    def apply_rotation_dir(direction, deg):
        order = ["right", "down", "left", "up"]
        idx = order.index(direction)
        steps = (deg // 90) % 4
        return order[(idx + steps) % 4]
    
    def mirror_direction_h(direction):
        return {"right": "left", "left": "right", "up": "up", "down": "down"}[direction]
    
    def mirror_direction_v(direction):
        return {"up": "down", "down": "up", "left": "left", "right": "right"}[direction]
    
    def arrow_apply(state, op, arg):
        s = state.copy()
        if op == "rotate":
            deg = arg if arg >= 0 else (360+arg)
            s["dir"] = apply_rotation_dir(s["dir"], deg)
        elif op == "mirror_h":
            s["dir"] = mirror_direction_h(s["dir"])
        elif op == "mirror_v":
            s["dir"] = mirror_direction_v(s["dir"])
        elif op == "map_colors":
            for k in ["bg", "ring", "arrow"]:
                s[k] = arg.get(s[k], s[k])
        return s
    
    def arrow_render(state):
        return draw_arrow_in_circle(direction=state["dir"], bg=state["bg"], ring_stroke=state["ring"], arrow_fill=state["arrow"])
    
    def pacman_init():
        if use_random_initial_state:
            dir = random.choice(["right", "down", "left", "up"])
            body = random.choice(COLOR_POOL)
            eye = random.choice(["black", "white"])
            bg = random.choice(["white", "black"])
        else:
            dir = "right"
            body = "yellow"
            eye = "black"
            bg = "white"
        return {"type": "pacman", "dir": dir, "body": body, "eye": eye, "bg": bg}
    
    def pacman_apply(state, op, arg):
        s = state.copy()
        if op == "rotate":
            deg = arg if arg >= 0 else (360+arg)
            s["dir"] = apply_rotation_dir(s["dir"], deg)
        elif op == "mirror_h":
            s["dir"] = mirror_direction_h(s["dir"])
        elif op == "mirror_v":
            s["dir"] = mirror_direction_v(s["dir"])
        elif op == "map_colors":
            for k in ["body", "eye", "bg"]:
                s[k] = arg.get(s[k], s[k])
        return s
    
    def pacman_render(state):
        return draw_pacman(orientation=state["dir"], body_fill=state["body"], eye_fill=state["eye"], bg=state["bg"])
    
    # New template initializers and renderers
    def circle_init():
        if use_random_initial_state:
            radius = random.randint(50, 80)
            # Make positions more offset from center for more obvious shifts
            center_x = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Left or right side
            center_y = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Top or bottom
            fill = random.choice(COLOR_POOL)
            bg = random.choice(["white", "black"])
        else:
            radius = 70
            center_x = 90  # Offset to left
            center_y = 90  # Offset to top
            fill = "blue"
            bg = "white"
        return {"type": "circle", "center_x": center_x, "center_y": center_y, "radius": radius, "fill": fill, "bg": bg}
    
    def circle_apply(state, op, arg):
        s = state.copy()
        if op == "rotate":
            # For circle, rotation doesn't change appearance, but we can swap center for effect
            deg = arg if arg >= 0 else (360+arg)
            if deg == 180:
                # Flip position - make shift more obvious
                s["center_x"] = 256 - s["center_x"]
                s["center_y"] = 256 - s["center_y"]
        elif op == "mirror_h":
            # Horizontal mirror - flip across vertical center (128)
            s["center_x"] = 256 - s["center_x"]
        elif op == "mirror_v":
            # Vertical mirror - flip across horizontal center (128)
            s["center_y"] = 256 - s["center_y"]
        elif op == "map_colors":
            s["fill"] = arg.get(s["fill"], s["fill"])
            s["bg"] = arg.get(s["bg"], s["bg"])
        return s
    
    def circle_render(state):
        return draw_circle(state["center_x"], state["center_y"], state["radius"], COL[state["fill"]], state["bg"])
    
    def square_init():
        if use_random_initial_state:
            size = random.randint(70, 110)
            # Make positions more offset from center for more obvious shifts
            center_x = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Left or right side
            center_y = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Top or bottom
            rotation = random.choice([0, 45, 90, 135])
            fill = random.choice(COLOR_POOL)
            bg = random.choice(["white", "black"])
        else:
            size = 100
            center_x = 90  # Offset to left
            center_y = 90  # Offset to top
            rotation = 0
            fill = "green"
            bg = "white"
        return {"type": "square", "center_x": center_x, "center_y": center_y, "size": size, "rotation": rotation, "fill": fill, "bg": bg}
    
    def square_apply(state, op, arg):
        s = state.copy()
        if op == "rotate":
            deg = arg if arg >= 0 else (360+arg)
            s["rotation"] = (s["rotation"] + deg) % 360
        elif op == "mirror_h":
            s["rotation"] = (360 - s["rotation"]) % 360
            # Horizontal mirror - flip across vertical center (128)
            s["center_x"] = 256 - s["center_x"]
        elif op == "mirror_v":
            s["rotation"] = (180 - s["rotation"]) % 360
            # Vertical mirror - flip across horizontal center (128)
            s["center_y"] = 256 - s["center_y"]
        elif op == "map_colors":
            s["fill"] = arg.get(s["fill"], s["fill"])
            s["bg"] = arg.get(s["bg"], s["bg"])
        return s
    
    def square_render(state):
        return draw_square(state["center_x"], state["center_y"], state["size"], COL[state["fill"]], state["bg"], state["rotation"])
    
    def triangle_init():
        if use_random_initial_state:
            size = random.randint(60, 100)
            # Make positions more offset from center for more obvious shifts
            center_x = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Left or right side
            center_y = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Top or bottom
            orientation = random.choice(["up", "down", "left", "right"])
            fill = random.choice(COLOR_POOL)
            bg = random.choice(["white", "black"])
        else:
            size = 90
            center_x = 90  # Offset to left
            center_y = 90  # Offset to top
            orientation = "up"
            fill = "purple"
            bg = "white"
        return {"type": "triangle", "center_x": center_x, "center_y": center_y, "size": size, "orientation": orientation, "fill": fill, "bg": bg}
    
    def triangle_apply(state, op, arg):
        s = state.copy()
        orient_order = ["up", "right", "down", "left"]
        if op == "rotate":
            deg = arg if arg >= 0 else (360+arg)
            steps = (deg // 90) % 4
            idx = orient_order.index(s["orientation"])
            s["orientation"] = orient_order[(idx + steps) % 4]
        elif op == "mirror_h":
            s["orientation"] = {"up": "up", "down": "down", "left": "right", "right": "left"}[s["orientation"]]
            # Horizontal mirror - flip across vertical center (128)
            s["center_x"] = 256 - s["center_x"]
        elif op == "mirror_v":
            s["orientation"] = {"left": "left", "right": "right", "up": "down", "down": "up"}[s["orientation"]]
            # Vertical mirror - flip across horizontal center (128)
            s["center_y"] = 256 - s["center_y"]
        elif op == "map_colors":
            s["fill"] = arg.get(s["fill"], s["fill"])
            s["bg"] = arg.get(s["bg"], s["bg"])
        return s
    
    def triangle_render(state):
        return draw_triangle(state["center_x"], state["center_y"], state["size"], COL[state["fill"]], state["bg"], state["orientation"])
    
    def star_init():
        if use_random_initial_state:
            outer_radius = random.randint(50, 75)
            inner_radius = random.randint(20, outer_radius // 2)
            # Make positions more offset from center for more obvious shifts
            center_x = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Left or right side
            center_y = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Top or bottom
            rotation = random.randint(0, 359)
            fill = random.choice(COLOR_POOL)
            bg = random.choice(["white", "black"])
        else:
            outer_radius = 65
            inner_radius = 32
            center_x = 90  # Offset to left
            center_y = 90  # Offset to top
            rotation = 0
            fill = "orange"
            bg = "white"
        return {"type": "star", "center_x": center_x, "center_y": center_y, "outer_radius": outer_radius, "inner_radius": inner_radius, "rotation": rotation, "fill": fill, "bg": bg}
    
    def star_apply(state, op, arg):
        s = state.copy()
        if op == "rotate":
            deg = arg if arg >= 0 else (360+arg)
            s["rotation"] = (s["rotation"] + deg) % 360
        elif op == "mirror_h":
            s["rotation"] = (360 - s["rotation"]) % 360
            # Horizontal mirror - flip across vertical center (128)
            s["center_x"] = 256 - s["center_x"]
        elif op == "mirror_v":
            s["rotation"] = (180 - s["rotation"]) % 360
            # Vertical mirror - flip across horizontal center (128)
            s["center_y"] = 256 - s["center_y"]
        elif op == "map_colors":
            s["fill"] = arg.get(s["fill"], s["fill"])
            s["bg"] = arg.get(s["bg"], s["bg"])
        return s
    
    def star_render(state):
        return draw_star(state["center_x"], state["center_y"], state["outer_radius"], state["inner_radius"], COL[state["fill"]], state["bg"], state["rotation"])
    
    def diamond_init():
        if use_random_initial_state:
            width = random.randint(70, 110)
            height = random.randint(70, 110)
            # Make positions more offset from center for more obvious shifts
            center_x = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Left or right side
            center_y = random.choice([random.randint(60, 90), random.randint(166, 196)])  # Top or bottom
            rotation = random.choice([0, 45, 90, 135])
            fill = random.choice(COLOR_POOL)
            bg = random.choice(["white", "black"])
        else:
            width = 90
            height = 90
            center_x = 90  # Offset to left
            center_y = 90  # Offset to top
            rotation = 0
            fill = "pink"
            bg = "white"
        return {"type": "diamond", "center_x": center_x, "center_y": center_y, "width": width, "height": height, "rotation": rotation, "fill": fill, "bg": bg}
    
    def diamond_apply(state, op, arg):
        s = state.copy()
        if op == "rotate":
            deg = arg if arg >= 0 else (360+arg)
            s["rotation"] = (s["rotation"] + deg) % 360
        elif op == "mirror_h":
            s["rotation"] = (360 - s["rotation"]) % 360
            # Horizontal mirror - flip across vertical center (128)
            s["center_x"] = 256 - s["center_x"]
        elif op == "mirror_v":
            s["rotation"] = (180 - s["rotation"]) % 360
            # Vertical mirror - flip across horizontal center (128)
            s["center_y"] = 256 - s["center_y"]
        elif op == "map_colors":
            s["fill"] = arg.get(s["fill"], s["fill"])
            s["bg"] = arg.get(s["bg"], s["bg"])
        return s
    
    def diamond_render(state):
        return draw_diamond(state["center_x"], state["center_y"], state["width"], state["height"], COL[state["fill"]], state["bg"], state["rotation"])
    
    # Template registry
    template_registry = {
        "lshape": (lshape_init, lshape_apply, lshape_render),
        "arrow_circle": (arrow_init, arrow_apply, arrow_render),
        "pacman": (pacman_init, pacman_apply, pacman_render),
        "circle": (circle_init, circle_apply, circle_render),
        "square": (square_init, square_apply, square_render),
        "triangle": (triangle_init, triangle_apply, triangle_render),
        "star": (star_init, star_apply, star_render),
        "diamond": (diamond_init, diamond_apply, diamond_render)
    }
    
    # Handle unknown templates gracefully
    if template_name not in template_registry:
        template_name = "lshape"  # Fallback to lshape
    
    init_fn, apply_fn, render_fn = template_registry[template_name]
    
    # Generate initial state
    state0 = init_fn()
    
    # If pipeline has color mappings, ensure initial state uses ALL colors that will be affected
    # This ensures ALL color mapping steps are visible
    color_mappings = [arg for op, arg in pipeline if op == "map_colors"]
    if color_mappings and use_random_initial_state:
        # Get all source colors from mappings - we need ALL of them visible
        source_colors_list = []
        for mapping in color_mappings:
            source_colors_list.extend(mapping.keys())
        source_colors = list(set(source_colors_list))  # Remove duplicates
        
        # Separate colors into COLOR_POOL colors and black/white
        pool_colors = [c for c in source_colors if c in COLOR_POOL]
        bw_colors = [c for c in source_colors if c in ["black", "white"]]
        
        # Ensure ALL mapped colors are assigned to visible properties
        # Different templates have different color properties
        if template_name == "lshape":
            # Can use: color, bg (2 colors)
            if len(pool_colors) >= 1:
                state0["color"] = pool_colors[0]
            elif len(source_colors) >= 1:
                state0["color"] = source_colors[0] if source_colors[0] in COLOR_POOL else random.choice(COLOR_POOL)
            if len(bw_colors) >= 1:
                state0["bg"] = bw_colors[0]
            elif len(pool_colors) >= 2:
                # Use second pool color even if not black/white - better than missing a mapping
                state0["bg"] = pool_colors[1] if pool_colors[1] in ["white", "black"] else random.choice(["white", "black"])
            elif len(source_colors) >= 2:
                state0["bg"] = source_colors[1] if source_colors[1] in ["white", "black"] else random.choice(["white", "black"])
        elif template_name == "arrow_circle":
            # Can use: arrow, ring, bg (3 colors!)
            # Strategy: Assign pool colors to arrow and ring/bg to ensure ALL pool colors are visible
            if len(pool_colors) >= 1:
                state0["arrow"] = pool_colors[0]  # First pool color goes to arrow
            elif len(source_colors) >= 1:
                state0["arrow"] = source_colors[0] if source_colors[0] in COLOR_POOL else random.choice(COLOR_POOL)
            # For ring and bg: prioritize showing second pool color if we have 2+
            if len(pool_colors) >= 2:
                # We have 2+ pool colors - assign second to ring so both are visible
                state0["ring"] = pool_colors[1]
                # bg can be black/white or third pool color
                if len(bw_colors) >= 1:
                    state0["bg"] = bw_colors[0]
                elif len(pool_colors) >= 3:
                    state0["bg"] = pool_colors[2]
                else:
                    state0["bg"] = random.choice(["white", "black"])
            elif len(bw_colors) >= 1:
                # Only 1 pool color, use black/white for ring
                state0["ring"] = bw_colors[0]
                if len(bw_colors) >= 2:
                    state0["bg"] = bw_colors[1]
                else:
                    state0["bg"] = random.choice(["white", "black"])
            elif len(source_colors) >= 2:
                state0["ring"] = source_colors[1] if source_colors[1] in ["black", "white"] else random.choice(["black", "white"])
                state0["bg"] = random.choice(["white", "black"])
            else:
                state0["ring"] = random.choice(["black", "white"])
                state0["bg"] = random.choice(["white", "black"])
        elif template_name == "pacman":
            # Can use: body, eye, bg (3 colors!)
            # Strategy: Assign pool colors to body and eye/bg to ensure ALL pool colors are visible
            if len(pool_colors) >= 1:
                state0["body"] = pool_colors[0]  # First pool color goes to body
            elif len(source_colors) >= 1:
                state0["body"] = source_colors[0] if source_colors[0] in COLOR_POOL else random.choice(COLOR_POOL)
            # For eye and bg: prioritize showing second pool color if we have 2+
            if len(pool_colors) >= 2:
                # We have 2+ pool colors - assign second to eye so both are visible
                state0["eye"] = pool_colors[1]
                # bg can be black/white or third pool color
                if len(bw_colors) >= 1:
                    state0["bg"] = bw_colors[0]
                elif len(pool_colors) >= 3:
                    state0["bg"] = pool_colors[2]
                else:
                    state0["bg"] = random.choice(["white", "black"])
            elif len(bw_colors) >= 1:
                # Only 1 pool color, use black/white for eye
                state0["eye"] = bw_colors[0]
                if len(bw_colors) >= 2:
                    state0["bg"] = bw_colors[1]
                else:
                    state0["bg"] = random.choice(["white", "black"])
            elif len(source_colors) >= 2:
                state0["eye"] = source_colors[1] if source_colors[1] in ["black", "white"] else random.choice(["black", "white"])
                state0["bg"] = random.choice(["white", "black"])
            else:
                state0["eye"] = random.choice(["black", "white"])
                state0["bg"] = random.choice(["white", "black"])
        elif template_name in ["circle", "square", "triangle", "star", "diamond"]:
            # Can use: fill, bg (2 colors only)
            # IMPORTANT: For 2-color templates with 2+ pool colors, we can only show 1 pool color
            # So we need to ensure the template selection prefers 3-color templates for 2+ pool colors
            # But if we end up here, prioritize showing at least one pool color
            if len(pool_colors) >= 1:
                state0["fill"] = pool_colors[0]  # Show first pool color
            elif len(source_colors) >= 1:
                state0["fill"] = source_colors[0] if source_colors[0] in COLOR_POOL else random.choice(COLOR_POOL)
            # For bg, prioritize black/white if available, otherwise use second pool color
            if len(bw_colors) >= 1:
                state0["bg"] = bw_colors[0]
            elif len(pool_colors) >= 2:
                # If we have 2+ pool colors but only 2-color template, use second pool color for bg
                # This ensures both colors are visible (though bg usually isn't pool colors)
                state0["bg"] = pool_colors[1] if pool_colors[1] in ["white", "black"] else random.choice(["white", "black"])
            elif len(source_colors) >= 2:
                state0["bg"] = source_colors[1] if source_colors[1] in ["white", "black"] else random.choice(["white", "black"])
    
    # Apply pipeline to get correct state
    def apply_pipeline(state, pipeline, appliers):
        s = state.copy()  # Make a copy to avoid mutating the original
        for op, arg in pipeline:
            s = appliers(s, op, arg)
        return s
    
    correct_state = apply_pipeline(state0, pipeline, apply_fn)
    
    # Generate distractors
    def make_distractors(state, pipeline, appliers):
        distractors = []
        # Skip last step
        s = state
        for i, (op, arg) in enumerate(pipeline):
            if i < len(pipeline) - 1:
                s = appliers(s, op, arg)
        distractors.append(("skip_last", s))
        # Skip middle step
        s = state
        for i, (op, arg) in enumerate(pipeline):
            if i == 1:
                continue
            s = appliers(s, op, arg)
        distractors.append(("skip_middle", s))
        # Skip first step
        s = state
        for i, (op, arg) in enumerate(pipeline):
            if i == 0:
                continue
            s = appliers(s, op, arg)
        distractors.append(("skip_first", s))
        # Wrong rotation direction
        if pipeline[0][0] == "rotate":
            wrong = state.copy()
            op, arg = pipeline[0]
            wrong = appliers(wrong, op, -arg if isinstance(arg, int) else arg)
            for op2, arg2 in pipeline[1:]:
                wrong = appliers(wrong, op2, arg2)
            distractors.append(("wrong_rot_dir", wrong))
        # No color map
        has_map = any(op == "map_colors" for op, _ in pipeline)
        if has_map:
            s = state.copy()
            for op, arg in pipeline:
                if op == "map_colors":
                    continue
                s = appliers(s, op, arg)
            distractors.append(("no_color_map", s))
        # Only first step
        s = state.copy()
        op, arg = pipeline[0]
        s = appliers(s, op, arg)
        distractors.append(("only_first", s))
        return distractors
    
    distractors = make_distractors(state0, pipeline, apply_fn)
    
    # Render SVG to base64
    def svg_to_data_url(svg_content):
        svg_bytes = svg_content.encode('utf-8')
        svg_b64 = base64.b64encode(svg_bytes).decode('utf-8')
        return f"data:image/svg+xml;base64,{svg_b64}"
    
    # Render reference image
    reference_svg = render_fn(state0)
    reference_data_url = svg_to_data_url(reference_svg)
    
    # Collect options
    def fingerprint(s):
        return json_module.dumps(s, sort_keys=True)
    
    options_states = [correct_state]
    used = {fingerprint(correct_state)}
    
    for name, st in distractors:
        fp = fingerprint(st)
        if fp not in used:
            used.add(fp)
            options_states.append(st)
        if len(options_states) >= 8:
            break
    
    # Fill to 8 options if needed
    while len(options_states) < 8 and distractors:
        options_states.append(distractors[len(options_states) % len(distractors)][1])
    
    # Shuffle options
    idxs = list(range(len(options_states)))
    random.shuffle(idxs)
    
    correct_idx = None
    option_data_urls = []
    for i, opt_i in enumerate(idxs):
        st = options_states[opt_i]
        svg = render_fn(st)
        option_data_urls.append(svg_to_data_url(svg))
        if fingerprint(st) == fingerprint(correct_state):
            correct_idx = i
    
    # Generate transform steps description
    def serialize_steps(pipeline):
        out = []
        for op, arg in pipeline:
            if op == "rotate":
                if arg > 0:
                    out.append(f"Rotate {arg}° clockwise")
                else:
                    out.append(f"Rotate {abs(arg)}° counterclockwise")
            elif op == "mirror_h":
                out.append("Horizontal mirror")
            elif op == "mirror_v":
                out.append("Vertical mirror")
            elif op == "map_colors":
                mapping = ", ".join([f"{k}→{v}" for k, v in arg.items()])
                out.append(f"Color mapping: {mapping}")
        return out
    
    transform_steps = serialize_steps(pipeline)
    
    # Generate puzzle ID
    puzzle_id = f"transform_{int(time.time()*1000)}_{random.randint(1000,9999)}"
    
    # Store puzzle state for validation
    active_transform_pipeline_puzzles[puzzle_id] = {
        "correct_index": correct_idx,
        "template": template_name,
        "pipeline": pipeline,
        "created_at": time.time()
    }
    
    # Determine grid size based on number of options
    num_options = len(option_data_urls)
    if num_options <= 4:
        grid_size = [2, 2]
    elif num_options <= 6:
        grid_size = [2, 3]
    else:
        grid_size = [2, 4]
    
    return {
        "puzzle_id": puzzle_id,
        "reference_image": reference_data_url,
        "option_images": option_data_urls,
        "transform_steps": transform_steps,
        "answer": correct_idx,
        "grid_size": grid_size,
        "prompt": "After following the transform steps, what will be the last image?"
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
    elif puzzle_type == "Static_Jigsaw":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "Drag the puzzle pieces to complete the jigsaw puzzle"
        )
    elif puzzle_type == "Transform_Pipeline":
        prompt = ground_truth[selected_puzzle].get(
            "prompt",
            "After following the transform steps, what will be the last image?"
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
    elif puzzle_type == "Hole_Counting":
        input_type = "hole_counting_select"
    elif puzzle_type == "Rotation_Match":
        input_type = "rotation_match_select"
    elif puzzle_type == "Rhythm":
        input_type = "rhythm_select"
    elif puzzle_type == "Backmost_Layer":
        input_type = "backmost_layer_select"
    elif puzzle_type == "Shadow_Direction":
        input_type = "shadow_direction_select"
    elif puzzle_type == "Global_Phase_Drift":
        input_type = "global_phase_drift_select"
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
    elif puzzle_type == "Static_Jigsaw":
        input_type = "jigsaw_puzzle"
    elif puzzle_type == "Transform_Pipeline":
        input_type = "transform_pipeline_select"
    elif puzzle_type == "Set_Game":
        input_type = "set_game_select"
    elif puzzle_type == "Dynamic_Jigsaw":
        input_type = "jigsaw_puzzle"
    elif puzzle_type == "Spooky_Jigsaw":
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
    elif puzzle_type == "Transform_Pipeline":
        # Check if we should generate a random puzzle or use ground truth
        use_generation = ground_truth.get("config", {}).get("generate_random", False)
        
        if use_generation:
            # Generate random puzzle
            try:
                config = ground_truth.get("config", {})
                puzzle = generate_transform_pipeline(config)
                puzzle_id = puzzle["puzzle_id"]
                
                additional_data = {
                    "reference_image": puzzle["reference_image"],  # Base64 data URL
                    "option_images": puzzle["option_images"],  # Base64 data URLs
                    "transform_steps": puzzle["transform_steps"],
                    "answer": puzzle["answer"],
                    "grid_size": puzzle["grid_size"]
                }
                
                response_data = {
                    'puzzle_type': puzzle_type,
                    'image_path': None,
                    'media_path': None,
                    'media_type': None,
                    'puzzle_id': puzzle_id,
                    'prompt': puzzle["prompt"],
                    'input_type': 'transform_pipeline_select',
                    'debug_info': f"Type: {puzzle_type}, Generated puzzle: {puzzle_id}"
                }
                response_data.update(additional_data)
                return jsonify(response_data)
            except Exception as e:
                # Fall back to ground truth if generation fails
                print(f"Transform_Pipeline generation failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Use ground truth puzzles
        reference_image = ground_truth[selected_puzzle].get("reference")
        option_images = ground_truth[selected_puzzle].get("options", [])
        transform_steps = ground_truth[selected_puzzle].get("transform_steps", [])
        if not reference_image or not option_images:
            return jsonify({'error': f'Invalid Transform_Pipeline data: {selected_puzzle}'}), 500

        additional_data = {
            "reference_image": f'/captcha_data/{puzzle_type}/{reference_image}',
            "option_images": [f'/captcha_data/{puzzle_type}/{img}' for img in option_images],
            "transform_steps": transform_steps,
            "answer": ground_truth[selected_puzzle].get("answer"),
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [2, 3])
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
        }
    elif puzzle_type == "Hole_Counting":
        # Load cell pool to map cell IDs to filenames
        cell_pool_path = os.path.join('captcha_data', puzzle_type, 'cell_pool.json')
        with open(cell_pool_path) as f:
            cell_pool = json.load(f)

        cell_ids = ground_truth[selected_puzzle].get("cells", [])
        if not cell_ids:
            return jsonify({'error': f'Invalid Hole_Counting data: {selected_puzzle}'}), 500

        # Convert cell IDs to image paths
        cell_images = []
        for cell_id in cell_ids:
            if cell_id in cell_pool:
                cell_images.append(f'/captcha_data/{puzzle_type}/{cell_pool[cell_id]["filename"]}')
            else:
                return jsonify({'error': f'Cell not found in pool: {cell_id}'}), 500

        additional_data = {
            "option_images": cell_images,
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [4, 4]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Rotation_Match":
        # Load cell pool to map cell IDs to filenames
        cell_pool_path = os.path.join('captcha_data', puzzle_type, 'cell_pool.json')
        with open(cell_pool_path) as f:
            cell_pool = json.load(f)

        cell_ids = ground_truth[selected_puzzle].get("cells", [])
        if not cell_ids:
            return jsonify({'error': f'Invalid Rotation_Match data: {selected_puzzle}'}), 500

        # Convert cell IDs to image paths
        cell_images = []
        for cell_id in cell_ids:
            if cell_id in cell_pool:
                cell_images.append(f'/captcha_data/{puzzle_type}/{cell_pool[cell_id]["filename"]}')
            else:
                return jsonify({'error': f'Cell not found in pool: {cell_id}'}), 500

        additional_data = {
            "option_images": cell_images,
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [4, 4]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Rhythm":
        # Load cell pool to map cell IDs to filenames
        cell_pool_path = os.path.join('captcha_data', puzzle_type, 'cell_pool.json')
        with open(cell_pool_path) as f:
            cell_pool = json.load(f)

        cell_ids = ground_truth[selected_puzzle].get("cells", [])
        reference_cell = ground_truth[selected_puzzle].get("reference_cell")
        if not cell_ids or not reference_cell:
            return jsonify({'error': f'Invalid Rhythm data: {selected_puzzle}'}), 500

        # Convert cell IDs to GIF paths
        cell_images = []
        for cell_id in cell_ids:
            if cell_id in cell_pool:
                cell_images.append(f'/captcha_data/{puzzle_type}/{cell_pool[cell_id]["filename"]}')
            else:
                return jsonify({'error': f'Cell not found in pool: {cell_id}'}), 500

        # Get reference GIF
        if reference_cell in cell_pool:
            reference_gif = f'/captcha_data/{puzzle_type}/{cell_pool[reference_cell]["filename"]}'
        else:
            return jsonify({'error': f'Reference cell not found: {reference_cell}'}), 500

        additional_data = {
            "reference_gif": reference_gif,
            "option_images": cell_images,
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [4, 4]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Backmost_Layer":
        # Load cell pool to map cell IDs to filenames
        cell_pool_path = os.path.join('captcha_data', puzzle_type, 'cell_pool.json')
        with open(cell_pool_path) as f:
            cell_pool = json.load(f)

        cell_ids = ground_truth[selected_puzzle].get("cells", [])
        reference_cell = ground_truth[selected_puzzle].get("reference_cell")
        if not cell_ids or not reference_cell:
            return jsonify({'error': f'Invalid Backmost_Layer data: {selected_puzzle}'}), 500

        # Convert cell IDs to image paths
        cell_images = []
        for cell_id in cell_ids:
            if cell_id in cell_pool:
                cell_images.append(f'/captcha_data/{puzzle_type}/{cell_pool[cell_id]["filename"]}')
            else:
                return jsonify({'error': f'Cell not found in pool: {cell_id}'}), 500

        # Get reference image
        if reference_cell in cell_pool:
            reference_image = f'/captcha_data/{puzzle_type}/{cell_pool[reference_cell]["filename"]}'
        else:
            return jsonify({'error': f'Reference cell not found: {reference_cell}'}), 500

        additional_data = {
            "reference_image": reference_image,
            "option_images": cell_images,
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [4, 4]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Shadow_Direction":
        # Load cell pool to map cell IDs to filenames
        cell_pool_path = os.path.join('captcha_data', puzzle_type, 'cell_pool.json')
        with open(cell_pool_path) as f:
            cell_pool = json.load(f)

        cell_ids = ground_truth[selected_puzzle].get("cells", [])
        reference_image_file = ground_truth[selected_puzzle].get("reference_image")

        if not cell_ids or not reference_image_file:
            return jsonify({'error': f'Invalid Shadow_Direction data: {selected_puzzle}'}), 500

        # Convert cell IDs to image paths
        cell_images = []
        for cell_id in cell_ids:
            if cell_id in cell_pool:
                cell_images.append(f'/captcha_data/{puzzle_type}/{cell_pool[cell_id]["filename"]}')
            else:
                return jsonify({'error': f'Cell not found in pool: {cell_id}'}), 500

        # Get reference image (arrow showing light direction)
        reference_image = f'/captcha_data/{puzzle_type}/{reference_image_file}'

        additional_data = {
            "reference_image": reference_image,
            "option_images": cell_images,
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [4, 4]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
        }
    elif puzzle_type == "Global_Phase_Drift":
        # Load 16 individual GIF files for the grid
        puzzle_dir = ground_truth[selected_puzzle].get("puzzle_dir")
        cell_files = ground_truth[selected_puzzle].get("cell_files", [])

        if not puzzle_dir or not cell_files:
            return jsonify({'error': f'Invalid Global_Phase_Drift data: {selected_puzzle}'}), 500

        # Build paths to 16 GIF files
        cell_gifs = []
        for cell_file in cell_files:
            cell_gifs.append(f'/captcha_data/{puzzle_type}/{puzzle_dir}/{cell_file}')

        prompt = "Watch the animations carefully - one cell is out of sync with the wave pattern"

        additional_data = {
            "cell_gifs": cell_gifs,
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [4, 4]),
            "answer": ground_truth[selected_puzzle].get("answer", [])
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
    elif puzzle_type == "Static_Jigsaw":
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
            return jsonify({'error': f'Invalid Static_Jigsaw data: {selected_puzzle}'}), 500

        additional_data = {
            "pieces": [f'/captcha_data/{puzzle_type}/{piece}' for piece in pieces],
            "grid_size": grid_size,
            "piece_size": piece_size,
            "correct_positions": correct_positions,
            "reference_image": f'/captcha_data/{puzzle_type}/{reference_image}' if reference_image else None
        }
    elif puzzle_type == "Set_Game":
        option_images = ground_truth[selected_puzzle].get("options", [])
        if not option_images:
            return jsonify({'error': f'Invalid Set_Game data: {selected_puzzle}'}), 500

        additional_data = {
            "option_images": [f'/captcha_data/{puzzle_type}/{img}' for img in option_images],
            "grid_size": ground_truth[selected_puzzle].get("grid_size", [4, 4]),
            "answer": ground_truth[selected_puzzle].get("answer", []),
            "num_sets": ground_truth[selected_puzzle].get("num_sets", 0)
        }
    elif puzzle_type == "Dynamic_Jigsaw":
        # Dynamic Jigsaw with animated GIF pieces
        pieces = ground_truth[selected_puzzle].get("pieces", [])
        grid_size = ground_truth[selected_puzzle].get("grid_size", [2, 2])
        piece_size = ground_truth[selected_puzzle].get("piece_size", 150)
        correct_positions = ground_truth[selected_puzzle].get("correct_positions", [])
        reference_image = ground_truth[selected_puzzle].get("image")

        if not pieces or not correct_positions:
            return jsonify({'error': f'Invalid Dynamic_Jigsaw data: {selected_puzzle}'}), 500

        additional_data = {
            "pieces": [f'/captcha_data/{puzzle_type}/{piece}' for piece in pieces],
            "grid_size": grid_size,
            "piece_size": piece_size,
            "correct_positions": correct_positions,
            "reference_image": f'/captcha_data/{puzzle_type}/{reference_image}' if reference_image else None
        }
    elif puzzle_type == "Spooky_Jigsaw":
        # Spooky Jigsaw with motion-based visibility
        pieces = ground_truth[selected_puzzle].get("pieces", [])
        grid_size = ground_truth[selected_puzzle].get("grid_size", [3, 3])
        piece_size = ground_truth[selected_puzzle].get("piece_size", 150)
        correct_positions = ground_truth[selected_puzzle].get("correct_positions", [])
        reference_image = ground_truth[selected_puzzle].get("image")

        if not pieces or not correct_positions:
            return jsonify({'error': f'Invalid Spooky_Jigsaw data: {selected_puzzle}'}), 500

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
    if puzzle_type not in ("Rotation_Match", "Shadow_Plausible", "Mirror",  "Squiggle", "Spooky_Circle_Grid", "Spooky_Circle_Grid_Direction", "Spooky_Shape_Grid", "Color_Cipher", "Color_Counting", "Hole_Counting", "Rhythm", "Backmost_Layer", "Shadow_Direction", "Global_Phase_Drift", "Trajectory_Recovery", "Storyboard_Logic", "Static_Jigsaw", "Transform_Pipeline", "Set_Game", "Dynamic_Jigsaw", "Spooky_Jigsaw"):
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
    # For Static_Jigsaw, allow None/empty answers to be handled gracefully (marked as incorrect)
    if not puzzle_id or (user_answer is None and puzzle_type != 'Static_Jigsaw'):
        return jsonify({'error': 'Missing puzzle_id or answer'}), 400
    
    ground_truth = load_ground_truth(puzzle_type)
    
    if puzzle_type not in ('Color_Cipher', 'Red_Dot', 'Spooky_Size', 'Static_Jigsaw', 'Transform_Pipeline') and puzzle_id not in ground_truth:
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
    elif puzzle_type == 'Transform_Pipeline':
        try:
            # Check if this is a generated puzzle (stored in active_transform_pipeline_puzzles)
            puzzle_state = active_transform_pipeline_puzzles.get(puzzle_id)
            
            if puzzle_state:
                # Generated puzzle - use stored correct index
                correct_index = puzzle_state.get('correct_index')
            else:
                # Ground truth puzzle
                if puzzle_id not in ground_truth:
                    return jsonify({'error': 'Invalid puzzle ID'}), 400
                correct_index = int(ground_truth[puzzle_id].get('answer'))
            
            user_index = int(user_answer)
            is_correct = user_index == correct_index
            correct_answer_info = correct_index
            
            # Clean up generated puzzle state after validation
            if puzzle_state:
                active_transform_pipeline_puzzles.pop(puzzle_id, None)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Transform_Pipeline'}), 400
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
    elif puzzle_type == 'Hole_Counting':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Hole_Counting'}), 400
    elif puzzle_type == 'Rotation_Match':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Rotation_Match'}), 400
    elif puzzle_type == 'Rhythm':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Rhythm'}), 400
    elif puzzle_type == 'Backmost_Layer':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Backmost_Layer'}), 400
    elif puzzle_type == 'Shadow_Direction':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Shadow_Direction'}), 400
    elif puzzle_type == 'Global_Phase_Drift':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Global_Phase_Drift'}), 400
    elif puzzle_type == 'Trajectory_Recovery':
        try:
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Trajectory_Recovery'}), 400
    elif puzzle_type == 'Set_Game':
        try:
            # For Set_Game, answer is a flat list of 3 card indices [7, 9, 11]
            correct_indices = sorted(ground_truth[puzzle_id].get('answer', []))
            user_indices = sorted(int(idx) for idx in user_answer)
            is_correct = user_indices == correct_indices
            correct_answer_info = correct_indices
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Set_Game'}), 400
    elif puzzle_type == 'Storyboard_Logic':
        try:
            # For Storyboard_Logic, order matters - check exact sequence match
            correct_order = ground_truth[puzzle_id].get('answer', [])
            user_order = [int(idx) for idx in user_answer]
            is_correct = user_order == correct_order
            correct_answer_info = correct_order
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid answer format for Storyboard_Logic'}), 400
    elif puzzle_type == 'Static_Jigsaw' or puzzle_type == 'Dynamic_Jigsaw' or puzzle_type == 'Spooky_Jigsaw':
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
            
            # Handle cases where agent submits directly without placing pieces
            # If user_answer is None, empty list, or not a list, mark as incorrect immediately
            if user_answer is None or not isinstance(user_answer, list) or len(user_answer) == 0:
                is_correct = False
                correct_answer_info = correct_positions
                
                # Clean up generated puzzle state after validation
                if puzzle_state:
                    active_jigsaw_puzzles.pop(puzzle_id, None)
                
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
                
                return jsonify({
                    'correct': False,
                    'user_answer': user_answer,
                    'correct_answer': correct_payload,
                    'details': {
                        'user_placements': user_answer if user_answer else [],
                        'correct_positions': correct_answer_info
                    }
                })
            
            user_placements = user_answer
            
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
            return jsonify({'error': f'Invalid answer format for Static_Jigsaw: {str(e)}'}), 400
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
        # Get the appropriate answer field based on puzzle type
        if puzzle_type == 'Dice_Count':
            answer_key = 'sum'
        else:
            answer_key = 'answer'
        correct_answer = ground_truth[puzzle_id].get(answer_key)
        is_correct = str(user_answer).lower() == str(correct_answer).lower()
        correct_answer_info = correct_answer
    
    # Get the appropriate answer field for response payload
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
            if puzzle_id in ground_truth:
                correct_payload = ground_truth[puzzle_id].get(answer_key)
            else:
                correct_payload = str(correct_answer_info) if correct_answer_info is not None else "Unknown"
    elif puzzle_type == 'Static_Jigsaw' or puzzle_type == 'Dynamic_Jigsaw' or puzzle_type == 'Spooky_Jigsaw':
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
    elif puzzle_type == 'Transform_Pipeline':
        # Format the correct answer index
        if isinstance(correct_answer_info, int):
            correct_payload = f"Option {correct_answer_info + 1} (index {correct_answer_info})"
        else:
            correct_payload = f"Correct answer: {correct_answer_info}"
    else:
        # Only access ground_truth if puzzle_id exists in it (not generated puzzles)
        if puzzle_id in ground_truth:
            correct_payload = ground_truth[puzzle_id].get(answer_key)
        else:
            # Fallback for generated puzzles not in ground_truth
            correct_payload = str(correct_answer_info) if correct_answer_info is not None else "Unknown"

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
    if puzzle_type == 'Static_Jigsaw' and not is_correct:
        # Include debug details for incorrect jigsaw puzzles
        response_body['details'] = {
            'user_placements': user_answer if puzzle_type == 'Static_Jigsaw' else None,
            'correct_positions': correct_answer_info if puzzle_type == 'Static_Jigsaw' else None
        }
    elif (puzzle_type == 'Dynamic_Jigsaw' or puzzle_type == 'Spooky_Jigsaw') and not is_correct:
        # Include debug details for incorrect jigsaw puzzles
        response_body['details'] = {
            'user_placements': user_answer,
            'correct_positions': correct_answer_info
        }

    return jsonify(response_body)

@app.route('/api/benchmark_results', methods=['POST'])
def record_benchmark():
    data = request.json or {}
    if not isinstance(data, dict):
        return jsonify({'status': 'error', 'message': 'Invalid benchmark result payload'}), 400

    # Merge in stored metadata if fields are missing
    if CURRENT_AGENT_METADATA:
        if 'model' not in data and CURRENT_AGENT_METADATA.get('model'):
            data['model'] = CURRENT_AGENT_METADATA['model']
        if 'provider' not in data and CURRENT_AGENT_METADATA.get('provider'):
            data['provider'] = CURRENT_AGENT_METADATA['provider']
        if 'agent_framework' not in data and CURRENT_AGENT_METADATA.get('agent_framework'):
            data['agent_framework'] = CURRENT_AGENT_METADATA['agent_framework']
        if 'agentFramework' not in data:
            camel_framework = CURRENT_AGENT_METADATA.get('agentFramework') or CURRENT_AGENT_METADATA.get('agent_framework')
            if camel_framework:
                data['agentFramework'] = camel_framework

    # Add timestamp if not provided
    from datetime import datetime
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().isoformat()

    # In a real system, you would save this data to a database
    # For this example, we'll just print it to the console
    print(f"Benchmark results: {data}")

    # Determine output filename based on pattern
    now = datetime.now()
    
    # Extract metadata for filename substitution
    model_name = data.get('model', 'unknown').replace('/', '_').replace('\\', '_')
    provider = data.get('provider', 'unknown').replace('/', '_').replace('\\', '_')
    framework = data.get('agent_framework', data.get('agentFramework', 'unknown')).replace('/', '_').replace('\\', '_')
    timestamp_str = now.strftime('%Y%m%d_%H%M%S')
    date_str = now.strftime('%Y%m%d')
    
    # Replace placeholders in filename pattern
    filename = BENCHMARK_RESULTS_FILE_PATTERN.format(
        model=model_name,
        provider=provider,
        framework=framework,
        timestamp=timestamp_str,
        date=date_str
    )
    
    # Ensure filename ends with .json if no extension
    if not filename.endswith(('.json', '.jsonl', '.txt')):
        filename += '.json'
    
    # Store results to file
    try:
        with open(filename, 'a') as f:
            f.write(json.dumps(data) + '\n')
        print(f"Results saved to: {filename}")
    except Exception as e:
        print(f"Error writing to {filename}: {e}")
        # Fallback to default filename
        try:
            with open('benchmark_results.json', 'a') as f:
                f.write(json.dumps(data) + '\n')
            print("Results saved to fallback file: benchmark_results.json")
        except Exception as fallback_error:
            print(f"Error writing to fallback file: {fallback_error}")

    return jsonify({'status': 'success', 'filename': filename})

@app.route('/api/types', methods=['GET'])
def get_types():
    """Get available CAPTCHA types"""
    return jsonify({
        'types': get_captcha_types()
    })

@app.route('/api/puzzle_types', methods=['GET'])
def get_puzzle_types():
    """Get available CAPTCHA types (for eval scripts compatibility)"""
    return jsonify(get_captcha_types())

@app.route('/api/agent_metadata', methods=['POST'])
def update_agent_metadata():
    """Store the latest agent metadata so benchmark results can reference it."""
    payload = request.json or {}
    if not isinstance(payload, dict):
        return jsonify({'status': 'error', 'message': 'Invalid metadata payload'}), 400

    normalized: dict[str, str] = {}
    model_value = payload.get('model')
    provider_value = payload.get('provider')
    framework_value = payload.get('agent_framework') or payload.get('agentFramework')

    if isinstance(model_value, str) and model_value.strip():
        normalized['model'] = model_value.strip()
    if isinstance(provider_value, str) and provider_value.strip():
        normalized['provider'] = provider_value.strip()
    if isinstance(framework_value, str) and framework_value.strip():
        normalized['agent_framework'] = framework_value.strip()
        normalized['agentFramework'] = normalized['agent_framework']

    global CURRENT_AGENT_METADATA
    CURRENT_AGENT_METADATA.clear()
    CURRENT_AGENT_METADATA.update(normalized)

    return jsonify({'status': 'success', 'metadata': CURRENT_AGENT_METADATA})

if __name__ == '__main__':
    # For local development
    if os.environ.get('DEVELOPMENT'):
        app.run(debug=True, host='127.0.0.1', port=7860)
    else:
        # For production on Hugging Face Spaces
        app.run(host='127.0.0.1', port=7860) 
