# CAPTCHA Puzzle Benchmark

A comprehensive web-based benchmark platform for testing and evaluating various CAPTCHA puzzle types. This application provides an interactive interface for users to solve different types of visual CAPTCHA challenges, track performance, and benchmark accuracy.

## Features

- **Multiple CAPTCHA Types**: Support for various puzzle types including:
  - Dice Count
  - Shadow Plausible
  - Mirror
  - Squiggle (Memory-based)
  - Color Cipher
  - Spooky Circle (various variants)
  - Deformation
  - Red Dot
  - Adversarial puzzles

- **Real-time Statistics**: Track total puzzles solved, correct answers, and accuracy rate
- **Difficulty Ratings**: Each puzzle is rated with a difficulty level (1-5 stars)
- **Interactive UI**: Modern, responsive web interface with smooth animations
- **RESTful API**: JSON API for puzzle generation and answer submission

## Requirements

- Python 3.11 or higher (required for browser-use integration)
- See `requirements.txt` or `pyproject.toml` for dependencies

## Installation

### Using uv (Highly Recommended! Faster! Easier & more reproducible!)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the application
# export BENCHMARK_RESULTS_FILE="results_{model}_{timestamp}.json" (if you want to change result file's name)
uv run app.py

### Use uv to run code so you do not have to activate virtual env everytime.
# Test the Browser-Use framework default agents (Their in house model BU1.0)
uv run agent_frameworks/browseruse_cli.py --url http://127.0.0.1:7860 --llm browser-use 

# Test with other models (e.x. openai gpt5 here)
uv run agent_frameworks/browseruse_cli.py --url http://127.0.0.1:7860 --llm openai --model gpt-5-2025-08-07
```

### Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Usage

### Development Mode

```bash
export DEVELOPMENT=1
python app.py
```

The application will run in debug mode on `http://127.0.0.1:7860`

### Production Mode

```bash
# Using gunicorn (recommended for production)
gunicorn -w 4 -b 127.0.0.1:7860 app:app

# Or run directly
python app.py
```

The application will run on `http://127.0.0.1:7860`

Note: To accept connections from other machines, use `0.0.0.0` instead of `127.0.0.1` in the gunicorn command.

### Docker

```bash
# Build the image
docker build -t captcha-benchmark .

# Run the container
docker run -p 7860:7860 captcha-benchmark
```

## Project Structure

```
captchav2/
├── app.py                 # Main Flask application
├── main.py                # Alternative entry point
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Modern Python project configuration
├── Dockerfile            # Docker configuration
├── captcha_data/         # CAPTCHA puzzle data files
│   ├── Adversarial/
│   ├── Color_Cipher/
│   ├── Deformation/
│   ├── Dice_Count/
│   ├── Mirror/
│   ├── Red_Dot/
│   ├── Shadow_Plausible/
│   ├── Spooky_Circle/
│   ├── Spooky_Circle_Grid/
│   ├── Spooky_Circle_Grid_Direction/
│   └── Squiggle/
├── static/               # Static assets (CSS, JS)
│   ├── css/
│   └── js/
└── templates/            # Jinja2 templates
    └── index.html
```

## API Endpoints

- `GET /` - Main web interface
- `GET /api/get_puzzle` - Get a new puzzle
- `POST /api/check_answer` - Submit an answer for verification
- `GET /api/puzzle_types` - Get list of available puzzle types

## Configuration

The application supports various configuration options through environment variables and the puzzle data files. Each puzzle type has its own `ground_truth.json` file containing puzzle definitions and correct answers.

## License

[Add your license information here]

## Contributing

[Add contribution guidelines if applicable]

