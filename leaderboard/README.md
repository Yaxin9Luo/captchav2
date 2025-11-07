---
title: CAPTCHAv2 Leaderboard
emoji: 🏆
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# CAPTCHAv2 Leaderboard

A comprehensive leaderboard for comparing model performance across different CAPTCHA puzzle types. This interactive dashboard allows you to:

- 📊 View performance rankings across different CAPTCHA categories
- 📈 Compare models using interactive visualizations
- 💰 Analyze cost-effectiveness of different models
- 🔄 Upload and update results easily

## Features

- **Interactive Leaderboard Table**: Sortable rankings with color-coded performance indicators
- **Performance Comparison Charts**: Visual bar charts showing pass rates across models
- **Performance by Type**: Detailed breakdown of performance across different CAPTCHA puzzle types
- **Cost-Effectiveness Analysis**: Scatter plot comparing performance vs. cost
- **Easy Upload**: Support for CSV and JSON result files

## How to Use

1. **View the Leaderboard**: Browse the current rankings and filter by category
2. **Sort Results**: Sort by Pass Rate, Duration, or Cost
3. **Upload Results**: Use the upload section to add new evaluation results
4. **Compare Models**: Use the visualizations to compare different models

## Uploading Results

The leaderboard supports multiple file formats:

- **CSV files**: Aggregated results with columns for Model, Provider, Agent Framework, Type, metrics, and per-type pass rates
- **JSON files**: Single object or array of aggregated results
- **benchmark_results.json**: Per-puzzle results in JSONL format (auto-converted)

See the upload section in the app for detailed instructions and file format requirements.

## Categories

The leaderboard tracks performance across various CAPTCHA types including:
- Dice Count
- Color Cipher
- Color Counting
- Dynamic Jigsaw
- Mirror
- Set Game
- Shadow Plausible
- Spooky variants (Circle, Jigsaw, Shape Grid, Size, Text)
- Trajectory Recovery
- Transform Pipeline
- And more...

## Model Types

Models are automatically categorized as:
- **Proprietary**: Commercial models (OpenAI, Anthropic, Google, etc.)
- **Open source**: Open source models (Llama, Mistral, Qwen, etc.)

