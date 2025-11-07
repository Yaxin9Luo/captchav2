# leaderboard/app.py
import pandas as pd, numpy as np, matplotlib.pyplot as plt, gradio as gr
import pathlib
import json
import csv

CATEGORY_MAP = {
    "Overall": ["Overall Pass Rate"],
    # You can define sets, e.g. "Vision-hard": ["Squiggle", "Shadow_Plausible"]
}

def get_results_path():
    """Get the path to results.csv, resolving relative to this file's location."""
    this_file = pathlib.Path(__file__).resolve()
    results_path = this_file.parent / "results.csv"
    return results_path

def get_runs_path():
    """Get the path to runs directory, resolving relative to this file's location."""
    this_file = pathlib.Path(__file__).resolve()
    runs_path = this_file.parent / "runs"
    runs_path.mkdir(parents=True, exist_ok=True)
    return runs_path

def infer_type(row):
    """Infer model type (Proprietary/Open source) from Provider or Model name."""
    provider = str(row.get("Provider", "")).lower()
    model = str(row.get("Model", "")).lower()
    
    # Open source indicators
    open_source_keywords = [
        "llama", "mistral", "qwen", "phi", "gemma", "falcon", "mpt", 
        "vicuna", "alpaca", "wizard", "openchat", "neural-chat",
        "browser-use", "browseruse", "open source", "opensource"
    ]
    
    # Check if any open source keyword appears
    for keyword in open_source_keywords:
        if keyword in provider or keyword in model:
            return "Open source"
    
    # Default to Proprietary if not found
    return "Proprietary"

def load_df(path=None):
    """Load the results CSV, creating empty dataframe if file doesn't exist."""
    if path is None:
        path = get_results_path()
    
    metadata_cols = ["Model", "Provider", "Agent Framework", "Type"]
    metric_cols = ["Overall Pass Rate", "Avg Duration (s)", "Avg Cost ($)"]
    expected_cols = metadata_cols + metric_cols
    
    if not pathlib.Path(path).exists():
        # Return empty dataframe with expected columns
        return pd.DataFrame(columns=expected_cols)
    
    try:
        df = pd.read_csv(path)
        # Handle empty CSV (only headers)
        if len(df) == 0:
            return pd.DataFrame(columns=expected_cols)
        
        # Ensure required columns exist
        if "Agent Framework" not in df.columns:
            # Try legacy "Notes" column
            if "Notes" in df.columns:
                df["Agent Framework"] = df["Notes"]
            else:
                df["Agent Framework"] = ""
        
        # Handle legacy "Overall" column
        if "Overall" in df.columns and "Overall Pass Rate" not in df.columns:
            df["Overall Pass Rate"] = df["Overall"]
        
        # Add Type column if missing, infer from Provider/Model
        if "Type" not in df.columns:
            df["Type"] = df.apply(infer_type, axis=1)
        
        # Convert numeric columns
        numeric_cols = metric_cols + [c for c in df.columns if c not in metadata_cols + metric_cols]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        
        return df
    except Exception as e:
        print(f"Error loading results.csv: {e}")
        return pd.DataFrame(columns=expected_cols)

def compute_score(df, category):
    # Get columns to compute score from
    # Map "Overall" category to "Overall Pass Rate" column
    if category == "Overall":
        # Use CATEGORY_MAP which maps "Overall" to ["Overall Pass Rate"]
        cols = CATEGORY_MAP.get("Overall", ["Overall Pass Rate"])
    elif category in CATEGORY_MAP:
        # Use predefined category mapping
        cols = CATEGORY_MAP[category]
    elif category in df.columns:
        # Category is a direct column name
        cols = [category]
    else:
        # Fallback: use "Overall Pass Rate" if it exists, otherwise all numeric columns
        if "Overall Pass Rate" in df.columns:
            cols = ["Overall Pass Rate"]
        else:
            numeric_cols = [c for c in df.columns if c not in ["Model", "Provider", "Agent Framework", "Type"]]
            cols = numeric_cols if numeric_cols else []
    
    # Filter to only existing columns
    cols = [c for c in cols if c in df.columns]
    
    # If no valid columns found, use all numeric columns except metadata/metrics
    if not cols:
        exclude_cols = ["Model", "Provider", "Agent Framework", "Type", "Avg Duration (s)", "Avg Cost ($)"]
        numeric_cols = [c for c in df.columns if c not in exclude_cols]
        cols = numeric_cols if numeric_cols else []
        # If still no columns, create a zero score
        if not cols:
            df = df.copy()
            df["Category Pass Rate"] = 0.0
            return df
    
    df = df.copy()
    if cols:
        df["Category Pass Rate"] = df[cols].mean(axis=1, skipna=True)
    else:
        df["Category Pass Rate"] = 0.0
    return df

def table_html(df):
    if len(df) == 0:
        return """
        <style>
          .leaderboard-container {
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
            margin: 20px 0;
          }
          table.lb {
            width: 100%;
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-size: 14px;
          }
          table.lb thead {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            color: white;
          }
          table.lb th {
            padding: 16px 20px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
          }
          table.lb td {
            padding: 16px 20px;
            border-bottom: 1px solid #e5e7eb;
            color: #374151;
          }
          table.lb tbody tr {
            transition: background-color 0.2s ease;
          }
          table.lb tbody tr:hover {
            background: #f9fafb;
          }
          table.lb tbody tr:last-child td {
            border-bottom: none;
          }
          .rank-badge {
            display: inline-block;
            width: 32px;
            height: 32px;
            line-height: 32px;
            text-align: center;
            border-radius: 50%;
            font-weight: 700;
            font-size: 14px;
          }
          .rank-1 { background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); color: #000; box-shadow: 0 2px 8px rgba(255, 215, 0, 0.4); }
          .rank-2 { background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%); color: #000; box-shadow: 0 2px 8px rgba(192, 192, 192, 0.4); }
          .rank-3 { background: linear-gradient(135deg, #cd7f32 0%, #e6a55d 100%); color: #fff; box-shadow: 0 2px 8px rgba(205, 127, 50, 0.4); }
          .rank-other { background: #f1f5f9; color: #64748b; }
          .pass-rate-cell {
            font-weight: 600;
            font-size: 15px;
          }
          .metric-cell {
            font-weight: 500;
            font-size: 14px;
            color: #6b7280;
          }
        </style>
        <div class="leaderboard-container">
          <table class="lb">
            <thead><tr><th>#</th><th>Model</th><th>Provider</th><th>Type</th><th>Agent Framework</th><th>Pass Rate</th><th>Avg Duration (s)</th><th>Avg Cost ($)</th></tr></thead>
            <tbody><tr><td colspan="8" style="text-align:center;padding:40px;color:#9ca3af;font-size:16px;">No results yet. Run evaluations to populate the leaderboard.</td></tr></tbody>
          </table>
        </div>
        """
    rows = []
    for i, r in df.iterrows():
        rank = i + 1
        rank_class = "rank-1" if rank == 1 else "rank-2" if rank == 2 else "rank-3" if rank == 3 else "rank-other"
        pass_rate = r['Category Pass Rate']
        pass_rate_color = "#10b981" if pass_rate >= 0.7 else "#f59e0b" if pass_rate >= 0.4 else "#ef4444"
        
        # Format duration and cost
        duration = r.get('Avg Duration (s)', None)
        duration_str = f"{duration:.2f}" if pd.notna(duration) and duration is not None else "N/A"
        
        cost = r.get('Avg Cost ($)', None)
        cost_str = f"${cost:.4f}" if pd.notna(cost) and cost is not None else "N/A"
        
        type_val = r.get('Type', 'Proprietary')
        type_color = "#10b981" if type_val == "Open source" else "#6366f1"
        
        rows.append(f"""
        <tr>
          <td><span class="rank-badge {rank_class}">{rank}</span></td>
          <td><strong style="color: #111827;">{r['Model']}</strong></td>
          <td style="color: #6b7280;">{r.get('Provider','')}</td>
          <td><span style="color: {type_color}; font-weight: 600;">{type_val}</span></td>
          <td style="color: #6b7280;">{r.get('Agent Framework','')}</td>
          <td class="pass-rate-cell" style="color: {pass_rate_color};">{pass_rate:.3f}</td>
          <td class="metric-cell">{duration_str}</td>
          <td class="metric-cell">{cost_str}</td>
        </tr>""")
    return f"""
    <style>
      .leaderboard-container {{
        background: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        overflow: hidden;
        margin: 20px 0;
      }}
      table.lb {{
        width: 100%;
        border-collapse: collapse;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        font-size: 14px;
      }}
          table.lb thead {{
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
            color: white;
          }}
      table.lb th {{
        padding: 16px 20px;
        text-align: left;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }}
      table.lb td {{
        padding: 16px 20px;
        border-bottom: 1px solid #e5e7eb;
        color: #374151;
      }}
      table.lb tbody tr {{
        transition: background-color 0.2s ease;
      }}
      table.lb tbody tr:hover {{
        background: #f9fafb;
      }}
      table.lb tbody tr:last-child td {{
        border-bottom: none;
      }}
      .rank-badge {{
        display: inline-block;
        width: 32px;
        height: 32px;
        line-height: 32px;
        text-align: center;
        border-radius: 50%;
        font-weight: 700;
        font-size: 14px;
      }}
      .rank-1 {{ background: linear-gradient(135deg, #ffd700 0%, #ffed4e 100%); color: #000; box-shadow: 0 2px 8px rgba(255, 215, 0, 0.4); }}
      .rank-2 {{ background: linear-gradient(135deg, #c0c0c0 0%, #e8e8e8 100%); color: #000; box-shadow: 0 2px 8px rgba(192, 192, 192, 0.4); }}
      .rank-3 {{ background: linear-gradient(135deg, #cd7f32 0%, #e6a55d 100%); color: #fff; box-shadow: 0 2px 8px rgba(205, 127, 50, 0.4); }}
      .rank-other {{ background: #f1f5f9; color: #64748b; }}
      .pass-rate-cell {{
        font-weight: 600;
        font-size: 15px;
      }}
      .metric-cell {{
        font-weight: 500;
        font-size: 14px;
        color: #6b7280;
      }}
    </style>
    <div class="leaderboard-container">
      <table class="lb">
        <thead><tr><th>#</th><th>Model</th><th>Provider</th><th>Type</th><th>Agent Framework</th><th>Pass Rate</th><th>Avg Duration (s)</th><th>Avg Cost ($)</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """

def perf_bar(df):
    plt.close("all")
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='white', dpi=150)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.tight_layout(); return fig
    d = df.sort_values("Category Pass Rate", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5*len(d))), facecolor='white', dpi=150)
    
    # Create gradient colors based on pass rate - CAPTCHA themed
    colors = []
    for pass_rate in d["Category Pass Rate"]:
        if pass_rate >= 0.7:
            colors.append('#10b981')  # verification green
        elif pass_rate >= 0.4:
            colors.append('#f59e0b')  # warning amber
        else:
            colors.append('#ef4444')  # error red
    
    bars = ax.barh(range(len(d)), d["Category Pass Rate"], color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, pass_rate) in enumerate(zip(bars, d["Category Pass Rate"])):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{pass_rate:.3f}', ha='left', va='center', fontsize=11, fontweight='600')
    
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["Model"], fontsize=12)
    ax.set_xlabel("Pass Rate", fontsize=12, fontweight='600', color='#374151')
    ax.set_xlim(0, 1.1)
    ax.set_title("Performance Comparison", fontsize=16, fontweight='700', color='#111827', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    fig.tight_layout()
    return fig

def perf_by_type(df_full, model_filter="Models Avg"):
    """
    Show average performance by puzzle type.
    
    Args:
        df_full: Full dataframe with all models
        model_filter: "Models Avg" for average across all models, or a specific model name
    """
    plt.close("all")
    
    # Filter by model if specified
    if model_filter and model_filter != "Models Avg":
        df_filtered = df_full[df_full["Model"] == model_filter].copy()
        if len(df_filtered) == 0:
            fig, ax = plt.subplots(figsize=(12, 5), facecolor='white', dpi=150)
            ax.text(0.5, 0.5, f"No data available for model: {model_filter}", ha="center", va="center", fontsize=14, color="gray")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
            fig.tight_layout(); return fig
        df_plot = df_filtered
        plot_title = f"Performance by Type - {model_filter}"
    else:
        df_plot = df_full
        plot_title = "Average Performance by CAPTCHA Type (All Models)"
    
    if len(df_plot) == 0:
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='white', dpi=150)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.tight_layout(); return fig
    
    # Average each per-type column across models (exclude metadata and metric columns)
    exclude_cols = ["Model", "Provider", "Agent Framework", "Type", "Overall Pass Rate", "Avg Duration (s)", "Avg Cost ($)", "Category Pass Rate"]
    numeric_cols = [c for c in df_plot.columns if c not in exclude_cols]
    type_cols = [c for c in numeric_cols if df_plot[c].notna().any() and df_plot[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    if len(type_cols) == 0:
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='white', dpi=150)
        ax.text(0.5, 0.5, "No per-type data available", ha="center", va="center", fontsize=14, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.tight_layout(); return fig
    
    # Calculate means, handling NaN values properly
    if model_filter == "Models Avg":
        # Average across all models
        means = df_plot[type_cols].mean(numeric_only=True)
    else:
        # For a single model, just get its values (should be one row)
        if len(df_plot) == 1:
            means = df_plot[type_cols].iloc[0]
        else:
            # If multiple rows (shouldn't happen), average them
            means = df_plot[type_cols].mean(numeric_only=True)
    
    # Filter out any NaN means
    means = means.dropna()
    
    if len(means) == 0:
        fig, ax = plt.subplots(figsize=(12, 5), facecolor='white', dpi=150)
        ax.text(0.5, 0.5, "No valid per-type data available", ha="center", va="center", fontsize=14, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.tight_layout(); return fig
    
    fig, ax = plt.subplots(figsize=(max(12, len(means) * 0.8), 6), facecolor='white', dpi=150)
    
    # Create gradient colors based on performance - CAPTCHA themed
    colors = []
    for val in means.values:
        if pd.isna(val):
            colors.append('#94a3b8')  # slate gray for NaN
        elif val >= 0.7:
            colors.append('#10b981')  # verification green
        elif val >= 0.4:
            colors.append('#f59e0b')  # warning amber
        else:
            colors.append('#ef4444')  # error red
    
    bars = ax.bar(range(len(means)), means.values, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, means.values):
        if not pd.isna(val):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='600')
    
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, max(1.1, means.max() * 1.1) if not means.empty else 1.1)
    ax.set_ylabel("Average Pass Rate", fontsize=12, fontweight='600', color='#374151')
    ax.set_title(plot_title, fontsize=16, fontweight='700', color='#111827', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_facecolor('#fafafa')
    fig.tight_layout()
    return fig

def cost_effectiveness_plot(df):
    """
    Create a cost-effectiveness scatter plot: Performance (X) vs Cost (Y).
    Color-coded by Type (Proprietary vs Open source).
    """
    plt.close("all")
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white', dpi=150)
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", fontsize=14, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.tight_layout(); return fig
    
    # Filter to rows with valid performance and cost data
    df_plot = df.copy()
    df_plot = df_plot[df_plot['Category Pass Rate'].notna() & df_plot['Avg Cost ($)'].notna()]
    
    if len(df_plot) == 0:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white', dpi=150)
        ax.text(0.5, 0.5, "No data with both performance and cost metrics", ha="center", va="center", fontsize=14, color="gray")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        fig.tight_layout(); return fig
    
    # Create figure with higher DPI for better resolution
    fig, ax = plt.subplots(figsize=(14, 9), facecolor='white', dpi=150)
    
    # Separate by type
    proprietary = df_plot[df_plot.get('Type', 'Proprietary') == 'Proprietary']
    open_source = df_plot[df_plot.get('Type', 'Proprietary') == 'Open source']
    
    # Plot points
    if len(proprietary) > 0:
        ax.scatter(proprietary['Category Pass Rate'], proprietary['Avg Cost ($)'], 
                  c='#6366f1', s=200, alpha=0.75, edgecolors='white', linewidth=2.5, 
                  label='Proprietary', zorder=3)
        # Add labels for proprietary models
        for idx, row in proprietary.iterrows():
            ax.annotate(row['Model'], 
                       (row['Category Pass Rate'], row['Avg Cost ($)']),
                       fontsize=10, alpha=0.85, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    if len(open_source) > 0:
        ax.scatter(open_source['Category Pass Rate'], open_source['Avg Cost ($)'], 
                  c='#10b981', s=200, alpha=0.75, edgecolors='white', linewidth=2.5, 
                  label='Open source', zorder=3)
        # Add labels for open source models
        for idx, row in open_source.iterrows():
            ax.annotate(row['Model'], 
                       (row['Category Pass Rate'], row['Avg Cost ($)']),
                       fontsize=10, alpha=0.85, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Calculate thresholds for quadrants (median or fixed thresholds)
    perf_threshold = df_plot['Category Pass Rate'].median() if len(df_plot) > 1 else 0.4
    cost_threshold = df_plot['Avg Cost ($)'].median() if len(df_plot) > 1 else 0.01
    
    # Add quadrant lines
    ax.axvline(x=perf_threshold, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    ax.axhline(y=cost_threshold, color='gray', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
    
    # Add quadrant annotations
    x_range = df_plot['Category Pass Rate'].max() - df_plot['Category Pass Rate'].min()
    y_range = df_plot['Avg Cost ($)'].max() - df_plot['Avg Cost ($)'].min()
    
    # Top-left: Low Performance, High Cost
    ax.text(df_plot['Category Pass Rate'].min() + x_range * 0.05, 
            df_plot['Avg Cost ($)'].max() - y_range * 0.05,
            '▲ Low Performance\nHigh Cost', 
            fontsize=12, color='#ef4444', weight='bold', 
            ha='left', va='top', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#ef4444', linewidth=1.5))
    
    # Bottom-right: High Performance, Low Cost
    ax.text(df_plot['Category Pass Rate'].max() - x_range * 0.05, 
            df_plot['Avg Cost ($)'].min() + y_range * 0.05,
            '▼ High Performance\nLow Cost', 
            fontsize=12, color='#10b981', weight='bold', 
            ha='right', va='bottom', alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#10b981', linewidth=1.5))
    
    # Styling
    ax.set_xlabel("Performance (Pass Rate)", fontsize=14, fontweight='600', color='#374151')
    ax.set_ylabel("Avg Cost ($)", fontsize=14, fontweight='600', color='#374151')
    ax.set_title("Cost-Effectiveness Analysis", fontsize=17, fontweight='700', color='#111827', pad=25)
    
    # Add padding to axes (more padding on right for legend space)
    x_pad = x_range * 0.15 if x_range > 0 else 0.1
    y_pad = y_range * 0.15 if y_range > 0 else 0.001
    ax.set_xlim(df_plot['Category Pass Rate'].min() - x_pad * 0.5, df_plot['Category Pass Rate'].max() + x_pad)
    ax.set_ylim(max(0, df_plot['Avg Cost ($)'].min() - y_pad * 0.5), df_plot['Avg Cost ($)'].max() + y_pad)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.grid(alpha=0.3, linestyle='--', zorder=0, linewidth=1)
    ax.set_facecolor('#fafafa')
    
    # Add legend - position it outside the plot area to avoid covering data
    # Use bbox_to_anchor to place it outside the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, 
              fancybox=True, shadow=True, fontsize=12, framealpha=0.95,
              edgecolor='#e5e7eb', facecolor='white')
    
    # Adjust layout to make room for legend
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    return fig

def convert_benchmark_results_json(file_path, model_name=None, provider=None, agent_framework=None):
    """
    Convert benchmark_results.json format (per-puzzle results) to aggregated format.
    
    Args:
        file_path: Path to benchmark_results.json file (Path object or string)
        model_name: Model name (if None, will try to infer from filename or use "Unknown")
        provider: Provider name (if None, will try to infer from model_name)
        agent_framework: Agent framework name (if None, will use "browser-use" as default)
    
    Returns:
        dict: Aggregated record with Model, Provider, Agent Framework, Type, metrics, and per-type pass rates
    """
    # Convert to Path object if needed
    file_path = pathlib.Path(file_path) if not isinstance(file_path, pathlib.Path) else file_path
    
    # Read the file - it's a JSONL file (one JSON object per line)
    puzzle_results = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    puzzle_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not puzzle_results:
        raise ValueError("No valid puzzle results found in file")
    
    # Try to extract model/provider from puzzle results first (if they were included)
    extracted_model = None
    extracted_provider = None
    extracted_agent_framework = None
    
    for result in puzzle_results[:10]:  # Check first 10 results
        if 'model' in result and result['model']:
            extracted_model = result['model']
        if 'provider' in result and result['provider']:
            extracted_provider = result['provider']
        if 'agent_framework' in result and result['agent_framework']:
            extracted_agent_framework = result['agent_framework']
        # Also check camelCase variants
        if 'agentFramework' in result and result['agentFramework']:
            extracted_agent_framework = result['agentFramework']
    
    # Use extracted values if available, otherwise use provided parameters
    if model_name is None:
        model_name = extracted_model
    
    if provider is None:
        provider = extracted_provider
    
    if agent_framework is None:
        agent_framework = extracted_agent_framework
    
    # Infer model/provider if still not available
    if model_name is None:
        # Try to infer from filename (e.g., "gpt-4_results.json" -> "gpt-4")
        filename = file_path.stem.lower()
        if 'benchmark_results' in filename:
            model_name = "Unknown Model"
        else:
            # Try to extract model name from filename
            model_name = filename.replace('_results', '').replace('_benchmark', '').replace('-', ' ').title()
    
    if provider is None:
        # Try to infer provider from model name
        model_lower = model_name.lower()
        if any(x in model_lower for x in ['gpt', 'openai']):
            provider = "OpenAI"
        elif any(x in model_lower for x in ['claude', 'anthropic']):
            provider = "Anthropic"
        elif any(x in model_lower for x in ['gemini', 'google']):
            provider = "Google"
        elif any(x in model_lower for x in ['llama', 'mistral', 'qwen', 'phi', 'gemma']):
            provider = "Open Source"
        else:
            provider = "Unknown"
    
    if agent_framework is None:
        agent_framework = "browser-use"  # Default assumption
    
    # Aggregate results
    # Group by puzzle_type
    puzzle_type_stats = {}
    total_correct = 0
    total_attempts = len(puzzle_results)
    total_duration = 0.0
    total_cost = 0.0
    cost_count = 0
    
    for result in puzzle_results:
        puzzle_type = result.get('puzzle_type', 'Unknown')
        
        # Initialize puzzle type stats if needed
        if puzzle_type not in puzzle_type_stats:
            puzzle_type_stats[puzzle_type] = {'correct': 0, 'total': 0}
        
        puzzle_type_stats[puzzle_type]['total'] += 1
        if result.get('correct', False):
            puzzle_type_stats[puzzle_type]['correct'] += 1
            total_correct += 1
        
        # Aggregate duration
        elapsed_time = result.get('elapsed_time')
        if elapsed_time is not None:
            try:
                total_duration += float(elapsed_time)
            except (ValueError, TypeError):
                pass
        
        # Aggregate cost
        cost = result.get('cost')
        if cost is not None:
            try:
                total_cost += float(cost)
                cost_count += 1
            except (ValueError, TypeError):
                pass
    
    # Calculate overall pass rate
    overall_pass_rate = total_correct / total_attempts if total_attempts > 0 else 0.0
    
    # Calculate average duration
    avg_duration = total_duration / total_attempts if total_attempts > 0 else None
    
    # Calculate average cost
    avg_cost = total_cost / cost_count if cost_count > 0 else None
    
    # Build aggregated record
    record = {
        "Model": model_name,
        "Provider": provider,
        "Agent Framework": agent_framework,
        "Overall Pass Rate": overall_pass_rate,
        "Avg Duration (s)": avg_duration,
        "Avg Cost ($)": avg_cost,
    }
    
    # Add per-type pass rates
    for puzzle_type, stats in puzzle_type_stats.items():
        pass_rate = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        record[puzzle_type] = pass_rate
    
    # Infer Type
    record["Type"] = infer_type(record)
    
    return record

def is_benchmark_results_format(data):
    """
    Check if the data is in benchmark_results.json format (per-puzzle results).
    
    Args:
        data: List of dictionaries or single dictionary
    
    Returns:
        bool: True if data appears to be in benchmark_results format
    """
    if isinstance(data, dict):
        data = [data]
    
    if not isinstance(data, list) or len(data) == 0:
        return False
    
    # Check if first record has benchmark_results.json structure
    first = data[0]
    required_fields = ['puzzle_type', 'puzzle_id', 'correct']
    has_required = all(field in first for field in required_fields)
    
    # Check if it's NOT the aggregated format (which would have Model, Provider, etc.)
    aggregated_fields = ['Model', 'Provider', 'Overall Pass Rate']
    is_not_aggregated = not any(field in first for field in aggregated_fields)
    
    return has_required and is_not_aggregated

def process_uploaded_file(file, model_name=None, provider=None, agent_framework=None):
    """
    Process an uploaded CSV or JSON file and merge with existing results.
    
    Args:
        file: File path string (from Gradio File component with type="filepath")
        model_name: Optional model name (for benchmark_results.json conversion)
        provider: Optional provider name (for benchmark_results.json conversion)
        agent_framework: Optional agent framework name (for benchmark_results.json conversion)
    
    Returns:
        tuple: (success_message, error_message)
    """
    if file is None:
        return None, "No file uploaded"
    
    try:
        # Gradio returns a file path string when type="filepath"
        file_path = pathlib.Path(file) if isinstance(file, str) else pathlib.Path(file.name)
        
        # Read the file based on extension
        if file_path.suffix.lower() == '.json':
            # Try reading as JSONL first (benchmark_results.json format)
            try:
                # Read first few lines to detect format
                with open(file_path, 'r') as f:
                    first_lines = [f.readline().strip() for _ in range(5)]
                    f.seek(0)
                    
                    # Try to parse as JSONL (one JSON object per line)
                    puzzle_results = []
                    for line in first_lines:
                        if line:
                            try:
                                puzzle_results.append(json.loads(line))
                            except json.JSONDecodeError:
                                break
                    
                    # Check if it's benchmark_results format
                    if puzzle_results and is_benchmark_results_format(puzzle_results):
                        # Read entire file as JSONL
                        puzzle_results = []
                        with open(file_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        puzzle_results.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        continue
                        
                        # Convert to aggregated format
                        record = convert_benchmark_results_json(
                            file_path, 
                            model_name=model_name,
                            provider=provider,
                            agent_framework=agent_framework
                        )
                        records = [record]
                    else:
                        # Try reading as regular JSON
                        f.seek(0)
                        data = json.load(f)
                        
                        # Normalize to list of records
                        if isinstance(data, dict):
                            records = [data]
                        elif isinstance(data, list):
                            records = data
                        else:
                            return None, f"Invalid JSON format: expected object or array, got {type(data).__name__}"
                        
                        # Check if it's benchmark_results format
                        if is_benchmark_results_format(records):
                            # Convert to aggregated format
                            record = convert_benchmark_results_json(
                                file_path,
                                model_name=model_name,
                                provider=provider,
                                agent_framework=agent_framework
                            )
                            records = [record]
            except Exception as e:
                # Fallback: try reading as regular JSON
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Normalize to list of records
                    if isinstance(data, dict):
                        records = [data]
                    elif isinstance(data, list):
                        records = data
                    else:
                        return None, f"Invalid JSON format: expected object or array, got {type(data).__name__}"
                    
                    # Check if it's benchmark_results format
                    if is_benchmark_results_format(records):
                        # Convert to aggregated format
                        record = convert_benchmark_results_json(
                            file_path,
                            model_name=model_name,
                            provider=provider,
                            agent_framework=agent_framework
                        )
                        records = [record]
                except Exception as json_err:
                    return None, f"Error reading JSON file: {str(json_err)}"
            
            # Handle legacy column names
            legacy_map = {"Notes": "Agent Framework", "Overall": "Overall Pass Rate"}
            for record in records:
                for old_key, new_key in legacy_map.items():
                    if old_key in record and new_key not in record:
                        record[new_key] = record.pop(old_key)
                
                # Infer Type if not present
                if "Type" not in record:
                    record["Type"] = infer_type(record)
            
            # Save individual JSON files to runs directory for aggregation
            runs_path = get_runs_path()
            import time
            for record in records:
                run_file = runs_path / f"run_{int(time.time() * 1000)}.json"
                with open(run_file, 'w') as f:
                    json.dump(record, f, indent=2)
            
            num_records = len(records)
            
        elif file_path.suffix.lower() == '.csv':
            # Handle CSV file
            df_uploaded = pd.read_csv(file_path)
            
            # Handle legacy column names
            if "Notes" in df_uploaded.columns and "Agent Framework" not in df_uploaded.columns:
                df_uploaded["Agent Framework"] = df_uploaded["Notes"]
            if "Overall" in df_uploaded.columns and "Overall Pass Rate" not in df_uploaded.columns:
                df_uploaded["Overall Pass Rate"] = df_uploaded["Overall"]
            
            # Add Type column if missing
            if "Type" not in df_uploaded.columns:
                df_uploaded["Type"] = df_uploaded.apply(infer_type, axis=1)
            
            # Convert to records and save as JSON files (for consistency with aggregation script)
            records = df_uploaded.to_dict('records')
            runs_path = get_runs_path()
            import time
            for record in records:
                run_file = runs_path / f"run_{int(time.time() * 1000)}.json"
                with open(run_file, 'w') as f:
                    json.dump(record, f, indent=2)
            
            num_records = len(records)
            
        else:
            return None, f"Unsupported file type: {file_path.suffix}. Please upload a .csv or .json file."
        
        # Aggregate runs into results.csv
        aggregate_runs_to_csv()
        
        return f"✅ Successfully uploaded {num_records} record(s). Leaderboard updated!", None
        
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON file: {str(e)}"
    except pd.errors.EmptyDataError:
        return None, "CSV file is empty"
    except Exception as e:
        return None, f"Error processing file: {str(e)}"

def aggregate_runs_to_csv():
    """
    Aggregate all JSON files in runs/ directory into results.csv.
    This consolidates all uploaded evaluation results into a single CSV file.
    Deduplicates records based on (Model, Provider, Agent Framework) combination,
    keeping the most recent entry for each unique combination.
    """
    runs_path = get_runs_path()
    results_path = get_results_path()
    
    # Gather all JSON files with their modification times
    records_with_time = []
    for path in runs_path.glob("*.json"):
        try:
            record = json.loads(path.read_text())
            # Store modification time for deduplication (most recent wins)
            mtime = path.stat().st_mtime
            records_with_time.append((mtime, record))
        except Exception as e:
            print(f"Warning: Skipping invalid JSON file {path}: {e}")
    
    if not records_with_time:
        # Create empty CSV with headers
        fixed_metadata = ["Model", "Provider", "Agent Framework", "Type"]
        fixed_metrics = ["Overall Pass Rate", "Avg Duration (s)", "Avg Cost ($)"]
        with results_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fixed_metadata + fixed_metrics)
            w.writeheader()
        return
    
    # Sort by modification time (most recent first)
    records_with_time.sort(key=lambda x: x[0], reverse=True)
    
    # Handle legacy column names and infer Type
    legacy_map = {"Notes": "Agent Framework", "Overall": "Overall Pass Rate"}
    processed_records = []
    for mtime, record in records_with_time:
        for old_key, new_key in legacy_map.items():
            if old_key in record and new_key not in record:
                record[new_key] = record.pop(old_key)
        
        # Infer Type if not present
        if "Type" not in record:
            record["Type"] = infer_type(record)
        
        processed_records.append(record)
    
    # Deduplicate: keep only the most recent record for each (Model, Provider, Agent Framework) combination
    seen = {}
    deduplicated_records = []
    
    for record in processed_records:
        # Create unique key from Model, Provider, and Agent Framework
        model = str(record.get("Model", "")).strip()
        provider = str(record.get("Provider", "")).strip()
        agent_framework = str(record.get("Agent Framework", "")).strip()
        unique_key = (model, provider, agent_framework)
        
        # Only add if we haven't seen this combination before
        # Since records are sorted by time (most recent first), the first occurrence is kept
        if unique_key not in seen:
            seen[unique_key] = True
            deduplicated_records.append(record)
    
    if not deduplicated_records:
        # Create empty CSV with headers
        fixed_metadata = ["Model", "Provider", "Agent Framework", "Type"]
        fixed_metrics = ["Overall Pass Rate", "Avg Duration (s)", "Avg Cost ($)"]
        with results_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fixed_metadata + fixed_metrics)
            w.writeheader()
        return
    
    # Build header: metadata → metrics → puzzle types
    fixed_metadata = ["Model", "Provider", "Agent Framework", "Type"]
    fixed_metrics = ["Overall Pass Rate", "Avg Duration (s)", "Avg Cost ($)"]
    puzzle_types = sorted({k for r in deduplicated_records for k in r.keys() 
                          if k not in fixed_metadata + fixed_metrics})
    header = fixed_metadata + fixed_metrics + puzzle_types
    
    # Write CSV
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in deduplicated_records:
            w.writerow(r)

def render(category, sort_column, sort_direction, model_filter="Models Avg"):
    df_full = load_df()  # Keep full dataset for perf_by_type
    df = df_full.copy()
    
    df = compute_score(df, category)
    
    # Determine sort column and direction
    ascending = (sort_direction == "Low→High")
    
    # Map sort column names to actual column names (only numeric/metric columns)
    sort_column_map = {
        "Pass Rate": "Category Pass Rate",
        "Avg Duration (s)": "Avg Duration (s)",
        "Avg Cost ($)": "Avg Cost ($)"
    }
    
    actual_sort_column = sort_column_map.get(sort_column, "Category Pass Rate")
    
    # Check if column exists
    if actual_sort_column not in df.columns:
        actual_sort_column = "Category Pass Rate"
    
    # Handle NaN values for numeric sorting
    df = df.copy()
    df['_sort_helper'] = df[actual_sort_column].fillna(float('inf') if ascending else float('-inf'))
    df = df.sort_values('_sort_helper', ascending=ascending).drop(columns=['_sort_helper'])
    df = df.reset_index(drop=True)
    
    # perf_by_type uses full dataset to show all puzzle types, with optional model filter
    # cost_effectiveness_plot needs df with Category Pass Rate computed
    return table_html(df), perf_bar(df), perf_by_type(df_full, model_filter), cost_effectiveness_plot(df)

def app():
    df = load_df()
    
    cats = ["Overall"]
    if len(df) > 0:
        # Get all puzzle type columns (exclude metadata and metric columns)
        exclude_cols = ["Model", "Provider", "Agent Framework", "Type", "Overall Pass Rate", "Avg Duration (s)", "Avg Cost ($)"]
        puzzle_cols = [c for c in df.columns if c not in exclude_cols]
        cats = ["Overall"] + puzzle_cols

    with gr.Blocks(title="CAPTCHAv2 Leaderboard", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
        gr.Markdown("""
        <div style="text-align: center; padding: 30px 0;">
          <h1 style="font-size: 42px; font-weight: 700; margin: 0; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            CAPTCHAv2 Leaderboard
          </h1>
          <p style="font-size: 16px; color: #64748b; margin-top: 10px;">
            Compare model performance across different CAPTCHA types
          </p>
        </div>
        """)
        
        # Upload section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Results")
                
                # Main accordion for the entire guide
                with gr.Accordion("📖 Step-by-Step Guide to Submit Results", open=False):
                    # Step 1: Run Evaluation Protocol
                    with gr.Accordion("Step 1: Run the Evaluation Protocol", open=False):
                        gr.Markdown("""
                        **Option A: Using browser-use Agent Framework**
                        
                        1. Start the CAPTCHA server:
                           ```bash
                           python app.py
                           ```
                           The server will run on `http://127.0.0.1:7860`
                        
                        2. Run the browser-use agent evaluation (default is their in house model BU1.0):
                           ```bash
                           python -m agent_frameworks.browseruse_cli \\
                             --url http://127.0.0.1:7860 \\
                             --llm browser-use \\
                           ```
                           Or with a different LLM:
                           ```bash
                           python -m agent_frameworks.browseruse_cli \\
                             --url http://127.0.0.1:7860 \\
                             --llm openai \\
                             --model gpt-4o 
                           ```
                        
                        3. The evaluation will automatically save results to `benchmark_results.json` in the project root.
                           Each puzzle attempt is logged as a JSON object with fields:
                           - `puzzle_type`, `puzzle_id`, `user_answer`, `correct_answer`, `correct`
                           - `elapsed_time`, `timestamp`
                           - `model`, `provider`, `agent_framework` 
                        
                        **Option B: Using Other Agent Frameworks**
                        
                        Follow your framework's evaluation protocol. Ensure results are saved in `benchmark_results.json` format
                        (JSONL: one JSON object per line) with the same field structure.
                        """)
                    
                    # Step 2: Convert Results
                    with gr.Accordion("Step 2: Convert Results to CSV Format", open=False):
                        gr.Markdown("""
                        **Method 1:  Convert to CSV Format (Recommended)**
                        
                        Use the provided conversion script (`convert_benchmark_to_csv.py` in the project root):
                        ```bash
                        python convert_benchmark_to_csv.py benchmark_results.json leaderboard/results.csv
                        ```
                        
                        **Method 2: Directly Upload to Leaderboard (Auto-conversion)**
                        
                        You can upload `benchmark_results.json` directly here. The system will automatically handle all.
                        
                        Optionally provide metadata below if auto-detection fails:
                        - Model Name (e.g., "gpt-4", "claude-3-sonnet", "bu-1-0")
                        - Provider (e.g., "OpenAI", "Anthropic", "browser-use")
                        - Agent Framework (e.g., "browser-use", "crewai")
                        """)
                    
                    # Step 3: Upload Results
                    with gr.Accordion("Step 3: Upload Results", open=False):
                        gr.Markdown("""
                        **Supported file formats:**
                        - ✅ `benchmark_results.json` - Per-puzzle results (JSONL format) 
                        - ✅ `results.csv` - Aggregated results  **Recommended**
                        - ✅ JSON files - Single object or array of aggregated results
                        
                        **File format requirements:**
                        
                        For `benchmark_results.json` (per-puzzle format):
                        ```json
                        {"puzzle_type": "Dice_Count", "puzzle_id": "dice1.png", "user_answer": "24", "correct_answer": 24, "correct": true, "elapsed_time": "12.5", "timestamp": "2025-01-01T00:00:00Z", "model": "bu-1-0", "provider": "browser-use", "agent_framework": "browser-use"}
                        ```
                        
                        For CSV (aggregated format):
                        - Required columns: `Model`, `Provider`, `Agent Framework`, `Type`, `Overall Pass Rate` , `Avg Duration (s)`, `Avg Cost ($)`, and puzzle type columns (e.g., `Dice_Count`, `Mirror`, etc.)                
                        """)
                
                file_upload = gr.File(
                    label="Upload Results File",
                    file_types=[".csv", ".json"],
                    type="filepath"
                )
                with gr.Row():
                    model_name_input = gr.Textbox(
                        label="Model Name (optional, for benchmark_results.json)",
                        placeholder="e.g., gpt-4, claude-3-sonnet",
                        container=True
                    )
                    provider_input = gr.Textbox(
                        label="Provider (optional, for benchmark_results.json)",
                        placeholder="e.g., OpenAI, Anthropic, Google",
                        container=True
                    )
                    agent_framework_input = gr.Textbox(
                        label="Agent Framework (optional, for benchmark_results.json)",
                        placeholder="e.g., browser-use, crewai",
                        value="browser-use",
                        container=True
                    )
                upload_btn = gr.Button("Upload & Update Leaderboard", variant="primary")
                upload_status = gr.Markdown("")
        
        gr.Markdown("---")
        
        with gr.Row():
            cat = gr.Dropdown(choices=cats, value="Overall", label="Category/Type", container=True)
            sort_col = gr.Dropdown(
                choices=["Pass Rate", "Avg Duration (s)", "Avg Cost ($)"],
                value="Pass Rate",
                label="Sort by",
                container=True
            )
            sort_dir = gr.Radio(
                choices=["High→Low", "Low→High"],
                value="High→Low",
                label="Sort Direction",
                container=True
            )
        
        # Model filter for Performance by Type plot
        model_choices = ["Models Avg"]
        if len(df) > 0 and "Model" in df.columns:
            model_choices.extend(sorted(df["Model"].unique().tolist()))
        
        with gr.Row():
            model_filter = gr.Dropdown(
                choices=model_choices,
                value="Models Avg",
                label="Model Filter (for Performance by Type plot)",
                container=True
            )
        
        out = gr.HTML(elem_classes="leaderboard-table")
        bar = gr.Plot(label="Performance Comparison")
        pertype_plot = gr.Plot(label="Performance by Type")
        cost_eff_plot = gr.Plot(label="Cost-Effectiveness Analysis")

        def handle_upload(file, model_filter_val, model_name_input_val, provider_input_val, agent_framework_input_val):
            if file is None:
                # Return current state if no file
                table, bar_fig, pertype_fig, cost_fig = render("Overall", "Pass Rate", "High→Low", model_filter_val or "Models Avg")
                return "Please select a file to upload.", table, bar_fig, pertype_fig, cost_fig
            
            # Use provided metadata or None (which will trigger auto-detection)
            model_name_val = model_name_input_val.strip() if model_name_input_val else None
            provider_val = provider_input_val.strip() if provider_input_val else None
            agent_framework_val = agent_framework_input_val.strip() if agent_framework_input_val else None
            
            success_msg, error_msg = process_uploaded_file(
                file,
                model_name=model_name_val,
                provider=provider_val,
                agent_framework=agent_framework_val
            )
            if error_msg:
                # Return current state with error message
                table, bar_fig, pertype_fig, cost_fig = render("Overall", "Pass Rate", "High→Low", model_filter_val or "Models Avg")
                return f"❌ {error_msg}", table, bar_fig, pertype_fig, cost_fig
            
            # Reload and render after successful upload
            # Re-render with current settings (use Overall as default since we can't access component values directly)
            table, bar_fig, pertype_fig, cost_fig = render("Overall", "Pass Rate", "High→Low", model_filter_val or "Models Avg")
            return success_msg, table, bar_fig, pertype_fig, cost_fig

        upload_btn.click(
            handle_upload,
            inputs=[file_upload, model_filter, model_name_input, provider_input, agent_framework_input],
            outputs=[upload_status, out, bar, pertype_plot, cost_eff_plot]
        )

        demo.load(lambda: render("Overall", "Pass Rate", "High→Low", "Models Avg"), outputs=[out, bar, pertype_plot, cost_eff_plot])
        for comp in (cat, sort_col, sort_dir, model_filter):
            comp.change(render, inputs=[cat, sort_col, sort_dir, model_filter], outputs=[out, bar, pertype_plot, cost_eff_plot])
    return demo

if __name__ == "__main__":
    app().launch()
