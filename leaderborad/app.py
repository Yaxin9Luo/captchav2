# leaderboard/app.py
import pandas as pd, numpy as np, matplotlib.pyplot as plt, gradio as gr

CATEGORY_MAP = {
    "Overall": ["Overall"],
    # You can define sets, e.g. "Vision-hard": ["Adversarial","Deformation","Squiggle"]
}

def load_df(path="leaderboard/results.csv"):
    df = pd.read_csv(path)
    for c in ["Overall"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    # All per-type columns become numeric
    for c in df.columns:
        if c not in ["Model","Provider","Notes"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def compute_score(df, category):
    cols = CATEGORY_MAP.get(category, [category] if category in df.columns else ["Overall"])
    df = df.copy()
    df["Category Score"] = df[cols].mean(axis=1, skipna=True)
    return df

def table_html(df):
    rows = []
    for i, r in df.iterrows():
        rows.append(f"""
        <tr>
          <td><b>{i+1}</b></td>
          <td>{r['Model']}</td>
          <td>{r.get('Provider','')}</td>
          <td>{r.get('Notes','')}</td>
          <td>{r['Category Score']:.3f}</td>
        </tr>""")
    return f"""
    <style>
      table.lb{{width:100%;border-collapse:collapse}}
      table.lb th,td{{padding:10px;border-bottom:1px solid #eee;text-align:left}}
      table.lb tr:hover{{background:#fafafa}}
    </style>
    <table class="lb">
      <thead><tr><th>#</th><th>Model</th><th>Provider</th><th>Notes</th><th>Score</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """

def perf_bar(df):
    plt.close("all")
    d = df.sort_values("Category Score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5*len(d))))
    ax.barh(range(len(d)), d["Category Score"])
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["Model"])
    ax.set_xlabel("Score"); ax.set_xlim(0, 1); ax.set_title("Performance")
    fig.tight_layout(); return fig

def perf_by_type(df):
    plt.close("all")
    # Average each per-type column across models
    numeric_cols = [c for c in df.columns if c not in ["Model","Provider","Notes"]]
    type_cols = [c for c in numeric_cols if c != "Overall" and df[c].notna().any()]
    means = df[type_cols].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(means)), means.values)
    ax.set_xticks(range(len(means))); ax.set_xticklabels(means.index, rotation=30, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.set_title("Per-Type Mean Accuracy")
    fig.tight_layout(); return fig

def render(model_filter, category, sort_by):
    df = load_df()
    if model_filter and model_filter != "All":
        df = df[df["Provider"] == model_filter] if "Provider" in df.columns else df
    df = compute_score(df, category)
    df = df.sort_values("Category Score", ascending=(sort_by=="Low→High")).reset_index(drop=True)
    return table_html(df), perf_bar(df), perf_by_type(df)

def app():
    df = load_df()
    providers = ["All"] + sorted(df["Provider"].dropna().unique().tolist()) if "Provider" in df.columns else ["All"]
    cats = ["Overall"] + [c for c in df.columns if c not in ["Model","Provider","Notes","Overall"]]

    with gr.Blocks(title="CAPTCHAv2 Leaderboard") as demo:
        gr.Markdown("## CAPTCHAv2 Leaderboard")
        with gr.Row():
            mf = gr.Dropdown(choices=providers, value="All", label="Provider")
            cat = gr.Dropdown(choices=["Overall"]+cats, value="Overall", label="Category/Type")
            sb = gr.Radio(choices=["High→Low","Low→High"], value="High→Low", label="Sort by")
        out = gr.HTML(); bar = gr.Plot(); pertype = gr.Plot()

        demo.load(lambda: render("All","Overall","High→Low"), outputs=[out, bar, pertype])
        for comp in (mf, cat, sb):
            comp.change(render, inputs=[mf, cat, sb], outputs=[out, bar, pertype])
    return demo

if __name__ == "__main__":
    app().launch()
