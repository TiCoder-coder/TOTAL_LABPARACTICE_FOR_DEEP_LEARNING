import pandas as pd
from IPython.display import HTML, display

THEME_AWARE_CSS = """
<style>
    :root {
        --table-bg: #ffffff;
        --table-header-bg: #f8f9fa;
        --table-header-text: #212529;
        --table-border: #dee2e6;
        --table-row-alt: #f1f3f5;
        --table-text: #212529;
        --table-muted: #6c757d;
        
        --success-bg: #d4edda;
        --success-text: #155724;
        --success-border: #c3e6cb;
        
        --danger-bg: #f8d7da;
        --danger-text: #721c24;
        --danger-border: #f5c6cb;
        
        --warning-bg: #fff3cd;
        --warning-text: #856404;
        --warning-border: #ffeeba;
        
        --info-bg: #e2e3e5;
        --info-text: #383d41;
        --info-border: #d6d8db;
        
        --primary-bg: #cce5ff;
        --primary-text: #004085;
        --primary-border: #b8daff;
        
        --badge-radius: 4px;
        --badge-padding: 3px 8px;
        --badge-font: 11px;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --table-bg: #1e1e1e;
            --table-header-bg: #2d2d2d;
            --table-header-text: #e0e0e0;
            --table-border: #444444;
            --table-row-alt: #252525;
            --table-text: #cccccc;
            --table-muted: #888888;
            
            --success-bg: #14331e;
            --success-text: #75b798;
            --success-border: #14331e;
            
            --danger-bg: #3d171a;
            --danger-text: #ea868f;
            --danger-border: #3d171a;
            
            --warning-bg: #403408;
            --warning-text: #ffda6a;
            --warning-border: #403408;
            
            --info-bg: #2d2d2d;
            --info-text: #a8a8a8;
            --info-border: #2d2d2d;
            
            --primary-bg: #092c34;
            --primary-text: #6edff6;
            --primary-border: #092c34;
        }
    }
    
    .theme-table {
        background-color: var(--table-bg) !important;
        color: var(--table-text) !important;
        border: 1px solid var(--table-border) !important;
        border-collapse: collapse !important;
        margin-bottom: 1em;
        width: 100%;
        font-size: 14px;
    }
    .theme-table th {
        background-color: var(--table-header-bg) !important;
        color: var(--table-header-text) !important;
        border-bottom: 2px solid var(--table-border) !important;
        border: 1px solid var(--table-border) !important;
        padding: 10px 14px !important;
        font-weight: 600 !important;
        text-align: left;
    }
    .theme-table td {
        border: 1px solid var(--table-border) !important;
        padding: 8px 14px !important;
    }
    .theme-table tbody tr:nth-child(even) {
        background-color: var(--table-row-alt) !important;
    }
    
    .badge {
        border-radius: var(--badge-radius);
        padding: var(--badge-padding);
        font-size: var(--badge-font);
        font-weight: bold;
        display: inline-block;
        border: 1px solid transparent;
        text-transform: uppercase;
    }
    .badge-success { background-color: var(--success-bg); color: var(--success-text); border-color: var(--success-border); }
    .badge-danger { background-color: var(--danger-bg); color: var(--danger-text); border-color: var(--danger-border); }
    .badge-warning { background-color: var(--warning-bg); color: var(--warning-text); border-color: var(--warning-border); }
    .badge-info { background-color: var(--info-bg); color: var(--info-text); border-color: var(--info-border); }
    .badge-primary { background-color: var(--primary-bg); color: var(--primary-text); border-color: var(--primary-border); }
    
    .cell-success { background-color: var(--success-bg) !important; color: var(--success-text) !important; font-weight: bold; }
    .cell-danger { background-color: var(--danger-bg) !important; color: var(--danger-text) !important; font-weight: bold; }
    .cell-warning { background-color: var(--warning-bg) !important; color: var(--warning-text) !important; }
    .cell-info { background-color: var(--info-bg) !important; color: var(--info-text) !important; }
    .cell-primary { background-color: var(--primary-bg) !important; color: var(--primary-text) !important; font-weight: bold; }
    .cell-muted { color: var(--table-muted) !important; }
</style>
"""

def inject_theme_css():
    display(HTML(THEME_AWARE_CSS))

def create_badge(text, style_type):
    return f'<span class="badge badge-{style_type}">{text}</span>'

def render_summary_cards(cards):
    html = '<div style="display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">'
    for title, value, status, badge_type in cards:
        html += f'''
        <div style="border-left: 5px solid var(--{badge_type}-text); padding: 15px 20px; background-color: var(--table-header-bg); border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 150px; border: 1px solid var(--table-border); border-left-width: 5px; border-left-color: var(--{badge_type}-text);">
            <div style="font-size: 12px; color: var(--table-muted); text-transform: uppercase; font-weight: bold;">{title}</div>
            <div style="font-size: 26px; font-weight: bold; margin: 8px 0; color: var(--table-text);">{value}</div>
            <div style="margin-top: 8px;">{create_badge(status, badge_type)}</div>
        </div>
        '''
    html += '</div>'
    return HTML(html)

def style_test_metrics_table(df):
    df_formatted = df.copy()
    def format_row(row):
        m, v = row["Metric"], row["Result"]
        if "Accuracy" in m:
            val = float(v)
            if val <= 1.0: val *= 100
            return f"{val:.2f}%"
        elif any(k in m for k in ["Loss", "Macro", "Weighted"]):
            return f"{float(v):.4f}"
        else:
            return f"{int(v):,}"
    df_formatted["Result"] = df.apply(format_row, axis=1)
    
    def get_classes(row):
        m = row["Metric"]
        if "Accuracy" in m or "Macro F1" == m: return ["", "cell-primary"]
        if "Correct" == m: return ["", "cell-success"]
        if "Incorrect" == m: return ["", "cell-danger"]
        if "Loss" in m: return ["", "cell-warning"]
        return ["", "cell-muted"]
        
    classes = pd.DataFrame(df_formatted.apply(get_classes, axis=1).tolist(), columns=["Metric", "Result"], index=df.index)
    return df_formatted.style.set_table_attributes('class="theme-table"').set_td_classes(classes).hide(axis="index")

def style_transform_table(df):
    df_formatted = df.copy()
    def format_applied(val):
        v = str(val).upper()
        if "TRAIN" in v: return create_badge("TRAIN ONLY", "primary")
        if "VAL" in v or "TEST" in v: return create_badge("VAL / TEST", "warning")
        return create_badge("ALL SPLITS", "info")
    df_formatted["Applied to"] = df["Applied to"].apply(format_applied)
    return df_formatted.style.set_table_attributes('class="theme-table"').hide(axis="index")

def style_leakage_table(df):
    df_formatted = df.copy()
    def format_status(val):
        v = str(val).upper()
        if v in ["0", "PASS", "SAFE"]: return create_badge("PASS", "success")
        if v.isdigit() and int(v) > 0: return create_badge(f"FAIL ({v})", "danger")
        if "FAIL" in v or "REVIEW" in v: return create_badge("FAIL / REVIEW", "danger")
        if "PARTIAL" in v: return create_badge("PARTIAL", "warning")
        return create_badge(v, "info")
    df_formatted["Status"] = df["Status"].apply(format_status)
    return df_formatted.style.set_table_attributes('class="theme-table"').hide(axis="index")

def style_model_comparison(df):
    df_formatted = df.copy()
    df_formatted["Train Acc (%)"] = df["Train Acc (%)"].apply(lambda x: f"{x:.2f}%")
    df_formatted["Val Acc (%)"] = df["Val Acc (%)"].apply(lambda x: f"{x:.2f}%")
    df_formatted["Val Loss"] = df["Val Loss"].apply(lambda x: f"{x:.4f}")
    df_formatted["Val F1"] = df["Val F1"].apply(lambda x: f"{x:.4f}")
    df_formatted["Gap (%)"] = df["Gap (%)"].apply(lambda x: f"{x:.2f} pp")
    
    def format_status(val):
        v = str(val)
        if "Winner" in v: return create_badge("WINNER", "success")
        if "Overfit" in v: return create_badge(v.upper(), "danger")
        if "Underfit" in v: return create_badge(v.upper(), "info")
        if "Historical" in v: return create_badge(v.upper(), "info")
        return create_badge(v.upper(), "info")
    df_formatted["Status"] = df["Status"].apply(format_status)
    
    val_acc_max = df["Val Acc (%)"].max()
    val_loss_min = df["Val Loss"].min()
    
    def get_classes(row):
        c = {col: "" for col in df.columns}
        st = str(row["Status"])
        if "Historical" in st or "Underfit" in st:
            for col in c: c[col] = "cell-muted"
        if row["Val Acc (%)"] == val_acc_max: c["Val Acc (%)"] += " cell-primary"
        if row["Val Loss"] == val_loss_min: c["Val Loss"] += " cell-success"
        if "Overfit" in st or row["Gap (%)"] > 10: c["Gap (%)"] += " cell-danger"
        if "Winner" in st: c["Experiment"] += " cell-success"
        return pd.Series(c)
        
    classes = df.apply(get_classes, axis=1)
    return df_formatted.style.set_table_attributes('class="theme-table"').set_td_classes(classes).hide(axis="index")

def style_distribution_table(df):
    df_formatted = df.copy()
    if "Total" not in df_formatted.columns:
        df_formatted["Total"] = df_formatted[["Train", "Validation", "Test"]].sum(axis=1)
    def get_classes(row):
        c = {col: "" for col in df_formatted.columns}
        if "Train" in c: c["Train"] = "cell-primary"
        if "Validation" in c: c["Validation"] = "cell-warning"
        if "Test" in c: c["Test"] = "cell-danger"
        return pd.Series(c)
    classes = df_formatted.apply(get_classes, axis=1)
    return df_formatted.style.set_table_attributes('class="theme-table"').set_td_classes(classes)

def style_training_history(df):
    df_formatted = df.copy()
    df_formatted["train_loss"] = df["train_loss"].apply(lambda x: f"{x:.4f}")
    df_formatted["val_loss"] = df["val_loss"].apply(lambda x: f"{x:.4f}")
    df_formatted["train_acc"] = df["train_acc"].apply(lambda x: f"{x:.2f}%")
    df_formatted["val_acc"] = df["val_acc"].apply(lambda x: f"{x:.2f}%")
    df_formatted["lr"] = df["lr"].apply(lambda x: f"{x:.1e}")
    
    val_acc_max = df["val_acc"].max()
    val_loss_min = df["val_loss"].min()
    
    def get_classes(row):
        c = {col: "" for col in df_formatted.columns}
        if row["val_acc"] == val_acc_max: c["val_acc"] = "cell-success"
        if row["val_loss"] == val_loss_min: c["val_loss"] = "cell-success"
        return pd.Series(c)
    classes = df.apply(get_classes, axis=1)
    return df_formatted.style.set_table_attributes('class="theme-table"').set_td_classes(classes).hide(axis="index")

def style_generic_table(df):
    return df.style.set_table_attributes('class="theme-table"')
