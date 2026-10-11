import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.after_request
def _add_no_cache_headers(response):
    # These pages reflect live session/dataset state (uploads, imputation,
    # etc.), so the browser must never serve a stale cached copy of them —
    # otherwise a reload after an action like "Apply imputation" can show
    # old numbers even though the server-side data actually changed.
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response

# Needed for session (Flask signs the session cookie with this). Change this
# to a random, secret value before deploying anywhere real.
app.secret_key = "dev-only-change-me"

# ---------------------------------------------------------------------------
# Upload configuration
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = os.path.join(app.root_path, "data", "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
MAX_PREVIEW_ROWS = 8

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# Placeholder dataset summary shown on the home page until a real dataset
# is uploaded, at which point these are computed from it instead.
# ---------------------------------------------------------------------------
DATASET_SUMMARY = {
    "total_students": 500,
    "avg_cgpa": 6.96,
    "avg_iq": 100.3,
    "placement_rate": 55.2,
    "not_placed_rate": 44.8,
    "columns": [
        {"name": "CGPA", "dtype": "float", "note": "cumulative grade point, 0-10 scale"},
        {"name": "IQ", "dtype": "int", "note": "standardized aptitude score"},
        {"name": "Placement", "dtype": "category", "note": "target: Placed / Not Placed"},
    ],
}

# ---------------------------------------------------------------------------
# The ML pipeline, left to right in build order. This single list drives:
#   - the left sidebar stepper (every page)
#   - the roadmap on the home page
#   - the stub page rendered for each stage that isn't wired up yet
# ---------------------------------------------------------------------------
PIPELINE_STEPS = [
    {
        "id": "upload",
        "step": "01",
        "label": "Upload Dataset",
        "endpoint": "upload_dataset",
        "eyebrow": "Stage 01 · Intake",
        "note": "Bring in the raw CGPA, IQ, and Placement records as a CSV "
        "or Excel file to start the pipeline.",
    },
    {
        "id": "features",
        "step": "02",
        "label": "Analyse Features",
        "endpoint": "analyse_features",
        "eyebrow": "Stage 02 · Records",
        "note": "Pick a feature and an analysis method to inspect it — "
        "distribution, spread, missingness, or its pull on the target.",
    },
    {
        "id": "descriptive",
        "step": "03",
        "label": "Descriptive Statistics",
        "endpoint": "descriptive_statistics",
        "eyebrow": "Stage 03 · Records",
        "note": "Mean, median, spread, and range for CGPA and IQ, "
        "split by placement outcome.",
    },
    {
        "id": "missing",
        "step": "04",
        "label": "Missing Value Analysis",
        "endpoint": "missing_values",
        "eyebrow": "Stage 04 · Records",
        "note": "Find gaps in the record and decide whether to impute, "
        "drop, or flag them.",
    },
    {
        "id": "visualize",
        "step": "05",
        "label": "Data Visualization",
        "endpoint": "visualize_data",
        "eyebrow": "Stage 05 · Records",
        "note": "Plot CGPA and IQ distributions against placement "
        "outcome to spot the pattern by eye.",
    },
    {
        "id": "preprocess",
        "step": "06",
        "label": "Preprocessing",
        "endpoint": "preprocess_data",
        "eyebrow": "Stage 06 · Preparation",
        "note": "Encode the target, scale the features, and split "
        "into train and test sets.",
    },
    {
        "id": "train",
        "step": "07",
        "label": "Model Training",
        "endpoint": "train_model",
        "eyebrow": "Stage 07 · Modelling",
        "note": "Fit a classifier on the training set and tune it "
        "against the pipeline defined so far.",
    },
    {
        "id": "evaluate",
        "step": "08",
        "label": "Model Evaluation",
        "endpoint": "evaluate_model",
        "eyebrow": "Stage 08 · Modelling",
        "note": "Accuracy, precision/recall, confusion matrix, and "
        "feature importance on held-out data.",
    },
    {
        "id": "predict",
        "step": "09",
        "label": "Predict Placement",
        "endpoint": "predict_placement",
        "eyebrow": "Stage 09 · Assessment",
        "note": "Enter a CGPA and IQ score and get a placement call "
        "from the trained model.",
    },
]

# Analysis methods offered in the Analyse Features dropdown. "numeric_only"
# methods are disabled client-side when a categorical/text feature is picked.
ANALYSIS_METHODS = [
    {"id": "summary", "label": "Summary Statistics", "numeric_only": False},
    {"id": "distribution", "label": "Distribution Plot", "numeric_only": False},
    {"id": "boxplot", "label": "Box Plot (outlier view)", "numeric_only": True},
    {"id": "value_counts", "label": "Value Counts", "numeric_only": False},
    {"id": "missing", "label": "Missing Value Report", "numeric_only": False},
    {"id": "outliers", "label": "Outlier Detection (IQR)", "numeric_only": True},
    {"id": "correlation", "label": "Correlation with Target", "numeric_only": True},
]


def _find_step(step_id):
    return next(step for step in PIPELINE_STEPS if step["id"] == step_id)


# Bump this string every time app.py, a template, or static assets change.
# It's rendered in the page footer so you can confirm at a glance — no
# terminal needed — that the browser is actually talking to the version of
# the server you think it is.
APP_BUILD = "build-2026-08-09-08 (Stage 05 visualization + light theme)"


@app.context_processor
def inject_pipeline():
    # Makes pipeline_steps available in every template without passing
    # it explicitly from each view.
    return {"pipeline_steps": PIPELINE_STEPS, "app_build": APP_BUILD}


# ---------------------------------------------------------------------------
# Upload / dataset helpers
# ---------------------------------------------------------------------------

def _allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _read_dataset(filepath):
    if filepath.lower().endswith(".csv"):
        return pd.read_csv(filepath)
    return pd.read_excel(filepath)


def _build_preview(df, max_rows=MAX_PREVIEW_ROWS):
    preview_df = df.head(max_rows).copy()
    for col in preview_df.select_dtypes(include="number").columns:
        preview_df[col] = preview_df[col].round(2)

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "column_names": list(df.columns),
        "head": preview_df.to_dict(orient="records"),
        "dtypes": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
    }


def _get_current_dataframe():
    """Returns (df, dataset_name) for whatever is currently in session,
    or (None, None) if nothing has been uploaded yet / the file is gone."""
    filepath = session.get("dataset_path")
    if not filepath or not os.path.exists(filepath):
        return None, None
    try:
        df = _read_dataset(filepath)
    except Exception:  # noqa: BLE001
        return None, None
    return df, session.get("dataset_name")


def _likely_target(df):
    """Best-effort guess at the target column, so 'Correlation with Target'
    and the grouped descriptive-stats table have something sensible to use."""
    exact_candidates = ("Placement", "Placed", "Target", "Label", "Outcome")
    for candidate in exact_candidates:
        if candidate in df.columns:
            return candidate

    # Fall back to a substring match (case-insensitive) so columns like
    # "PlacementStatus" or "placement_result" are still recognised.
    keywords = ("placement", "placed", "target", "label", "outcome")
    for col in df.columns:
        lowered = str(col).lower()
        if any(keyword in lowered for keyword in keywords):
            return col

    return df.columns[-1] if len(df.columns) else None


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


# Chalkboard-dark palette reused for every generated chart so plots match
# the site theme instead of matplotlib's default white background.
CHART_BG = "#fffdf8"
CHART_FG = "#322c22"
CHART_GOLD = "#bc7a34"
CHART_GRID = "#e3d8bf"
CHART_ACCENT2 = "#3f7a68"


def _style_axes(ax):
    ax.set_facecolor(CHART_BG)
    ax.figure.set_facecolor(CHART_BG)
    for spine in ax.spines.values():
        spine.set_color(CHART_GRID)
    ax.tick_params(colors=CHART_FG, labelsize=9)
    ax.xaxis.label.set_color(CHART_FG)
    ax.yaxis.label.set_color(CHART_FG)
    ax.title.set_color(CHART_FG)
    ax.grid(True, color=CHART_GRID, linewidth=0.6, alpha=0.6)


def _run_feature_analysis(df, feature, method):
    if feature not in df.columns:
        return {"ok": False, "error": f"'{feature}' is not a column in the current dataset."}

    series = df[feature]
    is_numeric = pd.api.types.is_numeric_dtype(series)

    if method == "summary":
        if is_numeric:
            desc = series.describe()
            rows = [
                {"label": "Count", "value": f"{desc.get('count', 0):.0f}"},
                {"label": "Mean", "value": f"{desc.get('mean', float('nan')):.3f}"},
                {"label": "Std Dev", "value": f"{desc.get('std', float('nan')):.3f}"},
                {"label": "Min", "value": f"{desc.get('min', float('nan')):.3f}"},
                {"label": "25%", "value": f"{desc.get('25%', float('nan')):.3f}"},
                {"label": "Median", "value": f"{desc.get('50%', float('nan')):.3f}"},
                {"label": "75%", "value": f"{desc.get('75%', float('nan')):.3f}"},
                {"label": "Max", "value": f"{desc.get('max', float('nan')):.3f}"},
                {"label": "Skew", "value": f"{series.skew():.3f}"},
            ]
        else:
            rows = [
                {"label": "Count", "value": f"{series.count()}"},
                {"label": "Unique values", "value": f"{series.nunique()}"},
                {"label": "Most frequent", "value": str(series.mode().iloc[0]) if not series.mode().empty else "—"},
                {"label": "Frequency of top value", "value": f"{series.value_counts().iloc[0]}" if series.count() else "0"},
            ]
        return {"ok": True, "kind": "table", "title": f"Summary statistics — {feature}", "rows": rows}

    if method == "distribution":
        fig, ax = plt.subplots(figsize=(6, 3.6))
        if is_numeric:
            clean = series.dropna()
            ax.hist(clean, bins=min(20, max(5, clean.nunique())), color=CHART_GOLD, edgecolor=CHART_BG)
            ax.set_xlabel(feature)
            ax.set_ylabel("Count")
            ax.set_title(f"Distribution of {feature}")
        else:
            counts = series.value_counts().sort_values(ascending=False).head(15)
            ax.bar(counts.index.astype(str), counts.values, color=CHART_GOLD, edgecolor=CHART_BG)
            ax.set_ylabel("Count")
            ax.set_title(f"Value distribution of {feature}")
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        _style_axes(ax)
        return {"ok": True, "kind": "image", "title": f"Distribution — {feature}", "image": _fig_to_base64(fig)}

    if method == "boxplot":
        if not is_numeric:
            return {"ok": False, "error": "Box plots need a numeric feature."}
        clean = series.dropna()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        box = ax.boxplot(
            clean, vert=False, patch_artist=True,
            boxprops={"facecolor": CHART_GOLD, "edgecolor": CHART_FG},
            medianprops={"color": CHART_BG, "linewidth": 2},
            whiskerprops={"color": CHART_FG},
            capprops={"color": CHART_FG},
            flierprops={"markeredgecolor": CHART_GOLD, "marker": "o", "markersize": 4},
        )
        ax.set_yticks([])
        ax.set_xlabel(feature)
        ax.set_title(f"Box plot — {feature}")
        _style_axes(ax)
        return {"ok": True, "kind": "image", "title": f"Box plot — {feature}", "image": _fig_to_base64(fig)}

    if method == "value_counts":
        counts = series.value_counts(dropna=False).head(20)
        rows = [{"label": ("Missing" if pd.isna(idx) else str(idx)), "value": str(val)} for idx, val in counts.items()]
        return {"ok": True, "kind": "table", "title": f"Value counts — {feature} (top {len(rows)})", "rows": rows}

    if method == "missing":
        total = len(series)
        missing = int(series.isna().sum())
        pct = (missing / total * 100) if total else 0.0
        rows = [
            {"label": "Total rows", "value": f"{total}"},
            {"label": "Missing values", "value": f"{missing}"},
            {"label": "Missing %", "value": f"{pct:.2f}%"},
            {"label": "Complete rows", "value": f"{total - missing}"},
        ]
        return {"ok": True, "kind": "table", "title": f"Missing value report — {feature}", "rows": rows}

    if method == "outliers":
        if not is_numeric:
            return {"ok": False, "error": "Outlier detection (IQR) needs a numeric feature."}
        clean = series.dropna()
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = clean[(clean < lower) | (clean > upper)]
        rows = [
            {"label": "Q1", "value": f"{q1:.3f}"},
            {"label": "Q3", "value": f"{q3:.3f}"},
            {"label": "IQR", "value": f"{iqr:.3f}"},
            {"label": "Lower fence", "value": f"{lower:.3f}"},
            {"label": "Upper fence", "value": f"{upper:.3f}"},
            {"label": "Outliers found", "value": f"{len(outliers)} ({len(outliers) / len(clean) * 100:.1f}% of non-missing)"},
        ]
        return {"ok": True, "kind": "table", "title": f"Outlier detection (IQR) — {feature}", "rows": rows}

    if method == "correlation":
        if not is_numeric:
            return {"ok": False, "error": "Correlation needs a numeric feature."}
        target = _likely_target(df)
        if target is None or target == feature:
            return {"ok": False, "error": "No other column available to correlate against."}
        target_series = df[target]
        if pd.api.types.is_numeric_dtype(target_series):
            numeric_target = target_series
        else:
            numeric_target = pd.factorize(target_series)[0]
            numeric_target = pd.Series(numeric_target, index=df.index).replace(-1, np.nan)
        paired = pd.concat([series, pd.Series(numeric_target, index=df.index, name=target)], axis=1).dropna()
        if len(paired) < 2:
            return {"ok": False, "error": "Not enough overlapping data to compute a correlation."}
        corr = paired[feature].corr(paired[target])
        rows = [
            {"label": "Correlated against", "value": target},
            {"label": "Pearson correlation", "value": f"{corr:.3f}"},
            {"label": "Rows used", "value": f"{len(paired)}"},
        ]
        return {"ok": True, "kind": "table", "title": f"Correlation with target — {feature}", "rows": rows}

    return {"ok": False, "error": f"Unknown analysis method '{method}'."}


@app.route("/")
def home():
    return render_template("index.html", stats=DATASET_SUMMARY, active_step=None)


@app.route("/upload", methods=["GET", "POST"])
def upload_dataset():
    step = _find_step("upload")
    error = None
    preview = None

    if request.method == "POST":
        uploaded_file = request.files.get("dataset")

        if uploaded_file is None or uploaded_file.filename == "":
            error = "Choose a CSV or Excel file before uploading."
        elif not _allowed_file(uploaded_file.filename):
            error = "Only .csv, .xlsx, or .xls files are accepted."
        else:
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            uploaded_file.save(filepath)

            try:
                df = _read_dataset(filepath)
            except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
                error = f"Could not read that file: {exc}"
            else:
                session["dataset_path"] = filepath
                session["dataset_name"] = filename
                preview = _build_preview(df)

    if preview is None and error is None and session.get("dataset_path"):
        df, _ = _get_current_dataframe()
        if df is not None:
            preview = _build_preview(df)
        else:
            session.pop("dataset_path", None)
            session.pop("dataset_name", None)

    return render_template(
        "upload.html",
        step=step,
        active_step="upload",
        error=error,
        preview=preview,
        dataset_name=session.get("dataset_name"),
    )


@app.route("/upload/clear", methods=["POST"])
def clear_dataset():
    filepath = session.pop("dataset_path", None)
    session.pop("dataset_name", None)
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
    return redirect(url_for("upload_dataset"))


@app.route("/features")
def analyse_features():
    step = _find_step("features")
    df, dataset_name = _get_current_dataframe()

    feature_options = []
    if df is not None:
        for col in df.columns:
            feature_options.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "numeric": bool(pd.api.types.is_numeric_dtype(df[col])),
            })

    return render_template(
        "features.html",
        step=step,
        active_step="features",
        dataset_name=dataset_name,
        feature_options=feature_options,
        analysis_methods=ANALYSIS_METHODS,
        has_dataset=df is not None,
    )


def _compute_descriptive_stats(df, max_groups=8):
    numeric_df = df.select_dtypes(include="number")
    categorical_df = df.select_dtypes(exclude="number")

    numeric_rows = []
    for col in numeric_df.columns:
        series = numeric_df[col]
        desc = series.describe()
        missing = int(series.isna().sum())
        numeric_rows.append({
            "name": col,
            "count": f"{desc.get('count', 0):.0f}",
            "mean": f"{desc.get('mean', float('nan')):.3f}",
            "std": f"{desc.get('std', float('nan')):.3f}",
            "min": f"{desc.get('min', float('nan')):.3f}",
            "q25": f"{desc.get('25%', float('nan')):.3f}",
            "median": f"{desc.get('50%', float('nan')):.3f}",
            "q75": f"{desc.get('75%', float('nan')):.3f}",
            "max": f"{desc.get('max', float('nan')):.3f}",
            "skew": f"{series.skew():.3f}" if series.count() > 2 else "—",
            "missing": f"{missing} ({missing / len(series) * 100:.1f}%)" if len(series) else "0",
        })

    categorical_rows = []
    for col in categorical_df.columns:
        series = categorical_df[col]
        missing = int(series.isna().sum())
        mode_val = series.mode()
        top = str(mode_val.iloc[0]) if not mode_val.empty else "—"
        freq = int(series.value_counts().iloc[0]) if series.count() else 0
        categorical_rows.append({
            "name": col,
            "count": f"{series.count()}",
            "unique": f"{series.nunique()}",
            "top": top,
            "freq": f"{freq}",
            "missing": f"{missing} ({missing / len(series) * 100:.1f}%)" if len(series) else "0",
        })

    overview = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_count": int(numeric_df.shape[1]),
        "categorical_count": int(categorical_df.shape[1]),
        "missing_total": int(df.isna().sum().sum()),
        "missing_pct": f"{df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}" if df.size else "0.00",
    }

    grouped = None
    target = _likely_target(df)
    if target and target in categorical_df.columns.tolist() + [
        c for c in numeric_df.columns if numeric_df[c].nunique() <= max_groups
    ]:
        group_series = df[target]
        group_values = sorted(group_series.dropna().unique().tolist(), key=str)
        if 1 < len(group_values) <= max_groups:
            group_counts = group_series.value_counts()
            feature_rows = []
            for col in numeric_df.columns:
                if col == target:
                    continue
                means = df.groupby(target)[col].mean()
                feature_rows.append({
                    "name": col,
                    "group_values": [
                        (f"{means.get(g, float('nan')):.3f}" if g in means.index else "—")
                        for g in group_values
                    ],
                })
            if feature_rows:
                grouped = {
                    "target": target,
                    "groups": [str(g) for g in group_values],
                    "group_counts": [int(group_counts.get(g, 0)) for g in group_values],
                    "feature_rows": feature_rows,
                }

    return {
        "overview": overview,
        "numeric_rows": numeric_rows,
        "categorical_rows": categorical_rows,
        "grouped": grouped,
    }


@app.route("/api/analyse-feature", methods=["POST"])
def api_analyse_feature():
    payload = request.get_json(silent=True) or {}
    feature = payload.get("feature")
    method = payload.get("method")

    df, _ = _get_current_dataframe()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset is loaded. Upload one on Stage 01 first."}), 400
    if not feature or not method:
        return jsonify({"ok": False, "error": "Choose both a feature and an analysis method."}), 400

    try:
        result = _run_feature_analysis(df, feature, method)
    except Exception as exc:  # noqa: BLE001 - surface computation errors to the UI, not a 500 page
        return jsonify({"ok": False, "error": f"Could not run that analysis: {exc}"}), 200

    status = 200 if result.get("ok") else 200
    return jsonify(result), status


def _suggest_missing_action(dtype_str, missing_pct, is_numeric):
    if missing_pct == 0:
        return "Complete — no action needed"
    if missing_pct >= 50:
        return "Consider dropping this column (too sparse)"
    if missing_pct >= 30:
        return "High missingness — impute carefully or drop"
    if is_numeric:
        return "Impute with mean/median"
    return "Impute with mode / 'Unknown' category"


def _compute_missing_report(df):
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    rows_with_missing = int(df.isna().any(axis=1).sum())
    complete_rows = int(df.shape[0] - rows_with_missing)

    overview = {
        "total_cells": total_cells,
        "total_missing": total_missing,
        "missing_pct": f"{(total_missing / total_cells * 100) if total_cells else 0:.2f}",
        "rows_with_missing": rows_with_missing,
        "complete_rows": complete_rows,
        "complete_rows_pct": f"{(complete_rows / df.shape[0] * 100) if df.shape[0] else 0:.2f}",
        "columns_with_missing": int((df.isna().sum() > 0).sum()),
    }

    rows = []
    chart_labels, chart_values = [], []
    for col in df.columns:
        series = df[col]
        missing = int(series.isna().sum())
        pct = (missing / len(series) * 100) if len(series) else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(series)
        rows.append({
            "name": col,
            "dtype": str(series.dtype),
            "missing": missing,
            "missing_pct": f"{pct:.2f}",
            "action": _suggest_missing_action(str(series.dtype), pct, is_numeric),
            "is_numeric": is_numeric,
        })
        if missing > 0:
            chart_labels.append(col)
            chart_values.append(pct)

    # Worst-first for both the table and the chart.
    rows.sort(key=lambda r: r["missing"], reverse=True)
    if chart_labels:
        paired = sorted(zip(chart_labels, chart_values), key=lambda p: p[1])
        chart_labels, chart_values = [p[0] for p in paired], [p[1] for p in paired]

    chart_image = None
    if chart_labels:
        display_labels = chart_labels[-25:]
        display_values = chart_values[-25:]
        height = max(2.5, 0.32 * len(display_labels))
        fig, ax = plt.subplots(figsize=(7, height))
        ax.barh(display_labels, display_values, color=CHART_GOLD, edgecolor=CHART_BG)
        ax.set_xlabel("Missing %")
        ax.set_title("Missing values by column" + (" (top 25)" if len(chart_labels) > 25 else ""))
        _style_axes(ax)
        chart_image = _fig_to_base64(fig)

    return {
        "overview": overview,
        "rows": rows,
        "chart_image": chart_image,
        "has_missing": total_missing > 0,
    }


@app.route("/descriptive")
def descriptive_statistics():
    step = _find_step("descriptive")
    df, dataset_name = _get_current_dataframe()

    stats = _compute_descriptive_stats(df) if df is not None else None

    return render_template(
        "descriptive.html",
        step=step,
        active_step="descriptive",
        dataset_name=dataset_name,
        has_dataset=df is not None,
        stats=stats,
    )


def _save_dataset(df, filepath):
    if filepath.lower().endswith(".csv"):
        df.to_csv(filepath, index=False)
    else:
        df.to_excel(filepath, index=False)


@app.route("/missing")
def missing_values():
    step = _find_step("missing")
    df, dataset_name = _get_current_dataframe()

    report = _compute_missing_report(df) if df is not None else None
    numeric_missing_features = []
    if report:
        numeric_missing_features = [
            r for r in report["rows"] if r["is_numeric"] and r["missing"] > 0
        ]

    last_imputation = session.pop("last_imputation", None)

    return render_template(
        "missing.html",
        step=step,
        active_step="missing",
        dataset_name=dataset_name,
        has_dataset=df is not None,
        report=report,
        numeric_missing_features=numeric_missing_features,
        last_imputation=last_imputation,
    )


@app.route("/api/impute-feature", methods=["POST"])
def api_impute_feature():
    payload = request.get_json(silent=True) or {}
    feature = payload.get("feature")
    method = payload.get("method")

    filepath = session.get("dataset_path")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"ok": False, "error": "No dataset is loaded. Upload one on Stage 01 first."})
    if not feature or method not in ("mean", "median"):
        return jsonify({"ok": False, "error": "Choose both a feature and a method (mean or median)."})

    try:
        df = _read_dataset(filepath)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Could not read the dataset: {exc}"})

    if feature not in df.columns:
        return jsonify({"ok": False, "error": f"'{feature}' is not a column in the current dataset."})

    series = df[feature]
    if not pd.api.types.is_numeric_dtype(series):
        return jsonify({"ok": False, "error": "Mean/median imputation only applies to numeric columns."})

    missing_before = int(series.isna().sum())
    if missing_before == 0:
        return jsonify({"ok": False, "error": f"'{feature}' already has no missing values."})

    mean_before = float(series.mean())
    median_before = float(series.median())
    std_before = float(series.std()) if series.count() > 1 else float("nan")

    fill_value = mean_before if method == "mean" else median_before
    df[feature] = series.fillna(fill_value)

    try:
        _save_dataset(df, filepath)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Could not save the imputed dataset: {exc}"})

    filled_series = df[feature]
    mean_after = float(filled_series.mean())
    median_after = float(filled_series.median())
    std_after = float(filled_series.std()) if filled_series.count() > 1 else float("nan")

    def _fmt(x):
        return f"{x:.4f}" if x == x else "N/A"  # x == x is False only for NaN

    result = {
        "ok": True,
        "feature": feature,
        "method": method,
        "missing_before": missing_before,
        "fill_value": _fmt(fill_value),
        "mean_before": _fmt(mean_before),
        "mean_after": _fmt(mean_after),
        "median_before": _fmt(median_before),
        "median_after": _fmt(median_after),
        "std_before": _fmt(std_before),
        "std_after": _fmt(std_after),
    }
    session["last_imputation"] = result
    return jsonify(result)


def _numeric_vs_target(df, feature, target):
    if feature not in df.columns:
        return {"ok": False, "error": f"'{feature}' is not a column in the current dataset."}
    if target not in df.columns:
        return {"ok": False, "error": f"No target column ('{target}') found to compare against."}
    if not pd.api.types.is_numeric_dtype(df[feature]):
        return {"ok": False, "error": f"'{feature}' isn't numeric — pick a numeric feature."}

    group_values = sorted(df[target].dropna().unique().tolist(), key=str)
    if not (1 < len(group_values) <= 10):
        return {"ok": False, "error": f"'{target}' needs 2–10 distinct outcomes to compare against; found {len(group_values)}."}

    data_by_group = [df.loc[df[target] == g, feature].dropna() for g in group_values]
    labels = [str(g) for g in group_values]
    palette = [CHART_GOLD, CHART_ACCENT2, "#8a6fae", "#c2554a", "#4a7fae", "#a58a3f"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    box = ax.boxplot(
        data_by_group, tick_labels=labels, patch_artist=True,
        medianprops={"color": CHART_FG, "linewidth": 2},
        whiskerprops={"color": CHART_FG},
        capprops={"color": CHART_FG},
        flierprops={"markeredgecolor": CHART_GOLD, "marker": "o", "markersize": 4},
    )
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(palette[i % len(palette)])
        patch.set_edgecolor(CHART_FG)
        patch.set_alpha(0.85)
    ax.set_xlabel(target)
    ax.set_ylabel(feature)
    ax.set_title(f"{feature} by {target}")
    _style_axes(ax)

    rows = []
    for g, series in zip(labels, data_by_group):
        rows.append({
            "label": g,
            "value": f"n={len(series)} · mean={series.mean():.3f} · median={series.median():.3f}",
        })

    return {
        "ok": True,
        "kind": "image",
        "title": f"{feature} vs {target}",
        "image": _fig_to_base64(fig),
        "rows": rows,
    }


def _categorical_vs_target(df, feature, target, max_categories=15):
    if feature not in df.columns:
        return {"ok": False, "error": f"'{feature}' is not a column in the current dataset."}
    if target not in df.columns:
        return {"ok": False, "error": f"No target column ('{target}') found to compare against."}
    if pd.api.types.is_numeric_dtype(df[feature]) and df[feature].nunique() > max_categories:
        return {"ok": False, "error": f"'{feature}' looks continuous — pick a categorical feature instead."}

    group_values = sorted(df[target].dropna().unique().tolist(), key=str)
    if not (1 < len(group_values) <= 10):
        return {"ok": False, "error": f"'{target}' needs 2–10 distinct outcomes to compare against; found {len(group_values)}."}

    counts = df.groupby([feature, target]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=group_values, fill_value=0)
    totals = counts.sum(axis=1).sort_values(ascending=False)
    top_categories = totals.head(max_categories).index
    counts = counts.loc[top_categories]
    pct = counts.div(counts.sum(axis=1), axis=0) * 100

    labels = [str(c) for c in counts.index]
    palette = [CHART_GOLD, CHART_ACCENT2, "#8a6fae", "#c2554a", "#4a7fae", "#a58a3f"]

    fig, ax = plt.subplots(figsize=(7, max(3.2, 0.4 * len(labels) + 1.5)))
    bottoms = np.zeros(len(labels))
    for i, g in enumerate(group_values):
        values = pct[g].values if g in pct.columns else np.zeros(len(labels))
        ax.barh(labels, values, left=bottoms, color=palette[i % len(palette)], edgecolor=CHART_BG, label=str(g))
        bottoms += values
    ax.set_xlabel(f"% of {target}")
    ax.set_title(f"{feature} vs {target} (top {len(labels)} categories)" if len(totals) > max_categories else f"{feature} vs {target}")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, title=target)
    legend.get_title().set_color(CHART_FG)
    for text in legend.get_texts():
        text.set_color(CHART_FG)
    _style_axes(ax)

    rows = []
    for cat in counts.index:
        breakdown = " · ".join(f"{g}: {pct.loc[cat, g]:.1f}% ({int(counts.loc[cat, g])})" for g in group_values)
        rows.append({"label": str(cat), "value": breakdown})

    return {
        "ok": True,
        "kind": "image",
        "title": f"{feature} vs {target}",
        "image": _fig_to_base64(fig),
        "rows": rows,
    }


@app.route("/visualize")
def visualize_data():
    step = _find_step("visualize")
    df, dataset_name = _get_current_dataframe()

    numeric_options, categorical_options = [], []
    target = None
    target_ok = False

    if df is not None:
        target = _likely_target(df)
        if target is not None:
            n_groups = df[target].dropna().nunique()
            target_ok = 1 < n_groups <= 10
        for col in df.columns:
            if col == target:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_options.append(col)
            else:
                categorical_options.append(col)

    return render_template(
        "visualize.html",
        step=step,
        active_step="visualize",
        dataset_name=dataset_name,
        has_dataset=df is not None,
        target=target,
        target_ok=target_ok,
        numeric_options=numeric_options,
        categorical_options=categorical_options,
    )


@app.route("/api/visualize-feature", methods=["POST"])
def api_visualize_feature():
    payload = request.get_json(silent=True) or {}
    feature = payload.get("feature")
    chart_type = payload.get("chart_type")

    df, _ = _get_current_dataframe()
    if df is None:
        return jsonify({"ok": False, "error": "No dataset is loaded. Upload one on Stage 01 first."})
    if not feature or chart_type not in ("numeric", "categorical"):
        return jsonify({"ok": False, "error": "Choose a feature first."})

    target = _likely_target(df)
    try:
        if chart_type == "numeric":
            result = _numeric_vs_target(df, feature, target)
        else:
            result = _categorical_vs_target(df, feature, target)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"Could not build that chart: {exc}"})

    return jsonify(result)


def _make_stage_view(step):
    """Builds a stub view for a pipeline stage. Replace the body of the
    returned function with real logic (pandas, sklearn, etc.) stage by
    stage - the routing and sidebar stay exactly as they are."""

    def view():
        return render_template("stage.html", step=step, active_step=step["id"])

    view.__name__ = step["endpoint"]
    return view


for _step in PIPELINE_STEPS:
    if _step["id"] in ("upload", "features", "descriptive", "missing", "visualize"):
        continue  # these have real routes defined above
    app.add_url_rule(
        f"/{_step['id']}",
        endpoint=_step["endpoint"],
        view_func=_make_stage_view(_step),
    )


if __name__ == "__main__":
    app.run(debug=True)
