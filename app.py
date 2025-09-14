import os
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

# --- Configuration ---
UPLOAD_FOLDER = "uploads"
PLOTS_FOLDER = "static/plots"
ALLOWED_EXTENSIONS = {"csv"}
SECRET_KEY = "change-this-to-a-random-secret-key"  # change before deployment

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PLOTS_FOLDER"] = PLOTS_FOLDER
app.secret_key = SECRET_KEY

# Allow uploads up to 500 MB
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

sns.set(style="whitegrid")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def unique_path(folder, name):
    base, ext = os.path.splitext(name)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    uniq = f"{base}_{stamp}_{uuid.uuid4().hex[:6]}{ext}"
    return os.path.join(folder, uniq)


def generate_plots(df, prefix):
    """Generates plots (from df) and returns list of web paths like '/static/plots/xxx.png'."""
    plot_files = []
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Histograms (up to 6)
    for col in numeric[:6]:
        try:
            plt.figure(figsize=(6, 4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f"Histogram: {col}")
            fname = f"{prefix}_hist_{col}.png".replace(" ", "_")
            path = os.path.join(app.config["PLOTS_FOLDER"], fname)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plot_files.append(f"/static/plots/{fname}")
        except Exception:
            app.logger.exception("Failed to create histogram for %s", col)

    # Boxplots (up to 3)
    for col in numeric[:3]:
        try:
            plt.figure(figsize=(6, 3))
            sns.boxplot(x=df[col].dropna())
            plt.title(f"Boxplot: {col}")
            fname = f"{prefix}_box_{col}.png".replace(" ", "_")
            path = os.path.join(app.config["PLOTS_FOLDER"], fname)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plot_files.append(f"/static/plots/{fname}")
        except Exception:
            app.logger.exception("Failed to create boxplot for %s", col)

    # Pie charts for categorical top counts (up to 3)
    for col in categorical[:3]:
        try:
            counts = df[col].fillna("<<Missing>>").value_counts().nlargest(6)
            if counts.sum() == 0:
                continue
            plt.figure(figsize=(5, 5))
            counts.plot.pie(autopct="%1.1f%%", startangle=90)
            plt.ylabel("")
            plt.title(f"Distribution: {col}")
            fname = f"{prefix}_pie_{col}.png".replace(" ", "_")
            path = os.path.join(app.config["PLOTS_FOLDER"], fname)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plot_files.append(f"/static/plots/{fname}")
        except Exception:
            app.logger.exception("Failed to create pie chart for %s", col)

    # Bar chart for top categories (first categorical)
    if categorical:
        col = categorical[0]
        try:
            counts = df[col].fillna("<<Missing>>").value_counts().nlargest(10)
            plt.figure(figsize=(8, 4))
            sns.barplot(x=counts.values, y=counts.index)
            plt.title(f"Top categories: {col}")
            fname = f"{prefix}_bar_{col}.png".replace(" ", "_")
            path = os.path.join(app.config["PLOTS_FOLDER"], fname)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plot_files.append(f"/static/plots/{fname}")
        except Exception:
            app.logger.exception("Failed to create bar chart for %s", col)

    # Correlation heatmap for numeric features (if >=2)
    if len(numeric) >= 2:
        try:
            corr = df[numeric].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("Correlation heatmap")
            fname = f"{prefix}_corr.png".replace(" ", "_")
            path = os.path.join(app.config["PLOTS_FOLDER"], fname)
            plt.tight_layout()
            plt.savefig(path)
            plt.close()
            plot_files.append(f"/static/plots/{fname}")
        except Exception:
            app.logger.exception("Failed to create correlation heatmap")

    return plot_files


def dataset_summary(df):
    """Return a DataFrame summarizing columns (dtype, non-null count, unique, mean/std if numeric)."""
    rows = []
    for col in df.columns:
        try:
            dtype = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            unique = int(df[col].nunique(dropna=True))
            sample = str(df[col].dropna().iloc[0]) if non_null > 0 else ""
            info = {"column": col, "dtype": dtype, "non_null_count": non_null, "unique_values": unique, "sample_value": sample}
            if pd.api.types.is_numeric_dtype(df[col]):
                info["mean"] = float(df[col].mean(skipna=True)) if non_null else None
                info["std"] = float(df[col].std(skipna=True)) if non_null else None
            else:
                info["mean"] = None
                info["std"] = None
            rows.append(info)
        except Exception:
            app.logger.exception("Error summarizing column %s", col)
    return pd.DataFrame(rows)


# --- Routes ---
@app.route("/", methods=["GET", "POST"])
def upload_file():
    message = None
    uploaded_filename = session.get("uploaded_filename", None)
    if request.method == "POST":
        action = request.form.get("action")
        if action == "upload":
            if "file" not in request.files:
                flash("No file part")
                return redirect(request.url)
            file = request.files["file"]
            if file.filename == "":
                flash("No selected file")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                path = unique_path(app.config["UPLOAD_FOLDER"], filename)
                file.save(path)
                basename = os.path.basename(path)
                session["uploaded_filename"] = basename  # store only basename
                size_mb = os.path.getsize(path) / (1024 * 1024)
                message = f"Uploaded {basename} ({size_mb:.2f} MB)"
                app.logger.info("Saved upload to %s (%.2f MB)", path, size_mb)
            else:
                flash("Allowed file types: csv")
                return redirect(request.url)
        elif action == "analyze":
            if not uploaded_filename:
                flash("No file uploaded yet. Please upload a CSV first.")
                return redirect(request.url)
            return redirect(url_for("results"))
    return render_template("upload.html", message=message, uploaded_filename=uploaded_filename)


@app.route("/results")
def results():
    uploaded_basename = session.get("uploaded_filename", None)
    if not uploaded_basename:
        flash("No file available for analysis. Please upload a CSV first.")
        return redirect(url_for("upload_file"))

    full_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_basename)
    if not os.path.exists(full_path):
        flash("Uploaded file missing on server. Please re-upload.")
        session.pop("uploaded_filename", None)
        return redirect(url_for("upload_file"))

    file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
    app.logger.info("Preparing analysis for %s (%.2f MB)", full_path, file_size_mb)

    # Try reading the full file, fallback to sample if memory fails or other errors
    df = None
    loaded_full = False
    try:
        # attempt to load full file (may OOM on very large files)
        df = pd.read_csv(full_path, low_memory=False)
        loaded_full = True
        app.logger.info("Loaded full CSV with shape %s", df.shape)
    except MemoryError:
        app.logger.exception("MemoryError while reading full CSV - will try sample")
    except Exception as e:
        app.logger.warning("Could not read full CSV (%s). Will attempt to read sample. Trace: %s", e, traceback.format_exc())

    if not loaded_full:
        # fallback to reading a sample (first chunk or nrows)
        try:
            # try chunked read -> use first chunk as sample
            chunks = pd.read_csv(full_path, chunksize=100000)
            df_sample = next(chunks)
            df = df_sample
            flash("Full file could not be loaded into memory. Analysis performed on a 100k-row sample.")
            app.logger.info("Loaded sample from CSV with shape %s", df.shape)
        except Exception as e:
            app.logger.exception("Failed to read sample from CSV: %s", e)
            flash(f"Failed to read file for analysis: {e}")
            return redirect(url_for("upload_file"))

    # Use df for summary if full loaded, otherwise summary from sample (user warned by flash)
    summary_source = "full file" if loaded_full else "sample"
    if loaded_full:
        summary_df = dataset_summary(df)
    else:
        summary_df = dataset_summary(df)

    # Rows / cols: try to get true row count if full file not loaded by counting lines
    if loaded_full:
        rows, cols = df.shape
    else:
        try:
            with open(full_path, "rb") as f:
                total_lines = sum(1 for _ in f)
            rows = max(total_lines - 1, 0)  # minus header
            cols = df.shape[1]
        except Exception:
            rows, cols = df.shape

    # Prepare sample_for_plots: if dataset too big, sample up to 100k rows
    try:
        if loaded_full and len(df) > 100000:
            sample_for_plots = df.sample(n=100000, random_state=42)
        else:
            sample_for_plots = df
    except Exception:
        # fallback if sampling fails
        sample_for_plots = df.head(100000)

    # Render head (sample)
    head_html = sample_for_plots.head(10).to_html(classes="table-sample", index=False, escape=False)
    summary_html = summary_df.to_html(classes="invisible-border-table", index=False, float_format="%.3f", na_rep="")

    # Generate plots from sample_for_plots
    prefix = os.path.splitext(uploaded_basename)[0]
    try:
        plots = generate_plots(sample_for_plots, prefix)
    except Exception:
        app.logger.exception("Failed to generate plots")
        plots = []

    return render_template("results.html",
                           filename=uploaded_basename,
                           rows=rows,
                           cols=cols,
                           head=head_html,
                           summary_html=summary_html,
                           plots=plots,
                           summary_source=summary_source)


# Friendly error message for oversized payloads
@app.errorhandler(413)
def too_large(e):
    return "File is too large! Please upload a file smaller than 500 MB.", 413


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
