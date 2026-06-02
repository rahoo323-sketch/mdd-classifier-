import os, io, re, warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import joblib

warnings.filterwarnings("ignore")

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

UPLOAD_FOLDER = "uploads"
MODEL_FOLDER  = "model"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

_model_cache = {}

def load_model(model_name):
    if model_name in _model_cache:
        return _model_cache[model_name]
    path = os.path.join(MODEL_FOLDER, f"deprescan_{model_name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found. "
            f"Please run: python train_model.py --file GSE98793_series_matrix.txt --model {model_name}"
        )
    pkg = joblib.load(path)
    _model_cache[model_name] = pkg
    return pkg

def parse_single_patient(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    start = end = None
    for i, line in enumerate(lines):
        if "!series_matrix_table_begin" in line:
            start = i + 1
        if "!series_matrix_table_end" in line:
            end = i
            break

    if start is None or end is None:
        raise ValueError("Could not find !series_matrix_table_begin marker.")

    expr_lines = lines[start:end]
    content = "".join(expr_lines)
    df = pd.read_csv(
        io.StringIO(content), sep="\t", index_col=0,
        na_values=["null","NULL","NA",""]
    )

    if df.shape[1] == 0:
        raise ValueError("No expression columns found in file.")

    series = df.iloc[:, 0]
    expr = {str(k): float(v) for k, v in series.items() if pd.notna(v)}
    return expr

def classify_single_patient(filepath, model_name="svm"):
    pkg = load_model(model_name)

    clf         = pkg["clf"]
    scaler      = pkg["scaler"]
    pca         = pkg["pca"]
    chosen_idx  = pkg["chosen_idx"]
    gene_names  = pkg["gene_names"]
    final_genes = pkg["final_genes"]
    classes     = clf.classes_

    expr = parse_single_patient(filepath)

    X_patient = np.array([
        np.log2(max(expr.get(g, 0), 0) + 1.0)
        for g in gene_names
    ], dtype=np.float64).reshape(1, -1)

    X_scaled = scaler.transform(X_patient)
    X_sel = X_scaled[:, chosen_idx]
    X_pca = pca.transform(X_sel)

    pred  = clf.predict(X_pca)[0]
    proba = clf.predict_proba(X_pca)[0]

    class_list = list(classes)
    if 1 in class_list:
        mdd_prob  = float(proba[class_list.index(1)])
        ctrl_prob = float(proba[class_list.index(0)])
    else:
        mdd_prob  = float(proba[0]) if pred == 1 else 1.0 - float(proba[0])
        ctrl_prob = 1.0 - mdd_prob

    confidence = mdd_prob if pred == 1 else ctrl_prob

    return {
        "prediction":   "MDD" if pred == 1 else "Healthy Control",
        "label":        int(pred),
        "confidence":   round(confidence * 100, 1),
        "auc":          pkg["auc"],
        "sensitivity":  pkg["sensitivity"],
        "specificity":  pkg["specificity"],
        "n_features":   pkg["n_features"],
        "n_components": pkg["n_components"],
        "final_genes":  final_genes[:10],
        "model":        model_name,
    }

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

@app.route("/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.endswith(".txt"):
        return jsonify({"error": "File must be a .txt GEO series matrix"}), 400
    model_name = request.form.get("model", "svm")
    filename   = secure_filename(f.filename)
    filepath   = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(filepath)
    try:
        result = classify_single_patient(filepath, model_name)
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.remove(filepath)
        except: pass

@app.route("/model-status", methods=["GET"])
def model_status():
    models = ["svm", "rf", "logistic", "boosting"]
    status = {}
    for m in models:
        path = os.path.join(MODEL_FOLDER, f"deprescan_{m}.pkl")
        if os.path.exists(path):
            try:
                pkg = joblib.load(path)
                status[m] = {
                    "available": True,
                    "auc": pkg.get("auc", "—"),
                    "acc": pkg.get("accuracy", "—")
                }
            except:
                status[m] = {"available": False}
        else:
            status[m] = {"available": False}
    return jsonify(status)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
