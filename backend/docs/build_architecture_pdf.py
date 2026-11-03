r"""Build the architecture & step-by-step guide PDF for the ML backend.

Usage (from backend/):
    .\.venv\Scripts\python.exe -m pip install -r docs\requirements-docs.txt
    .\.venv\Scripts\python.exe docs\build_architecture_pdf.py

Output: docs/Credit_Risk_ML_Backend_Architecture.pdf
Metrics are read from models/metrics.json, so the PDF reflects the model
that is actually trained. Nothing is hardcoded.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, Preformatted,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

BACKEND = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "Credit_Risk_ML_Backend_Architecture.pdf"
METRICS = BACKEND / "models" / "metrics.json"

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

NAVY = colors.HexColor("#1F3A5F")
BLUE = colors.HexColor("#DCE8F5")
GREEN = colors.HexColor("#DFF2E1")
AMBER = colors.HexColor("#FCEFD6")
GREY = colors.HexColor("#ECEFF3")
RED = colors.HexColor("#F8DEDE")
PURPLE = colors.HexColor("#E9E1F5")

# --------------------------------------------------------------------------- styles
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=17, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=NAVY, fontSize=13, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=ss["Heading3"], textColor=NAVY, fontSize=11, spaceBefore=7, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=5)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, leading=11.5)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.3, leading=10.8, spaceAfter=0)
CELL_B = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
HDR = ParagraphStyle("Hdr", parent=CELL_B, textColor=colors.white)
CODE = ParagraphStyle("Code", parent=ss["Code"], fontSize=7.8, leading=10, backColor=GREY,
                      borderPadding=5, leftIndent=4, rightIndent=4, spaceBefore=3, spaceAfter=7)
CAPTION = ParagraphStyle("Caption", parent=SMALL, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=10)
NOTE = ParagraphStyle("Note", parent=BODY, backColor=AMBER, borderPadding=6, borderColor=colors.HexColor("#E0B252"),
                      borderWidth=0.6, spaceBefore=4, spaceAfter=10)
TITLE = ParagraphStyle("Title", parent=ss["Title"], textColor=NAVY, fontSize=24, leading=30)
SUBTITLE = ParagraphStyle("Sub", parent=BODY, alignment=TA_CENTER, fontSize=12, leading=17, textColor=colors.HexColor("#333333"))


def P(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str], style=BODY) -> ListFlowable:
    return ListFlowable([ListItem(P(i, style), leftIndent=10) for i in items], bulletType="bullet", start="•", leftIndent=12)


def numbered(items: list[str], style=BODY) -> ListFlowable:
    return ListFlowable([ListItem(P(i, style), leftIndent=14) for i in items], bulletType="1", leftIndent=16)


def code(text: str) -> Preformatted:
    return Preformatted(text.strip("\n"), CODE)


def table(rows: list[list[str]], widths: list[float], header_bg=NAVY) -> Table:
    data = [[P(c, HDR if r == 0 else CELL) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FA")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


# --------------------------------------------------------------------------- diagram helpers
def box(d: Drawing, x, y, w, h, text: str, fill=BLUE, bold_first=True, font=7.6):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=fill, strokeColor=NAVY, strokeWidth=0.8))
    lines = text.split("\n")
    lead = font + 2.2
    top = y + h / 2 + (len(lines) - 1) * lead / 2 - font / 3
    for i, line in enumerate(lines):
        d.add(String(x + w / 2, top - i * lead, line, fontSize=font + (0.6 if i == 0 and bold_first else 0),
                     fontName="Helvetica-Bold" if (i == 0 and bold_first) else "Helvetica",
                     textAnchor="middle", fillColor=colors.black))


def arrow(d: Drawing, x1, y1, x2, y2, label: str | None = None, color=NAVY):
    d.add(Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=1.1))
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    s = 5
    p1 = (x2 - s * math.cos(ang - 0.4), y2 - s * math.sin(ang - 0.4))
    p2 = (x2 - s * math.cos(ang + 0.4), y2 - s * math.sin(ang + 0.4))
    d.add(Polygon([x2, y2, p1[0], p1[1], p2[0], p2[1]], fillColor=color, strokeColor=color))
    if label:
        d.add(String((x1 + x2) / 2 + 4, (y1 + y2) / 2, label, fontSize=6.8, fontName="Helvetica", fillColor=colors.HexColor("#333333")))


def system_diagram() -> Drawing:
    W, H = CONTENT_W, 400
    d = Drawing(W, H)
    cx = W / 2
    # Layer 1: clients
    box(d, 20, H - 50, W - 40, 38, "Clients (other team members)\nReact frontend  |  Dashboard  |  Reporting  |  Swagger UI /docs", PURPLE)
    arrow(d, cx, H - 50, cx, H - 72, "HTTP + JSON (CORS-allowed origins)")
    # Layer 2: FastAPI app
    box(d, 20, H - 118, W - 40, 44, "FastAPI application  (app/main.py)\nCORS middleware  |  uniform error handlers {error:{code,message,details}}  |  GET /api/health\nPydantic request/response validation  (app/schemas/credit_risk.py)", BLUE)
    arrow(d, cx, H - 118, cx, H - 138)
    # Layer 3: routes
    rw = (W - 40 - 30) / 4
    routes = [
        ("routes/data.py", "POST /api/data/upload\nGET /api/data/summary"),
        ("routes/training.py", "POST /api/model/train\nGET /api/model/metrics\nGET /api/model/status"),
        ("routes/prediction.py", "POST /api/predict"),
        ("routes/explanation.py", "POST /api/explain"),
    ]
    for i, (n, t) in enumerate(routes):
        box(d, 20 + i * (rw + 10), H - 196, rw, 56, f"{n}\n{t}", GREEN)
        arrow(d, 20 + i * (rw + 10) + rw / 2, H - 196, 20 + i * (rw + 10) + rw / 2, H - 216)
    # Layer 4: services
    sw = (W - 40 - 40) / 5
    services = [
        ("data_processing", "load CSV, validate,\ntarget mapping, leakage,\nquality report"),
        ("model_training", "split, pipelines,\n3 models, selection,\nsave/load joblib"),
        ("evaluation", "precision, recall, F1,\nROC-AUC, confusion,\nthreshold tuning"),
        ("prediction", "input validation,\nprobability, class,\nrisk bands"),
        ("explainability", "SHAP Tree/Linear\nglobal + local\nexplanations"),
    ]
    for i, (n, t) in enumerate(services):
        box(d, 20 + i * (sw + 10), H - 282, sw, 64, f"{n}.py\n{t}", AMBER, font=6.8)
    d.add(String(cx, H - 296, "Business logic layer (app/services/)  -  errors.py maps domain errors to HTTP status codes", fontSize=7, fontName="Helvetica", textAnchor="middle", fillColor=colors.HexColor("#444444")))
    arrow(d, cx, H - 300, cx, H - 318)
    # Layer 5: store
    box(d, 20, H - 356, W - 40, 36, "services/store.py  -  Store (in-memory cache + disk persistence, training lock)\nholds the current dataset and the active model artifact; reloads both on restart", GREY)
    arrow(d, cx, H - 356, cx, H - 370)
    # Layer 6: disk
    dw = (W - 40 - 20) / 3
    disk = [
        "data/uploads/current_dataset.csv\n+ current_dataset_meta.json",
        "models/credit_risk_model.joblib\n(pipeline + schema + SHAP data)",
        "models/metrics.json\n(all evaluation results)",
    ]
    for i, t in enumerate(disk):
        box(d, 20 + i * (dw + 10), 2, dw, 32, t, RED, bold_first=False, font=7.0)
    return d


def pipeline_diagram() -> Drawing:
    W, H = CONTENT_W, 215
    d = Drawing(W, H)
    w, h, gap = (W - 4 * 14) / 5, 52, 14
    row1 = [
        ("1. Load CSV", "parse, size check,\nstrip index cols"),
        ("2. Quality report", "missing, duplicates,\ndtypes, imbalance"),
        ("3. Map target", "labels -> 1 = default\n0 = non-default"),
        ("4. Drop columns", "leakage, IDs,\nconstant, all-missing"),
        ("5. De-duplicate", "remove exact\nduplicate rows"),
    ]
    row2 = [
        ("10. SHAP", "global importance +\nper-borrower factors"),
        ("9. Save artifact", "joblib pipeline +\nmetrics.json"),
        ("8. Evaluate (TEST)", "untouched 20%:\nall metrics"),
        ("7. Select + tune (VAL)", "best model by ROC-AUC,\nthreshold by max F1"),
        ("6. Split + fit (TRAIN)", "stratified 60/20/20,\nfit 3 pipelines on train"),
    ]
    y1, y2 = H - h - 10, 30
    for i, (a, b) in enumerate(row1):
        x = i * (w + gap)
        box(d, x, y1, w, h, f"{a}\n{b}", BLUE, font=7.2)
        if i < 4:
            arrow(d, x + w, y1 + h / 2, x + w + gap, y1 + h / 2)
    arrow(d, 4 * (w + gap) + w / 2, y1, 4 * (w + gap) + w / 2, y2 + h)
    fills = [PURPLE, RED, GREEN, AMBER, AMBER]
    for i, (a, b) in enumerate(row2):
        x = i * (w + gap)
        box(d, x, y2, w, h, f"{a}\n{b}", fills[i], font=7.2)
        if i > 0:
            arrow(d, x, y2 + h / 2, x - gap, y2 + h / 2)
    d.add(String(W / 2, 8, "Preprocessing statistics are learned from the TRAIN split only; VAL picks the model; TEST is only used for the final report.",
                 fontSize=7, fontName="Helvetica", textAnchor="middle", fillColor=colors.HexColor("#444444")))
    return d


def sklearn_pipeline_diagram() -> Drawing:
    W, H = CONTENT_W, 150
    d = Drawing(W, H)
    box(d, 0, 50, 92, 50, "Raw borrower\nrecord(s)\n(DataFrame)", GREY, font=7.2)
    arrow(d, 92, 75, 108, 75)
    box(d, 108, 50, 92, 50, "FeatureCoercer\nselect features,\ntext -> numbers,\nblank -> missing", AMBER, font=7.0)
    arrow(d, 200, 85, 220, 112)
    arrow(d, 200, 65, 220, 38)
    box(d, 220, 92, 150, 50, "Numeric branch\nmedian imputation\n+ missing-indicator columns\n-> StandardScaler", BLUE, font=7.0)
    box(d, 220, 8, 150, 50, "Categorical branch\nfill '__missing__'\n-> OneHotEncoder\n(unseen -> safe, max 30 cats)", BLUE, font=7.0)
    arrow(d, 370, 117, 392, 85)
    arrow(d, 370, 33, 392, 65)
    box(d, 392, 50, W - 392, 50, "Classifier\nLogisticRegression |\nRandomForest |\nDecisionTree", GREEN, font=7.0)
    return d


def prediction_flow_diagram() -> Drawing:
    W, H = CONTENT_W, 92
    d = Drawing(W, H)
    steps = [
        ("POST /api/predict", "JSON records", PURPLE),
        ("Pydantic", "shape, 1-1000\nrecords, top_k", BLUE),
        ("validate_records", "required fields,\ntypes, ranges,\nunseen categories", AMBER),
        ("pipeline", "predict_proba\n-> P(default)", GREEN),
        ("decide", "class via threshold\nrisk band via bands", GREEN),
        ("SHAP local", "top factors up /\ndown", RED),
        ("JSON response", "+ warnings\n+ disclaimer", PURPLE),
    ]
    gap = 8
    w = (W - gap * (len(steps) - 1)) / len(steps)
    for i, (a, b, f) in enumerate(steps):
        x = i * (w + gap)
        box(d, x, 18, w, 62, f"{a}\n{b}", f, font=6.6)
        if i < len(steps) - 1:
            arrow(d, x + w, 49, x + w + gap, 49)
    d.add(String(W / 2, 4, "Any validation problem stops the request with HTTP 422 and a list of exactly which field failed.",
                 fontSize=7, fontName="Helvetica", textAnchor="middle", fillColor=colors.HexColor("#444444")))
    return d


# --------------------------------------------------------------------------- page decorations
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(MARGIN, 10 * mm, "Credit Risk Scoring & Loan Default Analysis  -  ML Backend Architecture (Member 4)")
    canvas.drawRightString(PAGE_W - MARGIN, 10 * mm, f"Page {doc.page}")
    canvas.setStrokeColor(NAVY)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, PAGE_H - 12 * mm, PAGE_W - MARGIN, PAGE_H - 12 * mm)
    canvas.restoreState()


# --------------------------------------------------------------------------- content
def metrics_section(story: list) -> None:
    story.append(P("9. Current results", H1))
    if not METRICS.exists():
        story.append(P("No trained model found (models/metrics.json missing). Train a model and rebuild this PDF.", NOTE))
        return
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    synthetic = m.get("data", {}).get("is_synthetic", False)
    if synthetic:
        story.append(P("<b>These numbers come from SYNTHETIC development data.</b> They prove the pipeline works end to end, "
                       "but they say nothing about real-world credit performance. Do not present them as real results.", NOTE))
    s = m["split"]
    story.append(P(f"Trained at <b>{m['trained_at'][:19].replace('T', ' ')} UTC</b>. Selected model: <b>{m['selected_model']}</b> "
                   f"(best validation {m['selection_metric']}). Decision threshold: <b>{m['decision_threshold']}</b>. "
                   f"Split: {s['train_rows']} train / {s['validation_rows']} validation / {s['test_rows']} test rows, "
                   f"default rate about {s['test_default_rate']:.1%} in each split."))
    rows = [["Model", "Val " + m["selection_metric"], "Test precision", "Test recall", "Test F1", "Test ROC-AUC", "Train ROC-AUC"]]
    for c in m["comparison"]:
        name = c["model"] + ("  (selected)" if c["selected"] else "")
        rows.append([name, str(c.get("validation_" + m["selection_metric"])), str(c["test_precision"]), str(c["test_recall"]),
                     str(c["test_f1"]), str(c["test_roc_auc"]), str(c["train_roc_auc"])])
    story.append(table(rows, [118, 58, 58, 52, 46, 58, 60]))
    story.append(Spacer(1, 6))
    best = m["models"][m["selected_model"]]["test"]
    cm = best["confusion_matrix"]
    story.append(P("Confusion matrix of the selected model on the test split:", H3))
    cmt = Table([["", "Predicted non-default", "Predicted default"],
                 ["Actual non-default", str(cm["true_negative"]), str(cm["false_positive"])],
                 ["Actual default", str(cm["false_negative"]), str(cm["true_positive"])]],
                colWidths=[110, 110, 110])
    cmt.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")),
        ("BACKGROUND", (0, 0), (-1, 0), BLUE), ("BACKGROUND", (0, 0), (0, -1), BLUE),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3), ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(cmt)
    story.append(Spacer(1, 6))
    story.append(P(
        f"How to read it: of {best['n_default']} real defaults in the test set, the model caught "
        f"{cm['true_positive']} (recall {best['recall']}). Of all borrowers it flagged as default, "
        f"{best['precision']:.0%} really defaulted (precision {best['precision']}). Lowering the threshold catches more defaults "
        "but flags more good borrowers; the <i>min_default_recall</i> training option controls this trade-off."))
    excl = m["data"]["excluded_columns"]
    story.append(P("Columns removed before training: " + ", ".join(f"<b>{k}</b> ({v})" for k, v in excl.items()) + "."))


def build() -> Path:
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Credit Risk ML Backend - Architecture", author="Member 4 - ML Backend")
    st: list = []

    # ---------------- cover
    st += [Spacer(1, 60), P("Credit Risk Scoring &amp;<br/>Loan Default Analysis", TITLE), Spacer(1, 10),
           P("Machine-Learning Backend: Architecture and Step-by-Step Guide", SUBTITLE), Spacer(1, 4),
           P("Member 4 - Backend Machine Learning", SUBTITLE), Spacer(1, 30)]
    st.append(table([
        ["Item", "Details"],
        ["Purpose", "Learn from historical loans, predict the probability that a new borrower defaults, and explain why."],
        ["Stack", "Python 3.12, FastAPI, Pydantic, pandas, NumPy, scikit-learn, SHAP, joblib, pytest"],
        ["Models", "Logistic Regression, Random Forest, Decision Tree (compared automatically)"],
        ["Interface", "8 REST endpoints returning JSON, consumed by the frontend / dashboard team"],
        ["Location", "Everything lives in the <b>backend/</b> folder"],
        ["Generated", datetime.now().strftime("%d %B %Y")],
    ], [90, CONTENT_W - 90]))
    st += [Spacer(1, 20), P("Contents", H2), numbered([
        "The big picture", "Folder structure", "System architecture", "The ML pipeline, step by step",
        "Inside the saved model pipeline", "Prediction and explanation flow", "API reference", "Error handling",
        "Current results", "Run it yourself, step by step", "Testing", "Limitations and next steps",
    ]), PageBreak()]

    # ---------------- 1 big picture
    st.append(P("1. The big picture", H1))
    st.append(P("This backend is the \"brain\" of the project. It has no user interface. Other team members' frontend and "
                "dashboard send it HTTP requests and receive JSON answers. Its job has four parts:"))
    st.append(numbered([
        "<b>Understand the data</b> - load a loan CSV, check its quality, and report problems.",
        "<b>Learn</b> - train three classification models on past loans and keep the best one.",
        "<b>Predict</b> - for a new borrower, return the probability of default, a yes/no class and a Low/Medium/High risk band.",
        "<b>Explain</b> - use SHAP to show which borrower details pushed the risk up or down.",
    ]))
    st.append(P("Three principles shaped every design decision:", H3))
    st.append(bullets([
        "<b>No cheating (leakage prevention).</b> Information only known after a loan ends, such as recovered amounts, is removed. "
        "The test data is never used for learning or choosing.",
        "<b>Honest numbers.</b> Every metric is calculated from real predictions on held-out data. Nothing is hardcoded.",
        "<b>Safe inputs.</b> Every request is validated; bad data returns a clear error that names the field at fault.",
    ]))

    # ---------------- 2 folders
    st.append(P("2. Folder structure (all inside backend/)", H1))
    st.append(table([
        ["Path", "What it contains"],
        ["app/main.py", "Creates the FastAPI app, CORS, error handlers, the health endpoint, and registers all routes."],
        ["app/config.py", "Settings (folders, 20 MB upload limit, CORS origins), default risk bands, disclaimer texts."],
        ["app/routes/", "Thin HTTP layer: data.py, training.py, prediction.py, explanation.py, deps.py. Parse the request, call a service, return JSON."],
        ["app/schemas/credit_risk.py", "Pydantic models describing every request and response (the API contract)."],
        ["app/services/data_processing.py", "CSV loading, target mapping, feature typing, leakage detection, quality report."],
        ["app/services/model_training.py", "Train/val/test split, preprocessing pipeline, the 3 models, selection, joblib save/load."],
        ["app/services/evaluation.py", "Precision, recall, F1, ROC-AUC, confusion matrix, classification report, threshold tuning."],
        ["app/services/prediction.py", "Validates borrower records, scores them, assigns risk bands."],
        ["app/services/explainability.py", "SHAP explainers, global importance, per-borrower explanations."],
        ["app/services/store.py", "Keeps the current dataset and model in memory and on disk; blocks parallel training."],
        ["app/services/errors.py", "Named error types, each mapped to an HTTP status code."],
        ["data/synthetic_loans.csv", "SYNTHETIC development dataset (3,030 rows). Uploaded files go to data/uploads/."],
        ["models/", "credit_risk_model.joblib (trained pipeline) and metrics.json (all results)."],
        ["scripts/", "generate_synthetic_data.py and smoke_test.py (end-to-end check)."],
        ["tests/", "47 pytest tests: data, pipeline, metrics, prediction, SHAP, API, errors."],
        ["docs/", "This PDF and the script that builds it."],
        ["README.md", "Setup commands and the full API contract with examples."],
    ], [150, CONTENT_W - 150]))
    st.append(PageBreak())

    # ---------------- 3 architecture
    st.append(P("3. System architecture", H1))
    st.append(P("The code is split into layers. Each layer only talks to the one below it, so the HTTP details, "
                "the machine-learning logic and the storage can each change without breaking the others."))
    st.append(system_diagram())
    st.append(P("Figure 1 - Layered architecture of the ML backend", CAPTION))
    st.append(table([
        ["Layer", "Responsibility", "Why it is separate"],
        ["Clients", "Frontend, dashboard, reports (other members)", "They only need the JSON contract, not Python."],
        ["FastAPI app", "HTTP, CORS, validation, one error format", "Consistent behaviour for every endpoint."],
        ["Routes", "Translate HTTP into service calls", "Kept thin so logic is testable without a server."],
        ["Services", "All data and ML logic", "Pure Python, reused by API, scripts and tests."],
        ["Store", "Current dataset + model, persistence, training lock", "Survives restarts; prevents two trainings at once."],
        ["Disk", "CSV, joblib model, metrics JSON", "No database existed in the project to integrate with."],
    ], [70, 200, CONTENT_W - 270]))
    st.append(PageBreak())

    # ---------------- 4 pipeline
    st.append(P("4. The machine-learning pipeline, step by step", H1))
    st.append(pipeline_diagram())
    st.append(P("Figure 2 - What happens when POST /api/model/train is called", CAPTION))

    steps = [
        ("Step 1 - Load the CSV",
         "The upload must be a .csv file under 20 MB. Empty files, unreadable text and duplicate column names are rejected. "
         "Index columns that pandas sometimes writes (\"Unnamed: 0\") are dropped."),
        ("Step 2 - Data-quality report",
         "Counts missing values per column, exact duplicate rows, numbers stored as text (e.g. \"38,000\"), mixed types, "
         "and the class balance. It also suggests likely target columns. The report is returned on upload and by GET /api/data/summary."),
        ("Step 3 - Map the target",
         "The target column and which labels mean \"default\" are configurable. Labels are normalised (case, spaces, 1/1.0/True). "
         "If the target has more than two values (for example \"Current\" loans whose outcome is unknown), you must list "
         "negative_labels so unknown outcomes are not silently counted as good loans. Rows with no target are dropped and counted."),
        ("Step 4 - Remove unusable or leaking columns",
         "Dropped automatically: columns whose names signal post-outcome information (recoveries, total payments, last payment, "
         "settlement, outstanding principal...), ID columns, constant columns, all-missing columns and free text. "
         "A numeric column that on its own separates defaults almost perfectly (AUC 0.98 or more) is flagged as a warning. "
         "Extra columns can be excluded with exclude_columns."),
        ("Step 5 - Remove duplicates",
         "Exact duplicate rows are removed before splitting, so the same loan cannot appear in both training and test data."),
        ("Step 6 - Split and train",
         "Stratified, reproducible split: 60% train, 20% validation, 20% test, each with the same default rate. "
         "For each of the three models a full pipeline (Section 5) is fitted <b>on the training split only</b>. "
         "Because defaults are rare (about 20%), every model uses balanced class weights so it cannot score well by always saying \"no default\"."),
        ("Step 7 - Pick the threshold and the model (validation split)",
         "A model outputs a probability; a threshold turns it into yes/no. The threshold is tuned on validation data to maximise F1 "
         "for the default class, or to reach a minimum recall if min_default_recall is set. The model with the best validation score "
         "(ROC-AUC by default) is selected."),
        ("Step 8 - Evaluate (test split)",
         "The untouched test split gives the final, honest numbers for every model: precision, recall, default-class recall, F1, accuracy, "
         "balanced accuracy, ROC-AUC, PR-AUC, Brier score, confusion matrix and the full classification report. "
         "Train ROC-AUC is reported too, so overfitting is visible (e.g. a much higher train than test score)."),
        ("Step 9 - Save",
         "The selected pipeline, the feature schema (allowed fields, training ranges, known categories), the threshold, risk bands, "
         "metrics and SHAP background data are saved to models/credit_risk_model.joblib. Metrics also go to models/metrics.json. "
         "The file is written to a temporary name first and then swapped in, so a crash cannot leave a half-written model."),
        ("Step 10 - SHAP check",
         "Before the model is saved, global SHAP importance is computed once. If explanations fail, the model is not activated."),
    ]
    for title, text in steps:
        st.append(KeepTogether([P(title, H3), P(text)]))

    # ---------------- 5 saved model
    st.append(P("5. Inside the saved model pipeline", H1))
    st.append(P("Preprocessing and the classifier are saved together as one scikit-learn Pipeline. This guarantees that a borrower "
                "is transformed at prediction time exactly as the training data was."))
    st.append(sklearn_pipeline_diagram())
    st.append(P("Figure 3 - The scikit-learn pipeline stored in the joblib file", CAPTION))
    st.append(table([
        ["Component", "What it does", "Why"],
        ["FeatureCoercer", "Keeps only model features, converts \"1,200\" to 1200, blanks to missing, booleans to text.",
         "Learns nothing, so it cannot leak information. Same cleaning at train and predict time."],
        ["Median imputation + indicators", "Fills missing numbers with the training median and adds a \"was missing\" column.",
         "Missingness itself can carry risk signal."],
        ["StandardScaler", "Puts numeric features on a common scale.", "Needed by logistic regression; harmless for trees."],
        ["'__missing__' + OneHotEncoder", "Turns categories into 0/1 columns; rare ones (under 5 rows) are grouped; unseen values become all zeros.",
         "A new category in production never crashes the API."],
        ["Classifier", "Logistic Regression (C=1), Random Forest (200 trees), Decision Tree (depth 6).",
         "Simple, explainable models with balanced class weights."],
    ], [110, 200, CONTENT_W - 310]))
    st.append(P("Three models compared", H2))
    st.append(table([
        ["Model", "How it thinks", "Strength", "Weakness"],
        ["Logistic Regression", "Adds up weighted features into a score (log-odds).", "Stable, easy to explain, good with little data.", "Misses non-linear patterns."],
        ["Random Forest", "Averages 200 decision trees built on random subsets.", "Captures interactions; robust.", "Harder to explain; can overfit."],
        ["Decision Tree", "One tree of yes/no questions.", "Very easy to read.", "Unstable; usually less accurate."],
    ], [90, 160, 120, CONTENT_W - 370]))

    # ---------------- 6 prediction flow
    st.append(P("6. Prediction and explanation flow", H1))
    st.append(prediction_flow_diagram())
    st.append(P("Figure 4 - What happens when POST /api/predict is called", CAPTION))
    st.append(P("Input rules", H3))
    st.append(bullets([
        "Every model feature must be present as a key. Send <b>null</b> if a value is unknown; it will be imputed and a warning is returned.",
        "Numeric fields accept numbers or numeric text (\"38,000\"). Text such as \"high\", booleans or infinity are rejected.",
        "Values outside the training range are accepted with a warning, because the model is less reliable there.",
        "Unknown categories are accepted with a warning. Extra fields are ignored with a warning.",
    ]))
    st.append(P("Two separate decisions", H3))
    st.append(bullets([
        "<b>predicted_class</b> - default if probability is at least the tuned decision threshold.",
        "<b>risk_band</b> - from configurable probability bands. Default: Low below 0.20, Medium below 0.50, High otherwise.",
    ]))
    st.append(P("How SHAP explanations work", H3))
    st.append(P("SHAP starts from a baseline (the average model output) and assigns each borrower field a value showing how much it moved "
                "this borrower's score up (more risk) or down (less risk). Base value plus all SHAP values equals the model's output exactly; "
                "the tests check this. Tree models use TreeExplainer and report in probability units. Logistic regression uses "
                "LinearExplainer and reports in log-odds units. One-hot columns are added back together, so explanations always "
                "name the original field, such as home_ownership, not home_ownership==RENT."))
    st.append(P("SHAP explains how the <b>model</b> behaves. It does not prove that a factor <b>causes</b> default in the real world.", NOTE))
    st.append(PageBreak())

    # ---------------- 7 API
    st.append(P("7. API reference", H1))
    st.append(P("Base URL when running locally: <b>http://127.0.0.1:8000</b>. Interactive docs: <b>/docs</b>. Call order: upload, then train, then predict / explain."))
    st.append(table([
        ["Method + path", "Purpose", "Key input", "Key output"],
        ["GET /api/health", "Is the server up?", "-", "status, dataset_loaded, model_loaded"],
        ["POST /api/data/upload", "Upload and validate a loan CSV", "file, target_column, positive_labels, negative_labels, is_synthetic (form-data)", "dataset info + quality report (201)"],
        ["GET /api/data/summary", "Dataset statistics and quality", "optional target overrides (query)", "missing, duplicates, types, class balance, leakage warnings"],
        ["POST /api/model/train", "Train and compare the 3 models", "JSON, all optional: models, test_size, selection_metric, risk_bands...", "selected model, threshold, ranked comparison, warnings"],
        ["GET /api/model/metrics", "All evaluation results", "-", "per-model train/val/test metrics, confusion matrices, reports"],
        ["GET /api/model/status", "Is a model ready? Which inputs?", "-", "model name, feature schema, risk bands, test metrics"],
        ["POST /api/predict", "Score borrowers", "records (1-1000), include_explanation, top_k", "probability, class, risk band, top factors, warnings"],
        ["POST /api/explain", "SHAP explanations", "records (optional), include_global, top_k", "global importance, summary plot data, local factors"],
    ], [95, 95, 160, CONTENT_W - 350]))
    st.append(P("Example: predict one borrower", H3))
    st.append(code("""
POST /api/predict
{
  "records": [{
    "age": 29, "annual_income": 38000, "employment_length_years": 1.5,
    "employment_type": "contract", "home_ownership": "RENT", "loan_amount": 25000,
    "loan_term_months": 60, "interest_rate": 21.5, "loan_purpose": "small_business",
    "credit_score": 590, "debt_to_income": 34.0, "credit_utilization_pct": 85.0,
    "num_open_accounts": 12, "delinquencies_2y": 2
  }],
  "include_explanation": true,
  "top_k": 3
}
"""))
    st.append(code("""
200 OK
{
  "model": "logistic_regression", "decision_threshold": 0.553879,
  "trained_on_synthetic_data": true,
  "predictions": [{
    "index": 0, "default_probability": 0.966977,
    "predicted_class": "default", "predicted_label": 1, "risk_band": "High",
    "warnings": [],
    "top_factors": {
      "output_space": "log_odds",
      "increasing_risk": [
        {"feature": "credit_score", "value": 590.0, "shap_value": 1.35973, "effect": "increases_risk"},
        {"feature": "delinquencies_2y", "value": 2.0, "shap_value": 0.525033, "effect": "increases_risk"}
      ],
      "decreasing_risk": [ ... ]
    }
  }],
  "disclaimer": "Predictions are statistical estimates ..."
}
"""))
    st.append(P("Tip for the frontend: call GET /api/model/status first. Its feature_schema lists every input field, its training range "
                "and allowed categories, so the borrower form can be built from it automatically.", SMALL))

    # ---------------- 8 errors
    st.append(P("8. Error handling", H1))
    st.append(P("Every error, from any endpoint, has the same JSON shape, so the frontend needs only one error handler:"))
    st.append(code('{ "error": { "code": "model_not_trained", "message": "No trained model available. Call POST /api/model/train first.", "details": null } }'))
    st.append(table([
        ["HTTP", "code", "Typical cause"],
        ["409", "dataset_not_loaded / model_not_trained / training_in_progress", "Endpoints called in the wrong order, or two trainings at once"],
        ["413", "http_413", "Uploaded file larger than the limit (20 MB default)"],
        ["415", "http_415", "File is not .csv or has a wrong content type"],
        ["422", "request_validation_error", "JSON body has wrong fields or types (Pydantic)"],
        ["422", "data_validation_error / insufficient_data", "Wrong target column, unclear labels, fewer than 50 rows or fewer than 10 per class"],
        ["422", "invalid_input", "Borrower record missing fields or with wrong types; details list each field"],
        ["500", "internal_error", "Unexpected bug; details are logged on the server, not exposed"],
    ], [40, 190, CONTENT_W - 230]))
    st.append(Spacer(1, 10))

    # ---------------- 9 metrics
    metrics_section(st)

    # ---------------- 10 run
    st.append(P("10. Run it yourself, step by step", H1))
    st.append(P("All commands are for Windows PowerShell and run from the <b>backend</b> folder."))
    run_steps = [
        ("Create the environment and install packages (once)",
         "python -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"),
        ("(Optional) regenerate the synthetic dataset",
         ".\\.venv\\Scripts\\python.exe scripts\\generate_synthetic_data.py"),
        ("Start the API server (leave this terminal open)",
         ".\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000"),
        ("Open the interactive docs in a browser",
         "http://127.0.0.1:8000/docs"),
        ("Upload the dataset (second terminal)",
         'curl.exe -F "file=@data/synthetic_loans.csv" -F "target_column=loan_outcome" `\n'
         '         -F "positive_labels=Defaulted" http://127.0.0.1:8000/api/data/upload'),
        ("Train the models",
         "Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/model/train -ContentType application/json -Body '{}'"),
        ("Check the metrics",
         "Invoke-RestMethod http://127.0.0.1:8000/api/model/metrics | ConvertTo-Json -Depth 4"),
        ("Predict and explain",
         "Use POST /api/predict and POST /api/explain from /docs with the example body in Section 7."),
        ("Run all automated tests",
         ".\\.venv\\Scripts\\python.exe -m pytest -q"),
        ("Rebuild this PDF after retraining",
         ".\\.venv\\Scripts\\python.exe -m pip install -r docs\\requirements-docs.txt\n.\\.venv\\Scripts\\python.exe docs\\build_architecture_pdf.py"),
    ]
    for i, (title, cmd) in enumerate(run_steps, 1):
        st.append(KeepTogether([P(f"Step {i} - {title}", H3), code(cmd)]))
    st.append(P("Using the real dataset", H2))
    st.append(numbered([
        "Upload the team's CSV with its real target column name and the label(s) that mean default, e.g. target_column=loan_status, positive_labels=Charged Off.",
        "If the target has extra values such as Current or Late, also send negative_labels (e.g. Fully Paid).",
        "Read the summary: check leakage_warnings and excluded_columns, and add any post-outcome fields to exclude_columns.",
        "Train, review /api/model/metrics, then rebuild this PDF so the results section shows the real numbers.",
    ]))

    # ---------------- 11 testing
    st.append(P("11. Testing", H1))
    st.append(P("47 automated tests, all passing. Run with <b>pytest -q</b> (about 5-15 seconds). Each test uses its own temporary folders, "
                "so tests never touch the real model or data."))
    st.append(table([
        ["File", "Covers"],
        ["tests/test_data_processing.py", "CSV loading and rejection, target mapping and label inference, ambiguous labels, ID/leakage/constant detection, "
                                          "numbers stored as text, quality report, statistical leakage, too-small or single-class data."],
        ["tests/test_model_pipeline.py", "Missing-value imputation, categorical encoding, unseen categories, preprocessing fitted on train only, "
                                         "training all models, reproducibility, exact metric values, threshold tuning, probability range 0-1, "
                                         "risk bands, input validation, save/load, SHAP additivity for all 3 models, global importance."],
        ["tests/test_api.py", "Every endpoint: success responses, calling before data/model exists (409), wrong file type (415), "
                              "oversized file (413), invalid bodies (422), invalid borrower data (422), model survives a server restart."],
    ], [140, CONTENT_W - 140]))

    # ---------------- 12 limitations
    st.append(P("12. Limitations and next steps", H1))
    st.append(table([
        ["Limitation", "Impact", "Suggested next step"],
        ["No authentication", "Anyone who can reach the server can upload, train and predict.", "Run only locally, or put it behind the team's auth / API gateway."],
        ["Synthetic data only", "Current metrics are not real-world evidence.", "Upload the real dataset and retrain."],
        ["One dataset and one model at a time, file storage", "No model history; not built for several server workers.", "Add model versioning or a database if the project needs it."],
        ["Fixed hyperparameters", "Models may not be at their best.", "Add cross-validated hyperparameter search."],
        ["Probabilities not calibrated", "0.30 is a relative score, not an exact 30% chance.", "Add probability calibration on validation data."],
        ["Automatic leakage detection is heuristic", "Some post-outcome fields could slip through.", "Domain review of every column with the business team."],
        ["joblib files can run code when loaded", "A malicious model file is a security risk.", "Only load models produced by this service."],
    ], [130, 170, CONTENT_W - 300]))
    st.append(Spacer(1, 8))
    st.append(P("Predictions are statistical estimates based on historical data. They are not guarantees of borrower behaviour "
                "and should support, not replace, human credit decisions and applicable lending regulations.", NOTE))

    doc.build(st, onFirstPage=on_page, onLaterPages=on_page)
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path}")
