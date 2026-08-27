"""Streamlit interface for the leakage-safe DistilBERT transaction classifier."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
LOCAL_MODEL_DIR = APP_DIR / "distilbert_transaction_classifier_18"
DEFAULT_MODEL_REPO = "samrandt/distilbert-transaction-classifie"
DEFAULT_REVIEW_THRESHOLD = 0.80
MAX_LENGTH = 64
CATEGORIES = [
    "Education",
    "Entertainment",
    "Fees",
    "Gas & Fuel",
    "Groceries",
    "Healthcare",
    "Income",
    "Insurance",
    "Mortgage",
    "Personal Care",
    "Rent",
    "Restaurants",
    "Shopping",
    "Subscription",
    "Transfer",
    "Transportation",
    "Travel",
    "Utilities",
]
EXAMPLES = {
    "Airline ticket": "NORTHSTAR AIRLINES TICKET 4581",
    "Organic market": "GREEN VALLEY ORGANIC MARKET",
    "Payroll deposit": "ACME CORP PAYROLL DIRECT DEP",
    "Electric bill": "SUNRISE ENERGY ELECTRIC BILL ZIPCO",
}


st.set_page_config(
    page_title="AI Transaction Categorizer",
    page_icon="▥",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --navy: #0c2742;
            --navy-2: #123b5d;
            --teal: #0f8f98;
            --teal-soft: #e9f7f7;
            --ink: #10243e;
            --muted: #607086;
            --line: #dbe4ea;
            --surface: #f5f8fa;
            --amber: #b96805;
        }
        .stApp { background: #ffffff; color: var(--ink); }
        [data-testid="stAppViewContainer"] > .main .block-container {
            max-width: 1180px; padding-top: 2.4rem; padding-bottom: 4rem;
        }
        [data-testid="stSidebar"] { background: var(--navy); }
        [data-testid="stSidebar"] * { color: #f5fbff; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: .65rem .75rem; border-radius: 7px; margin: .08rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: rgba(255,255,255,.08);
        }
        [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
        h1, h2, h3 { color: var(--ink); letter-spacing: -.02em; }
        h1 { font-size: 2.15rem !important; line-height: 1.15 !important; margin-bottom: .4rem !important; }
        h2 { font-size: 1.35rem !important; }
        .page-subtitle { color: var(--muted); font-size: 1.02rem; margin: 0 0 1.5rem; max-width: 760px; }
        .metric-row { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; margin: 1.1rem 0 1.6rem; }
        .metric-card { border: 1px solid var(--line); border-radius: 10px; padding: 17px 19px; background: #fff; }
        .metric-value { color: var(--teal); font-size: 1.72rem; line-height: 1; font-weight: 750; }
        .metric-label { color: var(--ink); font-size: .92rem; font-weight: 650; margin-top: .4rem; }
        .metric-note { color: var(--muted); font-size: .75rem; margin-top: .25rem; }
        .result-card { border: 1px solid #b9dddd; background: var(--teal-soft); border-radius: 10px; padding: 1.25rem 1.35rem; min-height: 166px; }
        .result-label { color: var(--muted); font-size: .76rem; letter-spacing: .07em; text-transform: uppercase; font-weight: 700; }
        .result-category { color: var(--teal); font-size: 2rem; font-weight: 760; margin: .42rem 0 .25rem; }
        .result-confidence { color: var(--ink); font-size: 1.05rem; font-weight: 650; }
        .status-good, .status-review { display: inline-block; margin-top: .75rem; border-radius: 6px; padding: .34rem .55rem; font-size: .82rem; font-weight: 650; }
        .status-good { color: #08725f; border: 1px solid #a8d9cd; background: #effaf6; }
        .status-review { color: #8d5000; border: 1px solid #e9c98f; background: #fff8e9; }
        .prediction-row { display: grid; grid-template-columns: 140px 1fr 72px; align-items: center; gap: 12px; margin: .72rem 0; }
        .bar-track { height: 9px; background: #e6ebef; border-radius: 2px; overflow: hidden; }
        .bar-fill { height: 100%; background: var(--teal); }
        .subtle-box { border: 1px solid var(--line); background: var(--surface); border-radius: 9px; padding: 1rem 1.1rem; color: var(--muted); }
        .step-row { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; align-items: center; gap: 12px; margin-top: .7rem; }
        .step { border: 1px solid var(--line); border-radius: 8px; padding: .8rem .9rem; min-height: 74px; }
        .step strong { color: var(--ink); display: block; margin-bottom: .18rem; }
        .step span { color: var(--muted); font-size: .82rem; }
        .model-alert { border-left: 4px solid var(--amber); background: #fff8e9; padding: 1rem 1.1rem; border-radius: 4px; margin: 1rem 0; color: #6b4a20; }
        .sidebar-brand { font-size: 1.07rem; font-weight: 750; margin: .35rem 0 1.5rem; }
        .sidebar-foot { color: #b9c9d6 !important; font-size: .76rem; line-height: 1.5; }
        .stButton > button, .stDownloadButton > button { border-radius: 7px; font-weight: 650; }
        .stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
        [data-testid="stFileUploaderDropzone"] { background: var(--surface); border-color: var(--line); }
        @media (max-width: 800px) {
            [data-testid="stAppViewContainer"] > .main .block-container { padding: 1.4rem 1rem 3rem; }
            h1 { font-size: 1.72rem !important; }
            .metric-row { grid-template-columns: 1fr; }
            .step-row { grid-template-columns: 1fr; }
            .step-arrow { display: none; }
            .prediction-row { grid-template-columns: 105px 1fr 60px; gap: 8px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_secret(name: str, default: str = "") -> str:
    """Read a Streamlit secret when present without failing locally."""
    try:
        return str(st.secrets.get(name, default)).strip()
    except (FileNotFoundError, AttributeError):
        return default


def configured_model_source() -> str:
    model_repo = os.getenv("MODEL_REPO", "").strip() or get_secret("MODEL_REPO")
    if model_repo:
        return model_repo
    if LOCAL_MODEL_DIR.exists():
        return str(LOCAL_MODEL_DIR)
    return DEFAULT_MODEL_REPO


@st.cache_resource(show_spinner="Loading the fine-tuned DistilBERT model…")
def load_model(model_source: str):
    token = os.getenv("HF_TOKEN", "").strip() or get_secret("HF_TOKEN") or None
    tokenizer = AutoTokenizer.from_pretrained(model_source, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_source, token=token)
    if model.config.num_labels != len(CATEGORIES) or [
        model.config.id2label.get(i) for i in range(len(CATEGORIES))
    ] != CATEGORIES:
        raise ValueError("The model must use the project's original 18-category label mapping.")
    model.eval()
    return tokenizer, model


def clean_description(description: str) -> str:
    """Match the preprocessing used during training and notebook inference."""
    return re.sub(
        r"^\s*\[(debit|credit)\]\s*", "", str(description), flags=re.IGNORECASE
    ).strip()


def label_for_id(model, label_id: int) -> str:
    label = model.config.id2label.get(label_id)
    if label is None:
        label = model.config.id2label.get(str(label_id), f"Label {label_id}")
    return str(label)


def predict_descriptions(descriptions: list[str], tokenizer, model) -> pd.DataFrame:
    cleaned = [clean_description(value) for value in descriptions]
    encoded = tokenizer(
        cleaned,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.inference_mode():
        probabilities = torch.softmax(model(**encoded).logits, dim=-1)

    top_k = min(3, probabilities.shape[1])
    top_probabilities, top_ids = torch.topk(probabilities, k=top_k, dim=-1)
    rows: list[dict[str, object]] = []
    for index, original in enumerate(descriptions):
        row: dict[str, object] = {
            "transaction_description": original,
            "model_description": cleaned[index],
            "predicted_category": label_for_id(model, int(top_ids[index, 0])),
            "confidence": float(top_probabilities[index, 0]),
        }
        for rank in range(top_k):
            row[f"top_{rank + 1}_category"] = label_for_id(model, int(top_ids[index, rank]))
            row[f"top_{rank + 1}_confidence"] = float(top_probabilities[index, rank])
        rows.append(row)
    return pd.DataFrame(rows)


def render_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<p class="page-subtitle">{subtitle}</p>', unsafe_allow_html=True)


def render_metrics() -> None:
    st.markdown(
        """
        <div class="metric-row">
          <div class="metric-card"><div class="metric-value">18</div><div class="metric-label">Categories</div><div class="metric-note">Complete final taxonomy</div></div>
          <div class="metric-card"><div class="metric-value">99.93%</div><div class="metric-label">Test accuracy</div><div class="metric-note">Synthetic leakage-safe test set</div></div>
          <div class="metric-card"><div class="metric-value">0.9994</div><div class="metric-label">Macro F1</div><div class="metric-note">All categories weighted equally</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_setup() -> None:
    st.markdown(
        """
        <div class="model-alert"><strong>Model temporarily unavailable</strong><br>
        The app could not load the model from its configured source. Check the sidebar error
        and the model repository's availability. To use a different source, set
        <code>MODEL_REPO = "username/model-name"</code> in Streamlit Community Cloud → App settings → Secrets.</div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_rows(result: pd.Series) -> None:
    rows = []
    for rank in range(1, 4):
        category = result[f"top_{rank}_category"]
        probability = float(result[f"top_{rank}_confidence"])
        width = max(probability * 100, 0.5)
        rows.append(
            f'<div class="prediction-row"><strong>{rank}. {category}</strong>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.2f}%"></div></div>'
            f'<span>{probability:.2%}</span></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def clear_prediction() -> None:
    st.session_state.pop("latest_result", None)


def select_example(description: str) -> None:
    st.session_state.description = description
    clear_prediction()


def predict_page(model_bundle) -> None:
    render_header(
        "AI-Powered Transaction Categorization",
        "Classify short bank-transaction descriptions into a consistent spending taxonomy using a fine-tuned DistilBERT model.",
    )
    render_metrics()

    if "description" not in st.session_state:
        st.session_state.description = EXAMPLES["Airline ticket"]

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.subheader("Classify a transaction")
        description = st.text_input(
            "Transaction description",
            key="description",
            on_change=clear_prediction,
            placeholder="e.g. NORTHSTAR AIRLINES TICKET 4581",
        )
        submitted = st.button(
            "Predict category", type="primary", width="content", disabled=model_bundle is None
        )
        st.caption("Try an example")
        example_columns = st.columns(4)
        for column, (label, example) in zip(example_columns, EXAMPLES.items()):
            with column:
                st.button(
                    label,
                    key=f"example_{label}",
                    width="stretch",
                    on_click=select_example,
                    args=(example,),
                )

    if submitted and description.strip() and model_bundle is not None:
        tokenizer, model = model_bundle
        try:
            st.session_state.latest_result = predict_descriptions(
                [description], tokenizer, model
            ).iloc[0]
        except Exception as exc:  # Surface a useful UI error while preserving server logs.
            st.error(f"Prediction failed: {exc}")

    result = st.session_state.get("latest_result")
    with right:
        if result is None:
            st.markdown(
                '<div class="result-card"><div class="result-label">Prediction</div>'
                '<div class="result-category">Ready when you are</div>'
                '<div class="result-confidence">Enter a description to classify it.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            confidence = float(result["confidence"])
            needs_review = confidence < DEFAULT_REVIEW_THRESHOLD
            status_class = "status-review" if needs_review else "status-good"
            status_text = "Review recommended" if needs_review else "High confidence"
            st.markdown(
                f'<div class="result-card"><div class="result-label">Predicted category</div>'
                f'<div class="result-category">{result["predicted_category"]}</div>'
                f'<div class="result-confidence">{confidence:.2%} confidence</div>'
                f'<div class="{status_class}">{status_text}</div></div>',
                unsafe_allow_html=True,
            )

    if model_bundle is None:
        render_model_setup()
    elif submitted and not description.strip():
        st.warning("Enter a transaction description before predicting.")

    if result is not None:
        probability_column, guidance_column = st.columns([1.45, 1], gap="large")
        with probability_column:
            st.subheader("Top predictions")
            render_prediction_rows(result)
        with guidance_column:
            st.subheader("Decision guidance")
            if float(result["confidence"]) < DEFAULT_REVIEW_THRESHOLD:
                st.warning("Confidence is below 80%. Route this transaction to manual review.")
            else:
                st.success("This prediction is above the 80% auto-categorization threshold.")
            st.caption(
                "Ambiguous merchants such as CVS, Rite Aid, or marketplace payments may still need review even at high confidence."
            )

    st.divider()
    st.subheader("How it works")
    st.markdown(
        """
        <div class="step-row">
          <div class="step"><strong>1 · Clean description</strong><span>Remove leading debit or credit tags while preserving merchant context.</span></div>
          <div class="step-arrow">→</div>
          <div class="step"><strong>2 · DistilBERT</strong><span>Tokenize to 64 tokens and run the fine-tuned classifier.</span></div>
          <div class="step-arrow">→</div>
          <div class="step"><strong>3 · Decision output</strong><span>Return the category, confidence, and review recommendation.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def batch_page(model_bundle) -> None:
    render_header(
        "Batch Classify",
        "Upload a CSV, categorize transaction descriptions, and download an enriched result file.",
    )
    if model_bundle is None:
        render_model_setup()
    uploaded = st.file_uploader("Upload transactions", type=["csv"])
    threshold = st.slider("Manual-review threshold", 0.50, 0.99, DEFAULT_REVIEW_THRESHOLD, 0.01)
    st.caption("The CSV must contain a transaction description column. Processing is limited to 2,000 rows per upload.")

    if uploaded is None:
        sample = pd.DataFrame({"transaction_description": list(EXAMPLES.values())})
        st.download_button(
            "Download sample CSV",
            sample.to_csv(index=False).encode("utf-8"),
            "sample_transactions.csv",
            "text/csv",
        )
        return

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read this CSV: {exc}")
        return
    if frame.empty:
        st.warning("The uploaded CSV has no rows.")
        return
    if len(frame) > 2000:
        st.error("This app accepts up to 2,000 rows per upload.")
        return

    likely_columns = [column for column in frame.columns if "description" in column.lower()]
    default_index = frame.columns.get_loc(likely_columns[0]) if likely_columns else 0
    description_column = st.selectbox(
        "Description column", frame.columns.tolist(), index=int(default_index)
    )
    st.dataframe(frame.head(8), width="stretch", hide_index=True)

    if st.button("Classify file", type="primary", disabled=model_bundle is None):
        values = frame[description_column].fillna("").astype(str).tolist()
        if any(not value.strip() for value in values):
            st.error("Remove or fill blank descriptions before classifying the file.")
            return
        tokenizer, model = model_bundle
        with st.spinner(f"Classifying {len(values):,} transactions…"):
            chunks = []
            for start in range(0, len(values), 64):
                chunks.append(predict_descriptions(values[start : start + 64], tokenizer, model))
            predictions = pd.concat(chunks, ignore_index=True)
        output = frame.copy()
        output["predicted_category"] = predictions["predicted_category"]
        output["confidence"] = predictions["confidence"]
        output["review_recommended"] = output["confidence"] < threshold
        st.session_state.batch_output = output

    if "batch_output" in st.session_state:
        output = st.session_state.batch_output
        review_count = int(output["review_recommended"].sum())
        first, second, third = st.columns(3)
        first.metric("Transactions", f"{len(output):,}")
        second.metric("Categories found", int(output["predicted_category"].nunique()))
        third.metric("Review queue", f"{review_count:,}")
        formatted = output.copy()
        formatted["confidence"] = formatted["confidence"].map(lambda value: f"{value:.2%}")
        st.dataframe(formatted, width="stretch", hide_index=True)
        st.download_button(
            "Download categorized CSV",
            output.to_csv(index=False).encode("utf-8"),
            "categorized_transactions.csv",
            "text/csv",
            type="primary",
        )


def performance_page() -> None:
    render_header(
        "Model Performance",
        "Evaluation results from an untouched, leakage-safe synthetic test set of 6,721 transactions.",
    )
    render_metrics()
    st.info(
        "These results demonstrate the modeling and evaluation workflow on synthetic data; they are not verified production performance on live banking data."
    )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Evaluation summary")
        metrics = pd.DataFrame(
            {
                "Metric": ["Correct predictions", "Macro precision", "Macro recall", "Macro F1", "Weighted F1", "Gas & Fuel F1"],
                "Result": ["6,716 / 6,721", "0.9994", "0.9994", "0.9994", "0.9993", "1.0000"],
            }
        )
        st.dataframe(metrics, width="stretch", hide_index=True)
    with right:
        st.subheader("Leakage-safe split")
        split = pd.DataFrame(
            {
                "Dataset": ["Training", "Validation", "Test"],
                "Records": [31380, 6677, 6721],
                "Share": ["70.08%", "14.91%", "15.01%"],
                "Description overlap": [0, 0, 0],
            }
        )
        st.dataframe(split, width="stretch", hide_index=True)

    st.subheader("What the five errors revealed")
    errors = pd.DataFrame(
        {
            "Description": [
                "Payment Thank You - 2ndA",
                "Rite Aid #5658 Atlanta",
                "PayPal Inst Xfer Bath & Body Works",
                "CVS - San Diego",
                "CVS #55457 Las Vegas",
            ],
            "Actual": ["Transfer", "Shopping", "Shopping", "Shopping", "Healthcare"],
            "Predicted": ["Shopping", "Healthcare", "Personal Care", "Healthcare", "Shopping"],
        }
    )
    st.dataframe(errors, width="stretch", hide_index=True)
    st.caption(
        "The remaining errors concentrate around multi-purpose merchants and ambiguous category boundaries—a taxonomy problem that additional model complexity alone cannot resolve."
    )


def about_page(model_source: str | None) -> None:
    render_header(
        "About the Project",
        "A portfolio case study in NLP classification, data-leakage prevention, taxonomy design, and decision-aware deployment.",
    )
    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.subheader("Business problem")
        st.write(
            "Bank descriptions are short and inconsistent. A reliable classifier can standardize them for budgeting, merchant analytics, personalization, reporting, and lower manual-review effort."
        )
        st.subheader("Modeling approach")
        st.markdown(
            """
            - Fine-tuned `distilbert-base-uncased` across 18 categories
            - Split unique normalized descriptions before assigning rows
            - Selected the best checkpoint using validation macro F1
            - Preserved uncertainty with top-three probabilities and review routing
            """
        )
    with right:
        st.subheader("Category taxonomy")
        category_frame = pd.DataFrame(
            {"Category": CATEGORIES[:9], "Category ": CATEGORIES[9:]}
        )
        st.dataframe(category_frame, width="stretch", hide_index=True)
        st.subheader("Deployment status")
        if model_source:
            st.success("Fine-tuned model source configured.")
        else:
            st.warning("App code ready; fine-tuned model source not configured.")


def main() -> None:
    inject_styles()
    st.sidebar.markdown('<div class="sidebar-brand">▥ &nbsp; TxnCategorizer</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "Navigation",
        ["Predict Transaction", "Batch Classify", "Model Performance", "About"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.markdown(
        '<div class="sidebar-foot">DistilBERT · 18 categories<br>Portfolio demonstration</div>',
        unsafe_allow_html=True,
    )

    source = configured_model_source()
    bundle = None
    if source:
        try:
            bundle = load_model(source)
        except Exception as exc:
            st.sidebar.error("Model could not be loaded.")
            st.sidebar.caption(str(exc))

    if page == "Predict Transaction":
        predict_page(bundle)
    elif page == "Batch Classify":
        batch_page(bundle)
    elif page == "Model Performance":
        performance_page()
    else:
        about_page(source)


if __name__ == "__main__":
    main()
