# Streamlit deployment

The app supports three model locations, in this priority order:

1. A source configured with the `MODEL_REPO` environment variable or Streamlit secret.
2. A local `distilbert_transaction_classifier_18/` folder beside `streamlit_app.py`.
3. The default public model: [`samrandt/distilbert-transaction-classifie`](https://huggingface.co/samrandt/distilbert-transaction-classifie).

The uploaded model weights are approximately 268 MB, so Hugging Face hosts the model separately from the app's GitHub code. The public default does not need a token or Streamlit secrets.

## 1. Upload the exported model

The exported files have been uploaded at the repository root: `config.json`, `model.safetensors`, `tokenizer.json`, and `tokenizer_config.json`. The vocabulary is included in `tokenizer.json`; a separate `vocab.txt` is not required for this fast tokenizer. The ZIP archive and `training_args.bin` are not needed for inference.

The saved `config.json` must retain the 18-category `id2label` and `label2id` mappings from training.

## 2. Test locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If the model repository is private, also export `HF_TOKEN`.

## 3. Deploy on Streamlit Community Cloud

- Repository: `Samrandd/Portfolio`
- Branch: `master`
- Main file: `ai-powered-bank-transaction-categorization/streamlit_app.py`
- Python: 3.12

First upload `streamlit_app.py`, `requirements.txt`, and this deployment guide into the GitHub project folder. Then deploy using the settings above. The default model is already configured in the app.

Optional: in **App settings → Secrets**, override the model source:

```toml
MODEL_REPO = "samrandt/distilbert-transaction-classifie"
# HF_TOKEN = "hf_..." # private repositories only
```

Never commit a Hugging Face access token to GitHub.
