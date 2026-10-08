# Optional Kaggle CPU training

Local training is sufficient. Kaggle is an optional execution environment, not a required model component. No hosted training job has been submitted.

1. Create a private Kaggle input dataset containing this project directory. Include `flagger/`, `scripts/`, `data/processed/`, `data/paraphrase_check.tsv`, `models/semantic_encoder/`, `notebooks/` and `docs/`; raw downloads are unnecessary. Encoder assets are ignored by Git, so include that directory explicitly when preparing the input.
2. Create a Python notebook, attach that input, select **CPU**, and import `notebooks/bsc_demo.ipynb`.
3. Install missing runtime packages from `requirements.txt` if necessary, then run the cells. For fully offline execution, these packages must already be installed. The notebook finds the project root and copies supplied encoder assets into the writable output directory before training. Results go to `/kaggle/working/flagger-output` if the source is read-only.
4. Download the full output `models/` directory, including `semantic_encoder/`, and the reports. Keep the runtime versions compatible with the recorded metadata, or retrain locally from the prepared CSVs.

The notebook compares fixed MiniLM embeddings, lexical features and their combination. It does not fine-tune the encoder. It runs offline when packages and encoder assets are provided; otherwise enable internet for the one-time pinned checkpoint download. The web app is intended for local execution; Kaggle is used only for the training demonstration.
