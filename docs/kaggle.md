# Optional Kaggle CPU training

Local training is sufficient. Kaggle is an optional execution environment, not a required model component. No hosted training job has been submitted.

1. Create a private Kaggle input dataset containing this project directory. Include `flagger/`, `scripts/`, `data/processed/`, `notebooks/` and `docs/`; raw downloads are unnecessary.
2. Create a Python notebook, attach that input, select **CPU**, and import `notebooks/bsc_demo.ipynb`.
3. Run the cells. The notebook locates a project root containing `data/processed/train.csv`, uses the installed scikit-learn packages and writes trained models/reports to `/kaggle/working/flagger-output` if the source is read-only.
4. Download `models/staleness.joblib`, `models/metadata.json` and the reports from the output folder. Use an environment with the same scikit-learn version to load the artifact, or retrain locally from the prepared CSVs.

The notebook trains TF-IDF classifiers. It does not fine-tune an LLM or download a pretrained checkpoint, and it can run with internet disabled once project files are attached. The web app is intended for local execution; Kaggle is used only for the training demonstration.
