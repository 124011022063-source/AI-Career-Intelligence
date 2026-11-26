# AI Career Intelligence & Skill Development Platform

A Streamlit educational prototype that assesses technical and soft skills, compares career-role requirements, trains multiple classification algorithms on a bundled **synthetic demonstration dataset**, and creates a personalized learning roadmap.

## Features
- Technical skill self-assessment
- Soft skill self-assessment: communication, teamwork, problem solving, critical thinking, adaptability, time management, leadership and creativity
- Career recommendations from ML classifiers
- Weighted technical + soft skill gap analysis
- Random Forest, Decision Tree, K-Nearest Neighbors and Logistic Regression comparison
- Accuracy, macro precision, macro recall, macro F1-score and confusion matrix
- Technical and soft-skill learning hub with practice activities
- Personalized skill-development roadmap
- Downloadable TXT report

## Important data limitation
`data/demo_career_dataset.csv` is **synthetic demonstration data** generated from illustrative career skill profiles. It is included so the project can run without an external dataset. Model metrics are only a demonstration of the training/evaluation workflow. They must not be described as real-world or employer-validated accuracy.

For academic research, replace this file with a properly licensed, representative dataset containing the 18 skill columns plus a `Career` target column. Do not include personal identifiers. Re-run the app after replacing the CSV. Ensure each career class has enough examples for stratified train/test splitting.

## Run locally (Windows / VS Code)
1. Extract the ZIP.
2. Open the extracted `AI_Career_Intelligence_SoftSkills` folder in VS Code.
3. Open Terminal > New Terminal.
4. Run:
   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```
   If PowerShell blocks activation, skip activation and use:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
5. Open the local URL printed in the terminal, usually `http://localhost:8501`.

Python 3.12 is recommended for compatibility. If `py -3.12` is unavailable, install Python 3.12 from https://www.python.org/downloads/.

## GitHub and deployment
Upload the contents of this folder (so `app.py` and `requirements.txt` are in the repository root) to a GitHub repository. Then create a Streamlit Community Cloud app using:
- Repository: your GitHub repository
- Branch: `main`
- Main file path: `app.py`

## Algorithms
- Random Forest Classifier
- Decision Tree Classifier
- K-Nearest Neighbors Classifier
- Logistic Regression

The app reports held-out test metrics. Since the bundled dataset is synthetic, use the metrics only to demonstrate the evaluation pipeline.
