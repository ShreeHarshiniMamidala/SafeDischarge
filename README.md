SafeDischarge: Discharge Instruction Quality Analyzer
SafeDischarge is an intermediate healthcare machine learning project that evaluates written discharge instructions for quality, readability, and completeness. The project combines NLP-based text processing, rule-based clinical content checks, and a supervised ML classifier to help identify whether discharge instructions are likely to be low, medium, or high quality.

This project is inspired by the idea that discharge from hospital to home is a vulnerable transition period, and poor communication during discharge can increase the risk of adverse events and readmissions.

Problem Statement
Hospital discharge instructions are often the patient's main guide for medications, home care, follow-up, and warning signs after leaving the hospital. If these instructions are vague, incomplete, or too hard to read, patients may misunderstand what to do next, which can lead to unsafe transitions of care.

The AHRQ IDEAL discharge framework highlights five key areas that should be covered during discharge planning: what life at home will be like, medication review, warning signs, test results, and follow-up appointments.
​ In parallel, prior research has shown that templated discharge instructions can improve readability, with one study reporting an average grade level of 5.6 for templated instructions compared with 7.6 for clinician-generated instructions.
​

This project addresses the following question:

Can a machine learning system automatically analyze discharge instructions and estimate whether they are clear, complete, and patient-friendly?

Proposed Solution
SafeDischarge analyzes discharge instruction text through three components:

Readability analysis using common text difficulty metrics.

Completeness analysis using rule-based detection of key discharge elements.

Quality prediction using a machine learning pipeline trained on labeled examples of discharge instructions.

The app presents these outputs in a Streamlit dashboard so that users can paste instructions and immediately view the predicted quality label, readability values, and coverage of discharge elements.

Project Goals
Build a healthcare NLP project at an intermediate difficulty level.

Use ML to classify discharge instruction quality.

Use healthcare-inspired rules based on the IDEAL framework.
​

Create a visual dashboard for demonstration and portfolio use.

Make the project easy to run locally and easy to explain on GitHub.

Features
Predicts discharge instruction quality as low, medium, or high.

Computes readability metrics such as grade level and reading ease.

Detects whether key discharge elements are present:

Medication review

Follow-up guidance

Warning signs

Home-care instructions

Test-result explanation

Provides interactive visualizations in Streamlit using Plotly.

Includes synthetic dataset generation and public-source text collection.

How It Works
1. Data Collection
The project uses a combination of:

synthetic discharge instruction examples,

public patient education pages from MedlinePlus,

and discharge-planning guidance from AHRQ.

MedlinePlus provides downloadable health information resources, including XML files and patient-facing educational content.
​ AHRQ provides guidance for discharge planning and patient-family engagement through IDEAL discharge planning resources.

2. Readability Module
The readability module estimates how difficult the instruction is to understand. This is important because patient-facing materials are safer when they are written at a lower reading level, and prior studies have shown that templated discharge instructions can improve readability.

3. Completeness Module
The completeness checker looks for whether the instruction includes core elements recommended by AHRQ IDEAL discharge planning:

life at home,

medications,

warning signs,

test results,

and follow-up appointments.
​

4. ML Quality Model
A text classification model is trained using scikit-learn. The model uses the instruction_text field as input, converts the text into TF-IDF features, and predicts one of three quality labels:

low

medium

high

5. Streamlit Dashboard
The frontend allows the user to paste discharge instructions and see:

predicted quality,

readability summary,

completeness score,

and visual charts.

Streamlit apps are typically run with the streamlit run command pointing to the app file.
​

Project Structure
bash
Self_discharge_project/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── models/
│   ├── quality_model.joblib
│   └── label_encoder.joblib
├── notebooks/
│   └── 01_dataset_bootstrap.ipynb
├── src/
│   ├── 01_download_public_data.py
│   ├── 03_create_synthetic_dataset.py
│   ├── 05_train_quality_model.py
│   ├── 06_evaluate_model.py
│   ├── 07_predict.py
│   ├── completeness.py
│   ├── readability.py
│   └── utils.py
├── requirements.txt
└── README.md
Tech Stack
Python

pandas for data handling

scikit-learn for ML pipeline and classification

joblib for saving models

Streamlit for the app UI

Plotly for dashboard visualizations

Installation
1. Clone the repository
bash
git clone <your-repo-url>
cd Self_discharge_project
2. Create a virtual environment
Windows (PowerShell):

bash
python -m venv venv
venv\Scripts\activate
macOS/Linux:

bash
python -m venv venv
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
How to Run the Project
Run the project in this order.

Step 1: Download public text sources
bash
python src/01_download_public_data.py
This step downloads public healthcare-related web pages used as reference content.

Step 2: Create the synthetic dataset
bash
python src/03_create_synthetic_dataset.py
This creates the labeled dataset used for the first version of the ML model.

Step 3: Train the model
bash
python src/05_train_quality_model.py
This trains the classifier and saves the model files into the models/ folder.

Step 4: Evaluate the model
bash
python src/06_evaluate_model.py
This prints evaluation metrics for the trained model.

Step 5: Test a single prediction from the terminal
bash
python src/07_predict.py
Step 6: Launch the Streamlit dashboard
bash
streamlit run app/streamlit_app.py
This starts the local web app in your browser.
​

Example Use Case
Input:

Take your antibiotics as prescribed. Drink plenty of fluids. Follow up with your doctor in one week. Seek urgent care if breathing gets worse.

Expected output:

Readability metrics

Completeness flags

Predicted quality label

Dashboard visualizations

Model Labels
The project predicts three quality levels:

Label	Meaning
Low	Instructions are vague, incomplete, or hard to act on.
Medium	Instructions are somewhat useful but missing important details.
High	Instructions are clear, complete, and actionable.
Why This Project Matters
Transitions of care are a known safety risk area, and discharge planning quality can influence whether patients understand medications, follow-up, and warning signs after leaving the hospital.
 This project demonstrates how NLP and ML can be used to support healthcare communication quality in a simple, interpretable, and portfolio-ready format.

Limitations
The first version uses synthetic labeled data rather than a large real hospital discharge corpus.

The completeness module is rule-based and keyword-driven.

The quality model depends on the quality and diversity of the training examples.

This is a research/demo tool and should not be used for clinical decision-making.

Future Improvements
Expand the dataset with more public patient instruction sources.

Add predict_proba() confidence visualization.

Add explanation of which words or phrases influenced the prediction.

Integrate a RAG module using public patient education documents.

Replace synthetic labels with a more realistic annotated dataset.

References
The project is conceptually grounded in the following public resources:

AHRQ IDEAL Discharge Planning guidance.
​

PSNet overview of discharge planning and transitions of care.
​

Readability studies on discharge instructions and template-based improvement.

Streamlit CLI documentation for app execution.
