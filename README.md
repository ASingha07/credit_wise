# CreditWise - Loan Approval Predictor

A machine learning app that predicts if a loan gets approved or rejected.

## What This Does

You give it loan applicant details (age, income, credit score, etc.) and it tells you whether the loan should be approved or rejected. The predictions are based on a model trained on real historical loan data.

## What You Need

- Python 3.7+
- That's it. Everything else gets installed automatically.

Check if you have Python:
```bash
python --version
```

## Setup

### Step 1: Install Packages

```bash
pip install -r requirements.txt
```

That's it. You now have pandas, scikit-learn, and streamlit.

### Step 2: Train the Model

```bash
python train_model.py
```

This:
- Loads the loan data from `data/loan_approval_data.csv`
- Cleans and prepares it
- Trains 3 different models (Logistic Regression, KNN, Naive Bayes)
- Picks the best one and saves it to the `model/` folder

The output will show you which model won and why.

### Step 3: Run the App

```bash
streamlit run app.py
```

A browser window opens automatically at `http://localhost:8501`. If it doesn't, just visit that URL manually.

## Using the App

1. Fill in the form with loan applicant details
2. Click "Check Loan Approval"
3. See if it's approved or rejected

Done.

## Folder Layout

```
credit_wise/
├── app.py                       # Run this to use the app
├── train_model.py              # Run this first to train the model
├── requirements.txt            # List of packages to install
├── README.md                   # This file
├── data/
│   └── loan_approval_data.csv  # The training data
└── model/                      # Auto-created after training
    ├── best_model.pkl
    ├── best_model_name.pkl
    ├── scaler.pkl
    ├── encoders...
    └── (other saved tools)
```

**What each file does:**
- `train_model.py` - Trains the model once and saves everything to `model/`
- `app.py` - Loads the saved model and runs the web interface
- Data in `data/` is used only during training
- Everything in `model/` is created automatically

## How It Works

### Training Process

1. Load the CSV file
2. Clean up missing values
3. Convert text categories to numbers
4. Normalize numerical values
5. Train 3 models side-by-side
6. Compare accuracy, precision, recall
7. Save the best one

### Why Precision Matters

We pick the model with the highest **Precision** because:
- Approving a bad loan is worse than rejecting a good one
- Precision = "out of all loans we approved, how many were actually good?"
- This keeps false approvals low

### The 3 Models

- **Logistic Regression**: Fast and simple
- **KNN**: Good for non-linear patterns
- **Naive Bayes**: Works well with limited data

Whichever has the best precision score wins and gets saved.

## Common Tasks

### First time?

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

### Got new data?

Just run training again:
```bash
python train_model.py
streamlit run app.py
```

### Different port?

```bash
streamlit run app.py --server.port 8502
```

## Problems?

**Model not found** → Run `python train_model.py` first

**Streamlit not installed** → Run `pip install -r requirements.txt`

**Port 8501 already in use** → Use a different port: `streamlit run app.py --server.port 8502`

**Data file not found** → Make sure `data/loan_approval_data.csv` exists

**Can't find the data file when running** → Make sure you're in the right folder. Run from the project root directory.

**Dropdown options are wrong** → After changing the CSV data, run `python train_model.py` again, then update the dropdowns in `app.py` manually if needed
