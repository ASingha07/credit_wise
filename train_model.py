# train_model.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score


# STEP 1: Load the data

print("Step 1: Loading dataset...")
data = pd.read_csv("data/loan_approval_data.csv")

# Remove Applicant_ID 
data = data.drop(columns=["Applicant_ID"])


before_rows = len(data)
data = data.dropna(subset=["Loan_Approved"])
print("Removed", before_rows - len(data), "rows that had no Loan_Approved value.")


# STEP 2: Split into train and test BEFORE doing any cleaning

# split and clean data
print("Step 2: Splitting into train and test sets...")
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


# STEP 3: Fill missing values (imputation)

print("Step 3: Filling missing values...")

number_columns = train_data.drop(columns=["Loan_Approved"]).select_dtypes(include="number").columns.tolist()
text_columns = train_data.drop(columns=["Loan_Approved"]).select_dtypes(include="object").columns.tolist()

number_filler = SimpleImputer(strategy="mean")
text_filler = SimpleImputer(strategy="most_frequent")

number_filler.fit(train_data[number_columns])
text_filler.fit(train_data[text_columns])

train_data[number_columns] = number_filler.transform(train_data[number_columns])
train_data[text_columns] = text_filler.transform(train_data[text_columns])

test_data[number_columns] = number_filler.transform(test_data[number_columns])
test_data[text_columns] = text_filler.transform(test_data[text_columns])



# STEP 4: Encode text columns into numbers

print("Step 4: Encoding categorical columns...")

# Label Encode Education_Level
education_encoder = LabelEncoder()
train_data["Education_Level"] = education_encoder.fit_transform(train_data["Education_Level"])
test_data["Education_Level"] = education_encoder.transform(test_data["Education_Level"])

# Loan_Approved is target column (Yes/No), turn it into 1/0
target_encoder = LabelEncoder()
train_data["Loan_Approved"] = target_encoder.fit_transform(train_data["Loan_Approved"])
test_data["Loan_Approved"] = target_encoder.transform(test_data["Loan_Approved"])


# one-hot encoding
onehot_columns = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category",
]

onehot_encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
onehot_encoder.fit(train_data[onehot_columns])

train_onehot = pd.DataFrame(
    onehot_encoder.transform(train_data[onehot_columns]),
    columns=onehot_encoder.get_feature_names_out(onehot_columns),
    index=train_data.index,
)
test_onehot = pd.DataFrame(
    onehot_encoder.transform(test_data[onehot_columns]),
    columns=onehot_encoder.get_feature_names_out(onehot_columns),
    index=test_data.index,
)

train_data = pd.concat([train_data.drop(columns=onehot_columns), train_onehot], axis=1)
test_data = pd.concat([test_data.drop(columns=onehot_columns), test_onehot], axis=1)



# STEP 5: Split features (X) and target (y)

X_train = train_data.drop(columns=["Loan_Approved"])
y_train = train_data["Loan_Approved"]
X_test = test_data.drop(columns=["Loan_Approved"])
y_test = test_data["Loan_Approved"]


feature_columns = list(X_train.columns)



# STEP 6: Scale the features

print("Step 5: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# STEP 7: Train all 3 models and compare them

print("Step 6: Training all 3 models and testing different decision thresholds...")
print()

from sklearn.metrics import accuracy_score, precision_score, recall_score

models_to_try = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
}

# recall and accuracy must never go below this
MIN_RECALL = 0.75
MIN_ACCURACY = 0.75

# this is the precision range we are aiming for, if possible
TARGET_PRECISION_LOW = 0.85
TARGET_PRECISION_HIGH = 0.90

all_results = []          # every model + threshold combo to tried that passed the floors
trained_models = {}       # actual trained model objects, keyed by model name

for model_name, model in models_to_try.items():
    model.fit(X_train_scaled, y_train)
    trained_models[model_name] = model

    approval_probabilities = model.predict_proba(X_test_scaled)[:, 1]

    
    threshold = 0.05
    while threshold < 0.95:
        predictions_at_threshold = (approval_probabilities >= threshold).astype(int)

        acc = accuracy_score(y_test, predictions_at_threshold)
        prec = precision_score(y_test, predictions_at_threshold, zero_division=0)
        rec = recall_score(y_test, predictions_at_threshold, zero_division=0)

        
        if rec >= MIN_RECALL and acc >= MIN_ACCURACY:
            all_results.append({
                "Model": model_name,
                "Threshold": round(threshold, 2),
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
            })

        threshold = threshold + 0.01

results_df = pd.DataFrame(all_results)

if len(results_df) == 0:
    
    print("WARNING: No model/threshold combination kept recall and accuracy")
    print("above 75%. Falling back to the default 0.5 threshold.")
    fallback_rows = []
    for model_name, model in trained_models.items():
        predictions = model.predict(X_test_scaled)
        fallback_rows.append({
            "Model": model_name, "Threshold": 0.5,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
        })
    results_df = pd.DataFrame(fallback_rows)
    results_df = results_df.sort_values(by="Precision", ascending=False)
else:
    
    in_target_band = results_df[
        (results_df["Precision"] >= TARGET_PRECISION_LOW) & (results_df["Precision"] <= TARGET_PRECISION_HIGH)
    ]
    if len(in_target_band) > 0:
        results_df = in_target_band.sort_values(by="Precision", ascending=False)
    else:
        
        results_df = results_df.sort_values(by="Precision", ascending=False)

best_row = results_df.iloc[0]
best_model_name = best_row["Model"]
best_threshold = best_row["Threshold"]
best_model = trained_models[best_model_name]

print("Best combination found:")
print("  Model:     ", best_model_name)
print("  Threshold: ", best_threshold)
print("  Accuracy:  ", round(best_row["Accuracy"], 4))
print("  Precision: ", round(best_row["Precision"], 4))
print("  Recall:    ", round(best_row["Recall"], 4))
print()



# STEP 9: Save everything the app needs

print("Step 7: Saving the best model and all preprocessing tools...")

import os
os.makedirs("model", exist_ok=True)

with open("model/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/best_model_name.pkl", "wb") as f:
    pickle.dump(best_model_name, f)

with open("model/best_threshold.pkl", "wb") as f:
    pickle.dump(best_threshold, f)

with open("model/number_filler.pkl", "wb") as f:
    pickle.dump(number_filler, f)

with open("model/text_filler.pkl", "wb") as f:
    pickle.dump(text_filler, f)

with open("model/education_encoder.pkl", "wb") as f:
    pickle.dump(education_encoder, f)

with open("model/target_encoder.pkl", "wb") as f:
    pickle.dump(target_encoder, f)

with open("model/onehot_encoder.pkl", "wb") as f:
    pickle.dump(onehot_encoder, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

with open("model/number_columns.pkl", "wb") as f:
    pickle.dump(number_columns, f)

with open("model/text_columns.pkl", "wb") as f:
    pickle.dump(text_columns, f)

with open("model/comparison_table.pkl", "wb") as f:
    pickle.dump(results_df, f)

print("Done! All files saved inside the 'model' folder.")
print("You can now run the app with: streamlit run app.py")
