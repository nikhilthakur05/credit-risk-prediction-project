# importing libraries
import pandas as pd      
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Loading Dataset
df = pd.read_csv("credit_risk_dataset.csv")
print("Dataset Loaded Successfully")

# dataset overview
print("\nShape of Dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

#Data Visualization
#Loan Status Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='loan_status', data=df)
plt.title("Loan Status Distribution")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(
    df.corr(numeric_only=True),
    annot=True
)
plt.title("Correlation Heatmap")
plt.show()

#Cleaning Data
df = df.dropna()

# Label Encoding
le = LabelEncoder()

df["person_home_ownership"] = le.fit_transform(
    df["person_home_ownership"]
)

df["loan_intent"] = le.fit_transform(
    df["loan_intent"]
)

df["loan_grade"] = le.fit_transform(
    df["loan_grade"]
)

df["cb_person_default_on_file"] = le.fit_transform(
    df["cb_person_default_on_file"]
)

# Features & Target
X = df.drop("loan_status", axis=1)

y = df["loan_status"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)
print("Model Training Completed")

# Precision Recall F1
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)
print("Classification Report Completed")

# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred
)
plt.figure(figsize=(7,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    linewidths=1,
    cbar=True
)

plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Actual", fontsize=12)

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)
print("Confusion Matrix Saved")


# Feature Importance Analysis
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)

plt.figure(figsize=(12,6))

plt.barh(
    importance['Feature'],
    importance['Importance']
)

plt.title("Feature Importance Analysis", fontsize=16)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)

plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)
print("Feature Importance Graph Saved")

plt.savefig("feature_importance.png")
print("Feature Importance Graph Saved")

top_features = importance.head(5)

plt.figure(figsize=(8,5))

plt.pie(
    top_features["Importance"],
    labels=top_features["Feature"],
    autopct="%1.1f%%"
)

plt.title("Top 5 Important Features")

plt.savefig("top_features_pie_chart.png", dpi=300)

print("Pie Chart Saved")

# Accuracy Graph
plt.figure(figsize=(5,4))

plt.bar(
    ["Accuracy"],
    [accuracy]
)

plt.ylim(0,1)

plt.title("Model Accuracy", fontsize=16)
plt.ylabel("Score")

plt.tight_layout()

plt.savefig("accuracy_graph.png", dpi=300)

print("Accuracy Graph Saved")

# print("Model executed Successfully")


new_customer = [[
    25,      # person_age
    50000,   # person_income
    2,       # person_home_ownership
    3,       # person_emp_length
    4,       # loan_intent
    0,       # loan_grade
    10000,   # loan_amnt
    10.5,    # loan_int_rate
    0.20,    # loan_percent_income
    0,       # cb_person_default_on_file
    4        # cb_person_cred_hist_length
]]

prediction = model.predict(new_customer)

print("\nPrediction:", prediction)