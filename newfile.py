import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("car_evaluation.csv")

print(df.head())

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDataset information:")
print(df.info())


# Clean column names
df.columns = df.columns.str.strip()

print("Columns after cleaning:")
print(df.columns)


# Remove duplicate rows
df.drop_duplicates(inplace=True)

print("\nShape after removing duplicates:")
print(df.shape)


# Check target values
print("\nCar Evaluation values:")
print(df["unacc"].value_counts())


# Check missing values
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Target distribution
sns.countplot(data=df, x="unacc")

plt.title("Car Evaluation Distribution")
plt.xlabel("Car Evaluation")
plt.ylabel("Number of Cars")
plt.show()


# Buying price distribution
sns.countplot(data=df, x="vhigh")

plt.title("Buying Price Distribution")
plt.xlabel("Buying Price")
plt.ylabel("Number of Cars")
plt.show()


# Safety vs Car Evaluation
sns.countplot(
    data=df,
    x="low",
    hue="unacc"
)

plt.title("Safety vs Car Evaluation")
plt.xlabel("Safety")
plt.ylabel("Number of Cars")
plt.show()


# Maintenance price vs Car Evaluation
sns.countplot(
    data=df,
    x="vhigh.1",
    hue="unacc"
)

plt.title("Maintenance Price vs Car Evaluation")
plt.xlabel("Maintenance Price")
plt.ylabel("Number of Cars")
plt.show()


# Separate input and target
X = df.drop("unacc", axis=1)
y = df["unacc"]

print("\nInput features:")
print(X.columns)

print("\nTarget:")
print(y.name)


# Encode categorical input features
X = X.apply(LabelEncoder().fit_transform)

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("\nEncoded input features:")
print(X.head())

print("\nEncoded target:")
print(y[:10])


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# Standard scaling for KNN and SVM
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------
# K-NEAREST NEIGHBORS
# ------------------------------------------------

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print("\nK-Nearest Neighbors")

print("Accuracy:",
      accuracy_score(y_test, y_pred_knn))

print("Precision:",
      precision_score(y_test, y_pred_knn, average="weighted"))

print("Recall:",
      recall_score(y_test, y_pred_knn, average="weighted"))

print("F1 Score:",
      f1_score(y_test, y_pred_knn, average="weighted"))


# ------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\nRandom Forest")

print("Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("Precision:",
      precision_score(y_test, y_pred_rf, average="weighted"))

print("Recall:",
      recall_score(y_test, y_pred_rf, average="weighted"))

print("F1 Score:",
      f1_score(y_test, y_pred_rf, average="weighted"))


# ------------------------------------------------
# SUPPORT VECTOR MACHINE
# ------------------------------------------------

svm = SVC()

svm.fit(X_train_scaled, y_train)

y_pred_svm = svm.predict(X_test_scaled)

print("\nSupport Vector Machine")

print("Accuracy:",
      accuracy_score(y_test, y_pred_svm))

print("Precision:",
      precision_score(y_test, y_pred_svm, average="weighted"))

print("Recall:",
      recall_score(y_test, y_pred_svm, average="weighted"))

print("F1 Score:",
      f1_score(y_test, y_pred_svm, average="weighted"))


# ------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------

results = pd.DataFrame({

    "Model": [
        "K-Nearest Neighbors",
        "Random Forest",
        "Support Vector Machine"
    ],

    "Accuracy": [
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_svm)
    ],

    "Precision": [
        precision_score(y_test, y_pred_knn, average="weighted"),
        precision_score(y_test, y_pred_rf, average="weighted"),
        precision_score(y_test, y_pred_svm, average="weighted")
    ],

    "Recall": [
        recall_score(y_test, y_pred_knn, average="weighted"),
        recall_score(y_test, y_pred_rf, average="weighted"),
        recall_score(y_test, y_pred_svm, average="weighted")
    ],

    "F1 Score": [
        f1_score(y_test, y_pred_knn, average="weighted"),
        f1_score(y_test, y_pred_rf, average="weighted"),
        f1_score(y_test, y_pred_svm, average="weighted")
    ]
})


print("\nModel Comparison:")
print(results)


# Accuracy comparison graph
sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.title("Model Accuracy Comparison")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.show()


# ------------------------------------------------
# CONFUSION MATRIX - KNN
# ------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred_knn
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ------------------------------------------------
# CONFUSION MATRIX - RANDOM FOREST
# ------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred_rf
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ------------------------------------------------
# CONFUSION MATRIX - SVM
# ------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred_svm
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ------------------------------------------------
# CLASSIFICATION REPORTS
# ------------------------------------------------

print("\nK-Nearest Neighbors")
print(
    classification_report(
        y_test,
        y_pred_knn,
        target_names=label_encoder.classes_
    )
)


print("\nRandom Forest")
print(
    classification_report(
        y_test,
        y_pred_rf,
        target_names=label_encoder.classes_
    )
)


print("\nSupport Vector Machine")
print(
    classification_report(
        y_test,
        y_pred_svm,
        target_names=label_encoder.classes_
    )
)


# ------------------------------------------------
# BEST MODEL
# ------------------------------------------------

best_model = results.loc[
    results["Accuracy"].idxmax()
]

print("\nBest Model:")
print(best_model)


print(
    "\nThe models were used to classify cars "
    "into different evaluation categories."
)

print(
    "K-Nearest Neighbors, Random Forest and Support "
    "Vector Machine were compared using Accuracy, "
    "Precision, Recall and F1 Score."
)

print(
    "The model with the highest accuracy is:",
    best_model["Model"]
)