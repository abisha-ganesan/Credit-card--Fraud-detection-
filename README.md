# Credit-card--Fraud-detection-
Credit Card Fraud Detection – Data Science Project (Finance Domain) Built a machine learning model using Python to detect fraudulent credit card transactions. Handled class imbalance using undersampling and applied a Random Forest Classifier. Achieved strong accuracy and precision. Tools Used: Python, Pandas, Scikit-learn, Seaborn, Google Colab
# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Step 2: Create synthetic (fake) fraud dataset
np.random.seed(42)

n_samples = 1000

# Generate random data
amount = np.random.randint(100, 10000, n_samples)
old_balance = amount + np.random.randint(0, 5000, n_samples)
new_balance = old_balance - amount

# Generate class: 0 = not fraud, 1 = fraud (5% fraud)
fraud_flags = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])

# Create DataFrame
df = pd.DataFrame({
    "amount": amount,
    "oldbalanceOrg": old_balance,
    "newbalanceOrig": new_balance,
    "Class": fraud_flags
})

# Step 3: Preview
print("Preview of synthetic dataset:")
print(df.head())

# Step 4: Check class distribution
print("\nClass distribution:")
print(df["Class"].value_counts())

sns.countplot(x="Class", data=df)
plt.title("Synthetic Fraud vs Non-Fraud")
plt.show()

# Step 5: Balance the dataset (undersample)
fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0].sample(n=len(fraud), random_state=42)

data_balanced = pd.concat([fraud, normal])
print("\nBalanced class distribution:")
print(data_balanced["Class"].value_counts())

# Step 6: Train/test split
X = data_balanced.drop("Class", axis=1)
y = data_balanced["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 8: Evaluate model
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Purples")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
