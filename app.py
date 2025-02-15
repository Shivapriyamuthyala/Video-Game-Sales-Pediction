import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Generate synthetic loan approval dataset
np.random.seed(42)
num_samples = 1000
credit_score = np.random.randint(300, 850, num_samples)
income = np.random.uniform(20000, 100000, num_samples)
loan_amount = np.random.uniform(1000, 50000, num_samples)
loan_term = np.random.randint(12, 60, num_samples)
employment_years = np.random.randint(1, 20, num_samples)
loan_approval = np.random.choice([0, 1], num_samples)  # 0: Not approved, 1: Approved
# Create a DataFrame from the generated data
data = pd.DataFrame({
    'Credit_Score': credit_score,
    'Income': income,
    'Loan_Amount': loan_amount,
    'Loan_Term': loan_term,
    'Employment_Years': employment_years,
    'Loan_Approval': loan_approval
})
# Data visualization
sns.pairplot(data, hue='Loan_Approval', diag_kind='kde')
plt.show()
sns.countplot(x='Loan_Approval', data=data)
plt.title('Loan Approval Distribution')
plt.show()
# Data preprocessing: Select relevant features and handle missing values
X = data[['Credit_Score', 'Income', 'Loan_Amount', 'Loan_Term', 'Employment_Years']]
y = data['Loan_Approval']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize and train a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{classification_report}')