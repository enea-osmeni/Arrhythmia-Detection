import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data_arrhythmia.csv', sep=';')

# Clean data
df.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
df.dropna(axis=0, inplace=True)  # Drop rows with missing values

# Convert diagnosis to binary labels
df['diagnosis'] = (df['diagnosis'] > 1).astype(int)

# Split features and target variable
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Convert to NumPy arrays
X = X.astype(float).values
y = y.values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training using Random Forest
model = RandomForestClassifier()

# Define hyperparameters grid for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the model
random_search.fit(X_train, y_train)

# Predictions
y_pred = random_search.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save the model
filename = 'random_forest_model.pkl'
pickle.dump(random_search, open(filename, 'wb'))


# Fit the model
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_

# Create figure
fig, ax = plt.subplots(1, 1)

# Hide axes
ax.axis('tight')
ax.axis('off')

# Create table and add it to the plot
table_data = [[k, v] for k, v in best_params.items()]
table = ax.table(cellText=table_data, colLabels=['Hyperparameter', 'Value'], cellLoc = 'center', loc='center')

# Show plot with the table
plt.show()




# Visualizations

# Confusion Matrix
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))

# Create a heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)

# Add labels and title
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')

# Add tickmarks
class_names = ['0', '1']  # change these as per your labels
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)

plt.show()


# ROC Curve
plt.figure(figsize=(8, 6))
y_prob = random_search.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()