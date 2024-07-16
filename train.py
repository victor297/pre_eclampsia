import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Load the data from maternity.csv
data = pd.read_csv('maternity.csv')

# Encode the target variable
label_encoder = LabelEncoder()
data['RiskLevel'] = label_encoder.fit_transform(data['RiskLevel'])

# Split the data into features and target variable
X = data.drop('RiskLevel', axis=1)
y = data['RiskLevel']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
eval_set = [(X_train, y_train), (X_test, y_test)]

# Train the XGBoost model
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# Save the trained model to a file
joblib.dump(model, 'pre_eclampsia_model.joblib')

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Plot training and validation log loss
results = model.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

# Plot log loss
fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.savefig('xgb_log_loss.png')

# Plot feature importance
plt.figure()
plot_importance(model)
plt.savefig('xgb_feature_importance.png')
