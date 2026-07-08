# World Happiness Predictor Project
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

df = pd.read_csv("World Happiness Report.csv")

"""print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())"""

#Handle missing values
df['Job Satisfaction'] = df['Job Satisfaction'].fillna(df['Job Satisfaction'].mean())

#Create a targetvariable: Happy=1 if above median happiness score, else 0
median_score = df['Happiness Score'].median()
df['Happy'] = (df['Happiness Score'] > median_score).astype(int)
print(df['Happy'].value_counts())

#Encode Region(text) into numbers
le = LabelEncoder()
df['Region_Encoded'] = le.fit_transform(df['Region'])

#Select features(X) and target (y)
features= ['Economy', 'Family', 'Health', 'Freedom', 'Corruption', 'Generosity','Job Satisfaction', 'Region_Encoded']
X = df[features]
y = df['Happy']

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)
#Predict and check accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy*100, "%")

#Decision Tree Model
dt_model = DecisionTreeClassifier(max_depth=3,random_state=42)
dt_model.fit(X_train_scaled, y_train)
# Predict and check accuracy
y_pred_dt = dt_model.predict(X_test_scaled)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

#Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
#Predict and check accuracy
y_pred_rf = rf_model.predict(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

#K-Nearest Neighbors Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
#Predict and check accuracy
y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)


print("\n----Model Comparison----")
print("Logistic Regression Accuracy:", accuracy*100, "%")
print("Decision Tree Accuracy:", accuracy_dt*100, "%")
print("Random Forest Accuracy:", accuracy_rf*100, "%")
print("KNN Accuracy:", accuracy_knn*100, "%")

#Confusion Matrix
cm=confusion_matrix(y_test, y_pred)
print("\n----Confusion Matrix (Logistic Regression)----")
print(cm)

#Precision, Recall, F1 Score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n----Classification Metrics (Logistic Regression)----")
print("Precision:", round(precision*100, 2), "%")
print("Recall:", round(recall*100, 2), "%")
print("F1 Score:", round(f1*100, 2), "%")

#Or get all of them at once in a clean report
print("\n----Full Classification Report (Logistic Regression)----")
print(classification_report(y_test, y_pred))

#Cross-validation for Logistic Regression
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("\n----Cross-Validation Scores (Logistic Regression)----")
print("CV Scores:", cv_scores)
print("Mean Accuracy:", round(cv_scores.mean()*100, 2), "%")
print("Standard Deviation:", round(cv_scores.std()*100, 2), "%")

#ROC-AUC Score
y_prob = model.predict_proba(X_test_scaled)[:, 1]
auc_score = roc_auc_score(y_test, y_prob)
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

print("\n----ROC-AUC Score (Logistic Regression)----")
print("AUC Score:", round(auc_score, 3))

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0,1],[0,1],linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Logistic Regression)')
plt.legend()
plt.savefig('roc_curve.png')
plt.show()

#Predicting happiness for a new country
new_country = pd.DataFrame({
    'Economy': [1.2],
    'Family': [1.3],
    'Health': [0.8],
    'Freedom': [0.5],
    'Corruption': [0.2],
    'Generosity': [0.3],
    'Job Satisfaction': [80],
    'Region_Encoded': [le.transform(['Western Europe'])[0]]
})

new_country_scaled = scaler.transform(new_country)
new_prediction = model.predict(new_country_scaled)

print("\n----Prediction for New Country----")
if new_prediction[0] == 1:
    print("Predicted: Happy country.") 
else:
    print("Predicted: Not a happy country.")

#Chart comparing models
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [accuracy*100, accuracy_dt*100, accuracy_rf*100]

#Plotting the model accuracy comparison
plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.savefig('model_accuracy_comparison.png')  # Save the figure as a PNG file
plt.show()
