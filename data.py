# World Happiness Predictor Project
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
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

print("\n----Model Comparison----")
print("Logistic Regression Accuracy:", accuracy*100, "%")
print("Decision Tree Accuracy:", accuracy_dt*100, "%")
print("Random Forest Accuracy:", accuracy_rf*100, "%")

#Confusion Matrix
cm=confusion_matrix(y_test, y_pred)
print("\n----Confusion Matrix (Logistic Regression)----")
print(cm)

#Chart comparing models
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [accuracy*100, accuracy_dt*100, accuracy_rf*100]

plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.title('Model Accuracy Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.savefig('model_accuracy_comparison.png')  # Save the figure as a PNG file
plt.show()
