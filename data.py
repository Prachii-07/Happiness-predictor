# World Happiness Predictor Project
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("World Happiness Report.csv")

print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

#Handle missing values
df['Job Satisfaction'] = df['Job Satisfaction'].fillna(df['Job Satisfaction'].mean())

#Create a targetvariable: Happy=1 if above median happiness score, else 0
median_score = df['Happiness Score'].median()
df['Happy'] = (df['Happiness Score'] > median_score).astype(int)

print(df['Happy'].value_counts())

