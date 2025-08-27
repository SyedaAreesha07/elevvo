# 📌 Titanic Dataset Exploratory Data Analysis (EDA)
# ---------------------------------------------------

# Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
# Kaggle se Titanic dataset download karke same folder me "train.csv" rakho
df = pd.read_csv("train.csv")

# Step 3: Basic Info
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)   # Missing Age -> median se fill
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Missing Embarked -> mode se fill
df.drop(columns=['Cabin'], inplace=True)  # Cabin column bohot missing hai -> drop

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Step 5: Summary Statistics
print("\nSummary Statistics:\n", df.describe())

# Step 6: Group-Based Insights
print("\nSurvival Rate by Gender:\n", df.groupby('Sex')['Survived'].mean())
print("\nSurvival Rate by Passenger Class:\n", df.groupby('Pclass')['Survived'].mean())

# Step 7: Visualization

# Survival count plot
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title("Survival Count")
plt.show()

# Survival by Gender
plt.figure(figsize=(6,4))
sns.barplot(x='Sex', y='Survived', data=df, palette='Set1')
plt.title("Survival Rate by Gender")
plt.show()

# Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.barplot(x='Pclass', y='Survived', data=df, palette='muted')
plt.title("Survival Rate by Class")
plt.show()

# Survival by Gender and Class
plt.figure(figsize=(8,6))
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=df, palette='Set2')
plt.title("Survival Rate by Class and Gender")
plt.show()

# Heatmap for Correlation
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
