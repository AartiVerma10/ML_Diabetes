# -Project Goal-

The primary objective of this project is to build a machine learning model to predict diabetes. It uses a K Nearest Neighbors (KNN) classifier.


---


## -1. Data Loading and Exploration -

Loading: The project starts by importing the necessary libraries (numpy, pandas, matplotlib.pyplot, seaborn) and loading a dataset named "diabetes.csv".

Initial Inspection: It uses print(dataset) and dataset.info() to get a feel for the data. The dataset contains 768 rows and 9 columns, representing various health metrics (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age) and the target variable, Outcome (1 for diabetes, 0 for no diabetes).

Missing Value Check: The code checks for missing values using dataset.isnull().

## -2. Data Cleaning and Preprocessing-

Identifying Invalid Data: I noticed an anomaly using dataset.describe(): the minimum value for columns like Glucose, BloodPressure, SkinThickness, Insulin, and BMI is 0. This is medically impossible (you can't have a blood pressure or BMI of 0).

Quantifying the Problem: A loop calculates the percentage of these invalid 0 values for each feature (e.g., Insulin has a massive 48.70% invalid data).

Imputation: To fix this, the code replaces the 0 values with more realistic estimates:

Glucose and BloodPressure 0s are replaced with the median.
BMI, SkinThickness, and Insulin 0s are replaced with the mean.

## -3. Data Visualization-

Correlation: A heatmap (sns.heatmap) is generated to visualize the correlations between all the variables. This helps identify which features are most strongly related to each other and to the diabetes outcome.

Density Plots: Kernel Density Estimation (KDE) plots are created for Pregnancies, Glucose, and Insulin. These plots overlay the distribution of the feature for people with diabetes (Outcome=1) versus those without (Outcome=0). This visualizes how well a single feature might separate the two classes.

## -4. Machine Learning Model Preparation-

Splitting Features and Target: The data is separated into X (the independent variables/features) by dropping the "Outcome" column, and y (the dependent variable/target), which is just the "Outcome" column.

Train-Test Split: The train_test_split function from sklearn.model_selection is used to divide the data. 67% of the data is used to train the model (X_train, y_train), and the remaining 33% is kept aside to test its performance (X_test, y_test).

## -5. Model Training and Evaluation (KNN)-

Algorithm Selection: The chosen algorithm is K-Nearest Neighbors (KNN).

Hyperparameter Tuning (Finding the best 'K'): The code uses a for loop to test the KNN model with different numbers of neighbors, from k=1 to k=10.

Evaluation: For each value of 'K', it trains the model and calculates both the training accuracy and the test accuracy.

Visualizing Performance: Finally, it plots a line graph showing how training and test accuracy change as the number of neighbors ('K') increases. This plot is crucial for finding the "sweet spot" where the model generalizes well to new data without overfitting (where training accuracy is high but test accuracy drops).


---



## In the final cells i introduced two major changes to your pipeline: scaled data using StandardScaler and switched machine learning model to a Decision Tree Classifier.


## 1. The StandardScaler Method
Before feeding data into the new model, used StandardScaler on the training and testing sets.

What it does: In the dataset, features are on completely different scales. For example, DiabetesPedigreeFunction has values like 0.6, while Insulin has values up to 846. StandardScaler normalizes these features so that every column has a mean of 0 and a standard deviation of 1.

Why it matters: Many machine learning algorithms perform poorly if one feature has huge numbers and another has tiny numbers, because the algorithm might unfairly prioritize the larger numbers.


## 2. The DecisionTreeClassifier Method
I then trained a Decision Tree, which is a model that makes predictions by learning simple "if/then" decision rules 
inferred from the data features (e.g., "If Glucose > 120 and BMI > 30, then Diabetes = 1").

The Parameter Used:

random_state=15: You passed this parameter into the classifier. Because the math behind building a decision tree has some randomness in how it splits the data, setting a random_state locks that randomness. It ensures that every time i ran this cell, i will get the exact same tree and the exact same accuracy numbers.

## 3. Understanding the Results (The "Overfitting" Problem)
The most important part of the final cells is the output it generated:

Training accuracy: 1.0 (100%)

Test accuracy: ~0.71 (71.2%)

This is a textbook example of Overfitting. Because i did not give the Decision Tree any parameters to stop it from growing (like a max_depth parameter), 
the tree kept creating branches until it had perfectly memorized every single patient in your training data (hence the 1.0 accuracy).
 However, because it memorized the specifics of the training data rather than learning general patterns, 
 it performed significantly worse (71%) when tested on unseen data.


---


## Theory about K-Nearest Neighbors (KNN) Algorithm.

### 1. The Core Concept

KNN is a supervised, lazy-learning, instance-based algorithm used for both classification and regression.

Lazy Learning: It doesn't "learn" a mathematical model during training. It simply memorizes the training dataset.

Instance-based: It predicts the label of a new data point by finding the K closest training examples (neighbors) in the feature space and taking a majority vote (for classification) or an average (for regression).

### 2. Distance Metrics (How we define "Nearest")

The algorithm calculates the distance between the test point and all training points using one of these metrics:

Euclidean Distance (Default, p=2): The straight-line distance. Best for continuous variables. $\sqrt{\sum (x_i - y_i)^2}$

Manhattan Distance (p=1): The sum of absolute differences. Best for high-dimensional or grid-like data. $\sum |x_i - y_i|$

Minkowski Distance: The generalized form of Euclidean and Manhattan, controlled by the parameter p.

### 3. Hyperparameters to Tune

n_neighbors (K): The number of voting neighbors.

Small K (e.g., 1, 3): Low bias, high variance (prone to overfitting noise).

Large K (e.g., 20, 50): High bias, low variance (prone to underfitting).

Rule of thumb: Choose an odd number (to break ties) and typically start around the square root of the number of samples in your training set ($\sqrt{n}$).

weights: * uniform: All neighbors have an equal vote.

distance: Closer neighbors have a heavier vote than ones further away.

### 4. Mandatory Preprocessing: Feature Scaling

Because KNN relies on physical distance calculations, you must scale your features. If Feature A ranges from 0-1 and Feature B ranges from 0-1000, Feature B will completely dominate the distance calculation.

Solution: Use StandardScaler (Z-score normalization) or MinMaxScaler.

### 5. The Complete Python Pipeline (Scikit-Learn)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Data (assuming you are using the diabetes dataset from earlier)
df = pd.read_csv("diabetes.csv")

# 2. Separate Features (X) and Target (y)
X = df.drop('Outcome', axis=1) # Replace 'Outcome' with your target column
y = df['Outcome']

# 3. Split into Training and Testing Sets
# random_state ensures reproducibility. stratify ensures target class balance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Feature Scaling (CRITICAL FOR KNN)
scaler = StandardScaler()
# Fit ONLY on training data to prevent data leakage, then transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Initialize and Train the Model
# n_neighbors=5 and metric='minkowski' (p=2) are default
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn.fit(X_train_scaled, y_train)

# 6. Make Predictions
y_pred = knn.predict(X_test_scaled)

# 7. Evaluate the Model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

```
### 6. Pros and Cons

*Pros:*

Incredibly simple to understand and explain.

No training period (fast initial setup).

Naturally handles multi-class classification natively.

*Cons:*

Computationally Expensive at Test Time: Must calculate the distance to every single training point for every new prediction.

High Memory Footprint: Requires keeping the entire dataset in memory.

Curse of Dimensionality: Performance degrades drastically if you have too many features (columns). In high dimensions, all points become nearly equidistant.