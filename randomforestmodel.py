import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import pickle
import warnings
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, log_loss, \
    roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
plt.style.use('ggplot')
import os

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Allow CORS for all domains


@app.route('/randomforestmodel', methods=['GET'])
def model():
    # Path to your dataset
    file_path = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
    data = pd.read_csv(file_path)

    features = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction','StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]

    # Preprocessing
    df = data.copy()
    y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    X = df.loc[:, df.columns != 'Attrition']
    X=X[features]
    # Train/test split
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.7, random_state=21)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=21)
    print("X_test Success",X_test.iloc[12])
    print("X_test Failure", X_test.iloc[11])

    print("Training set size:", X_train.shape, y_train.shape)
    # Parameter tuning for RandomForest
    grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': [None, 'sqrt'],
        'max_depth': [None, 5, 7, 14, 21],
        'min_samples_split': [2, 5, 8],
        'min_samples_leaf': [3, 4, 5],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=92)
    rf_cv = RandomizedSearchCV(estimator=rf, param_distributions=grid, n_iter=10, scoring='roc_auc', cv=5,
                               random_state=92, n_jobs=-1)
    rf_cv.fit(X_train, y_train)

    # Validate model
    val_auc = roc_auc_score(y_val, rf_cv.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(y_test, rf_cv.predict_proba(X_test)[:, 1])
    val_acc = accuracy_score(y_val, rf_cv.predict(X_val))
    test_acc = accuracy_score(y_test, rf_cv.predict(X_test))

    print("predict",rf_cv.predict(X_test))
    print("Validation Accuracy: {:.1f}% and AUC = {:.3f}".format(val_acc * 100, val_auc))
    print("Test set Accuracy: {:.1f}% and AUC = {:.3f}".format(test_acc * 100, test_auc))
    # Save the model to a file
    with open('random_forest_model.pkl', 'wb') as model_file:
        pickle.dump(rf_cv, model_file)
    return "model pickled"

if __name__ == '__main__':
        app.run(debug=True)