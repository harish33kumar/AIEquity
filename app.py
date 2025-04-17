from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, log_loss, roc_auc_score
import xgboost as xgb
plt.style.use('ggplot')

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # Allow CORS for all domains

@app.route('/genderpaypredict', methods=['POST'])
def genderPayPredict():

    requestdata = request.json
    sample_df = pd.DataFrame([requestdata])
    print(sample_df.info())
    # Load and preprocess the training data
    pay_gap = pd.read_csv("Glassdoor Gender Pay Gap.csv", encoding="UTF-8")
    pay_gap.dropna(inplace=True)  # Handle missing values
    # Convert column names to lowercase
    pay_gap.columns = [col.lower() for col in pay_gap.columns]
    # Ensure sample_df column names match those in pay_gap

    # Convert categorical columns to numeric using pd.cut and map
    pay_gap_logistic = pay_gap

    bins = [17, 35, 50, 66]
    labels = ['Young Adults', 'Middle-aged Adults', 'Old Adults']
    # Convert 'age' column to numeric, setting errors='coerce' to turn non-numeric values into NaN
    pay_gap_logistic['age'] = pd.to_numeric(pay_gap_logistic['age'], errors='coerce')
    sample_df['age'] = pd.to_numeric(sample_df['age'], errors='coerce')

    # Handle NaN values if needed (e.g., fill with a specific value or drop them)
    # Example: dropping NaNs
    pay_gap_logistic.dropna(subset=['age'], inplace=True)
    pay_gap_logistic['age'] = pd.cut(pay_gap_logistic['age'], bins=bins, labels=labels)
    sample_df['age'] = pd.cut(sample_df['age'], bins=bins, labels=labels)

    bins = [34000, 100000, 180000]
    labels = ['Low income', 'High income']
    pay_gap_logistic['basepay'] = pd.to_numeric(pay_gap_logistic['basepay'], errors='coerce')
    # Example: dropping NaNs
    pay_gap_logistic.dropna(subset=['basepay'], inplace=True)
    pay_gap_logistic['basepay'] = pd.cut(pay_gap_logistic['basepay'], bins=bins, labels=labels)
    pay_gap_logistic["basepay"] = pay_gap_logistic.basepay.replace(to_replace=['Low income', 'High income'],value=[0, 1])
    bins = [1700, 5000, 10000, 11500]
    labels = ['Low', 'Middle', 'High']
    pay_gap_logistic['bonus'] = pd.cut(pay_gap_logistic['bonus'], bins=bins, labels=labels)
    sample_df['bonus'] = pd.cut(sample_df['bonus'], bins=bins, labels=labels)

    pay_gap_logistic["gender"] = pay_gap_logistic.gender.replace(to_replace=['Male', 'Female'], value=[0, 1])
    sample_df["gender"] = sample_df.gender.replace(to_replace=['Male', 'Female'], value=[0, 1])

    dummies = ['jobtitle', 'age', 'education', 'dept', 'bonus']
    for column in dummies:
        if column in pay_gap_logistic.columns:
            pay_gap_logistic[column] = pay_gap_logistic[column].str.lower()  # Convert to lowercase
            sample_df[column] = sample_df[column].str.lower()

    dummy_pay_gap_logistic = pd.get_dummies(pay_gap_logistic[dummies])
    # We will concat the two data frames, and drop the old columns.
    pay_gap_logistic1 = pd.concat([pay_gap_logistic, dummy_pay_gap_logistic], axis=1)
    pay_gap_logistic1.drop(dummies, axis=1, inplace=True)
    dummy_columns = pay_gap_logistic1.columns.tolist()

    dummy_sample_df_logistic = pd.get_dummies(sample_df[dummies])
    sample_df[dummy_pay_gap_logistic.columns]=0
    sample_df[dummy_sample_df_logistic.columns]=dummy_sample_df_logistic

    sample_df.drop(dummies, axis=1, inplace=True)

    print("pay_gap_logistic1:", dummy_columns)
    print("request_data:", sample_df)


    X = pay_gap_logistic1.drop('basepay', axis=1)
    y = pay_gap_logistic1['basepay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    pay_gap_rf = RandomForestClassifier(n_estimators=150)
    pay_gap_rf.fit(X_train, y_train)
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',enable_categorical=True)
    xgb_clf.fit(X_train, y_train)

    y_pred = log_reg.predict(X_test)
    y_pred_rf = pay_gap_rf.predict(X_test)
    y_pred_xgb = xgb_clf.predict(X_test)

    print("X_test:", X_test)
    print("y_pred_rf",y_pred_rf)

    m_names = ['accuracy_score', 'log_loss']
    metrics = [accuracy_score, log_loss]
    for n, m in zip(m_names, metrics):
        print('{:.5f} : {}'.format(m(y_test, y_pred), n))

    for n, m in zip(m_names, metrics):
        print('{:.5f} : {}'.format(m(y_test, y_pred_rf), n))

    for n, m in zip(m_names, metrics):
        print('{:.5f} : {}'.format(m(y_test, y_pred_xgb), n))

    logistic_acc = float(accuracy_score(y_test, y_pred))
    rf_acc = float(accuracy_score(y_test, y_pred_rf))
    xgb_acc = float(accuracy_score(y_test, y_pred_xgb))

    y_pred_logistic = log_reg.predict(sample_df)
    y_pred_rf = pay_gap_rf.predict(sample_df)
    y_pred_xgb =  xgb_clf.predict(sample_df)

    # Compare accuracy and choose the best model
    if logistic_acc >= rf_acc and logistic_acc >= xgb_acc:
        selected_model = 'logisticRegression'
        accuracy = logistic_acc
        prediction = y_pred_logistic[0]
    elif rf_acc >= xgb_acc:
        selected_model = 'randomForestClassifier'
        accuracy = rf_acc
        prediction = y_pred_rf[0]
    else:
        selected_model = 'xgboost'
        accuracy = xgb_acc
        prediction = y_pred_xgb[0]

    prediction = (lambda x: 'Yes' if x == 1 else 'No')(prediction)
    print(prediction)
    prediction_text = "High income" if prediction == 'Yes' else "Low income"
    message = f"The prediction indicates a {prediction_text} based on the provided data."
    return jsonify({'modelName': selected_model, 'accuracy': accuracy*100, 'prediction': prediction,'message': message})


@app.route('/predict', methods=['POST'])
def predict():
    requestdata = request.json

    features = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction',
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]

    # Create a DataFrame for the incoming sample matching the training features
    sample_df = pd.DataFrame([requestdata], columns=features)
    sample_df = sample_df.apply(pd.to_numeric, errors='coerce')
    print("sample_df:", sample_df.info())

    try:
        # Load the model
        with open('random_forest_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)

        # Prediction
        prediction = loaded_model.predict(sample_df)
        print("Predicted value:", prediction)
        prediction = (lambda x: 'Yes' if x == 1 else 'No')(prediction)
        if prediction=='Yes':
            return jsonify({'status': 'Warning', 'prediction': prediction, 'modelName':'Random Forest Classifier','confidence': 0.84, 'primaryfactor': 'Compensation',
             'secondaryfactor': 'Workload and Careergrowth'})
        else:
            return jsonify({'status': 'Good', 'prediction': prediction, 'modelName':'Random Forest Classifier', 'confidence': 0.84})

    except Exception as e:
        print("An error occurred:", e)
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)