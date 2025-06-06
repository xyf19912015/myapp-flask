# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:19:12 2024

@author: user
"""

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier  # 修改为 XGBoost
import requests
import io
import random

app = Flask(__name__)

random_state = 42
random.seed(random_state)
np.random.seed(random_state)

def load_data():
    url = 'https://raw.githubusercontent.com/xyf19912015/myapp-flask/master/KDlast3.csv'
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # 检查请求是否成功
        data = pd.read_csv(io.StringIO(response.content.decode('utf-8')), encoding='gbk')
        
        if data.empty:
            raise ValueError("Loaded data is empty.")
        
        return data
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except pd.errors.ParserError as e:
        print(f"Parser error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return None

def train_model():
    data = load_data()
    if data is None:
        print("Failed to load data. Model training cannot proceed.")
        return None, None, None, None, None

    # 特征和标签
    X = data.drop('PCAA', axis=1)
    y = data['PCAA']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(sampling_strategy=0.5, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # 使用 XGBoost 进行建模
    xgb_classifier = XGBClassifier(random_state=random_state, eval_metric='logloss')

    param_grid = {
        'n_estimators': [400],
        'learning_rate': [0.01],
        'max_depth': [6],
        'min_child_weight': [1],
        'gamma': [0.1],
        'subsample': [1.0],
        'colsample_bytree': [0.5],
        'reg_alpha': [0.01],
        'reg_lambda': [0.1]
    }

    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    best_xgb = grid_search.best_estimator_
    
    # 直接指定固定的最佳阈值
    best_threshold = 0.2283
    best_youden_index = None  # 可选：如果需要，可以设为 None 或其他有效值

    return scaler, best_xgb, X.columns, best_threshold, best_youden_index

# 训练模型并获取参数
scaler, best_xgb, feature_names, best_threshold, best_youden_index = train_model()

@app.route('/')
def home():
    annotations = {
        'No. of involved CAs': 'Number of Involved Coronary Arteries, n',
        'Zmax of initial CALs': 'Zmax of initial CALs',
        'Age': 'Age of onset, years',
        'IGT': 'Day of first time to use IVIG, day',
        'WBC': 'White blood cell, 10^9/L',
        'PLT': 'Platelets',
        'HB': 'Hemoglobin, g/dL'
    }
    return render_template('index.html', features=annotations.keys(), annotations=annotations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = []
        for feature in feature_names:
            if feature in request.form:
                value = float(request.form[feature])
                if feature == 'Zmax of initial CALs':
                    if value < 2:
                        value = 0
                    elif 2 <= value < 2.5:
                        value = 1
                    elif 2.5 <= value < 5:
                        value = 2
                    elif 5 <= value < 10:
                        value = 3
                    else:
                        value = 4
                input_features.append(value)
            else:
                input_features.append(0.0)  # 对于缺少的特征，用0填充
    except KeyError as e:
        return f"Error: Missing input for feature: {e.args[0]}"

    input_df = pd.DataFrame([input_features], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    prediction_proba = best_xgb.predict_proba(input_scaled)[:, 1][0]
    risk_level = "High Risk!" if prediction_proba > best_threshold else "Low Risk!"
    risk_color = "red" if prediction_proba > best_threshold else "green"
    prediction_rounded = round(prediction_proba, 4)

    return render_template('result.html', prediction=prediction_rounded, youden_index=best_youden_index, best_threshold=best_threshold, risk_level=risk_level, risk_color=risk_color)

if __name__ == '__main__':
    app.run(debug=True)
