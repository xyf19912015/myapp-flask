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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report
from xgboost import XGBClassifier
import requests
import io

app = Flask(__name__)

random_state = 42

# 模型训练部分
def train_model():
    # 加载数据并处理
    url = 'https://raw.githubusercontent.com/xyf19912015/myapp-flask/master/KDlast3.csv'
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')), encoding='gbk')

    # 特征和标签
    X = data.drop('PCAA', axis=1)
    y = data['PCAA']

    # 手动指定要使用的特征（根据你的具体需求）
    selected_features = ['Num of involved CAs', 'Zmax of initial CALs', 'Age', 'DF', 'AST', 'WBC', 'PLT', 'HB']

    # 过滤所选特征
    X = X[selected_features]

    # 特征标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 过采样处理不平衡
    smote = SMOTE(sampling_strategy=0.5)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # 训练集测试集分割
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # 训练模型
    xgb_classifier = XGBClassifier()

    # 使用网格搜索进行超参数优化
    param_grid = {
        'n_estimators': [300],
        'learning_rate': [0.05],
        'max_depth': [5],
        'min_child_weight': [1],
        'gamma': [0.1],
        'subsample': [0.6],
        'colsample_bytree': [1.0],
        'reg_lambda': [1.0],
        'reg_alpha': [0.1]
    }

    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 最佳模型
    best_xgb = grid_search.best_estimator_

    return scaler, best_xgb, selected_features, X.columns, y_test, X_test

scaler, best_xgb, selected_features, original_columns, y_test, X_test = train_model()

@app.route('/')
def home():
    annotations = {
        'Num of involved CAs': 'Number of Involved Coronary Arteries, n',
        'Zmax of initial CALs': 'Zmax of initial CALs',
        'Age': 'Age of onset, years',
        'DF': 'Duration of fever, day',
        'AST': 'Aspartate aminotransferase, U/L',
        'WBC': 'White blood cell, 10^9/L',
        'PLT': 'Platelets',
        'HB': 'Hemoglobin, g/dL'
    }
    return render_template('index.html', features=selected_features, annotations=annotations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = []
        for feature in selected_features:
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
    except KeyError as e:
        return f"Error: Missing input for feature: {e.args[0]}"

    input_df = pd.DataFrame([input_features], columns=selected_features)

    # Align input dataframe columns with original columns
    aligned_input_df = pd.DataFrame(np.zeros((1, len(original_columns))), columns=original_columns)
    aligned_input_df[selected_features] = input_df

    # Standardizing input
    input_scaled = scaler.transform(aligned_input_df[selected_features])

    # Prediction
    prediction_proba = best_xgb.predict_proba(input_scaled)[:, 1][0]
    prediction = best_xgb.predict(input_scaled)[0]
    prediction_rounded = round(prediction_proba, 4)

    # Calculate Youden's index
    fpr, tpr, thresholds = roc_curve(y_test, best_xgb.predict_proba(X_test)[:, 1])
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    youden_index_value = np.max(youden_index)

    return render_template('result.html', prediction=prediction_rounded, youden_index=youden_index_value)

if __name__ == '__main__':
    app.run(debug=True)
