# -*- coding: utf-8 -*-
"""
Created on Thu May 16 16:59:21 2024

@author: user
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
import shap
import joblib  # 用于持久化模型
from flask_bootstrap import Bootstrap
import requests
from io import StringIO
from sklearn.metrics import roc_curve

app = Flask(__name__)
Bootstrap(app)

# 加载数据并处理
url = 'https://raw.githubusercontent.com/xyf19912015/myapp-flask/master/KDlast2.csv'
response = requests.get(url)
content = response.content

# 查看获取的内容是否正常
print("Response status code:", response.status_code)
print("Content snippet:", content[:500])  # 检查前500个字符

data = pd.read_csv(StringIO(content.decode('gbk')))
print("DataFrame columns:", data.columns)

# 确保 'PCAA' 列存在
if 'PCAA' in data.columns:
    X = data.drop(columns=['PCAA'])
    y = data['PCAA']
    print("X head:", X.head())
    print("y head:", y.head())
else:
    print("'PCAA' column not found in DataFrame")
    raise KeyError("'PCAA' column not found in DataFrame")

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 过采样处理不平衡
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 训练集测试集分割
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# XGBoost classifier
xgb_classifier = XGBClassifier()

# Feature selection with RFECV
selector = RFECV(estimator=xgb_classifier, step=1, cv=5, scoring='accuracy', min_features_to_select=1)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 参数设置
param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.05],
    'max_depth': [5],
    'min_child_weight': [1],
    'gamma': [0],
    'subsample': [0.6],
    'colsample_bytree': [1.0],
    'reg_lambda': [1.0],
    'reg_alpha': [0.1]
}

grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

best_xgb = grid_search.best_estimator_
selected_columns = X.columns[selector.support_]

# 保存模型和Scaler
joblib.dump(best_xgb, 'best_xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selector, 'selector.pkl')

# 加载保存的模型和Scaler
model = joblib.load('best_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
selector = joblib.load('selector.pkl')

# 转换训练数据选定特征为数据框
X_train_selected_df = pd.DataFrame(X_train_selected, columns=selected_columns)
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_columns)

# 创建SHAP解释器
explainer = shap.Explainer(model, X_train_selected_df)

annotations_dict = {
    "DF": "Duration of fever (d):",
    "IGT": "Initial IVIG treatment time (d):",
    "age": "Age (m):",
    "PLT": "Platelet count (10^9/L):",
    "ALB": "Albumin (g/L):",
    "WBC": "White blood cell count (10^9/L):",
    "HB": "Hemoglobin (g/dL):",
    "AST": "Aspartate aminotransferase (U/L):"
}

@app.route('/')
def index():
    return render_template('index.html', features=selected_columns.tolist(), annotations=annotations_dict)

def calculate_optimal_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]
    return optimal_threshold

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_dict = {feature: [float(data[feature])] for feature in selected_columns}

    input_df = pd.DataFrame.from_dict(input_dict)
    input_df = input_df[selected_columns]

    input_data_scaled = scaler.transform(input_df)
    input_data_selected = selector.transform(input_data_scaled)

    prediction_proba = model.predict_proba(input_data_selected)[:, 1][0]
    prediction = round(prediction_proba * 100, 2)

    shap_values = explainer(pd.DataFrame(input_data_selected, columns=selected_columns))

    optimal_threshold = calculate_optimal_threshold(y_test, model.predict_proba(X_test_selected)[:, 1])
    if prediction_proba >= optimal_threshold:
        advice = "High risk. Immediate clinical intervention required."
    else:
        advice = "Low risk. Regular monitoring recommended."

    return jsonify({
        'probability': prediction,
        'advice': advice,
        'optimal_threshold': optimal_threshold,  # 输出最优阈值
        'shap_values': shap_values.values[0].tolist(),
    })

if __name__ == '__main__':
    app.run(debug=True)
