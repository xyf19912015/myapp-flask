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

app = Flask(__name__)
Bootstrap(app)

# 加载数据并处理
data = pd.read_csv("C:/Users/user/Desktop/Python/ML/master/KDlast.csv", encoding='gbk')

X = data.drop('PCAA', axis=1)
y = data['PCAA']

# 特征标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 过采样处理不平衡
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# 训练集测试集分割
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
xgb_classifier = XGBClassifier()

# Feature selection with RFECV
selector = RFECV(estimator=xgb_classifier, step=1, cv=5, scoring='accuracy', min_features_to_select=1)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Grid search parameters for XGBoost
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
    "age": "Age (y):",
    "PLT": "Platelet count (10^9/L):",
    "ALB": "Albumin (g/L):",
    "IL-6": "Interleukin-6 (pg/mL):",
    "WBC": "White blood cell count (10^9/L):",
    "HB": "Hemoglobin (g/dL):",
    "ALT": "Alanine Transaminase (U/L):"
}

@app.route('/')
def index():
    return render_template('index.html', features=selected_columns.tolist(), annotations=annotations_dict)

def generate_annotations(data):
    age_annotation = 1 if float(data['age']) < 1 else 0
    fever_annotation = 0 if float(data['DF']) <= 10 else 1
    ivig_annotation = 0 if 5 <= float(data['IGT']) <= 7 else 1
    
    data['age'] = age_annotation  # 修改原始数据
    data['DF'] = fever_annotation  # 修改原始数据
    data['IGT'] = ivig_annotation  # 修改原始数据
    
    return {
        "age_annotation": f"Age ≤1 y, {age_annotation}",
        "fever_annotation": f"Fever time ≤10 d, {fever_annotation}",
        "ivig_annotation": f"Initial IVIG treatment was ≥5 d and ≤7 d, {ivig_annotation}"
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # 生成注释并修改数据
    annotations = generate_annotations(data)
    input_dict = {feature: [float(data[feature])] for feature in selected_columns}

    input_df = pd.DataFrame.from_dict(input_dict)
    input_df = input_df[selected_columns]

    input_data_scaled = scaler.transform(input_df)
    input_data_selected = selector.transform(input_data_scaled)

    # 预测
    prediction_proba = model.predict_proba(input_data_selected)[:, 1][0]
    prediction = round(prediction_proba * 100, 2)

    # 计算SHAP值
    shap_values = explainer(pd.DataFrame(input_data_selected, columns=selected_columns))

    # 生成建议
    advice = "Please refer to the medical guidelines."
    if prediction_proba < 0.20:
        advice = "Low probability. Regular monitoring recommended."
    elif 0.20 <= prediction_proba < 0.50:
        advice = "Moderate probability. Consider additional tests."
    elif 0.50 <= prediction_proba < 0.80:
        advice = "High probability. Further clinical evaluation and potential intervention needed."
    else:
        advice = "Very high probability. Immediate clinical intervention required."

    return jsonify({
        'probability': prediction,
        'advice': advice,
        'shap_values': shap_values.values[0].tolist(),
        'annotations': annotations  # 返回注释
    })

if __name__ == '__main__':
    app.run(debug=True)