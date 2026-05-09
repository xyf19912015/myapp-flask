# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:13:21 2026

@author: Administrator
Flask + Random Forest + ADASYN
Maximum baseline coronary artery Z-score 特征做区间转换
固定阈值 0.3242
"""

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import requests
import io

# ===================== 基础设置 =====================
app = Flask(__name__)
random_state = 42
np.random.seed(random_state)

# ===================== 特征和注释 =====================
FEATURES = [
    "Maximum baseline coronary artery Z-score",
    "Number of involved coronary arteries",
    "Day of initial IVIG treatment, days",
    "Age, years",
    "IVIG-resistant Kawasaki disease"
]

annotations = {
    "Maximum baseline coronary artery Z-score": "Maximum baseline coronary artery Z-score",
    "Number of involved coronary arteries": "Number of involved coronary arteries",
    "Day of initial IVIG treatment, days": "Day of initial IVIG treatment, days",
    "Age, years": "Age, years",
    "IVIG-resistant Kawasaki disease": "IVIG-resistant Kawasaki disease"
}

# ===================== 加载数据 =====================
def load_data():
    url = 'https://raw.githubusercontent.com/xyf19912015/myapp-flask/master/Development_setHCHmodel.csv'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')), encoding='gbk')
    X = data[FEATURES]
    y = data['PCAA']
    return X, y

# ===================== 训练模型 =====================
def train_model():
    X, y = load_data()

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ADASYN 过采样
    adasyn = ADASYN(sampling_strategy=0.5, random_state=random_state)
    X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y)

    # 随机森林训练
    rf = RandomForestClassifier(random_state=random_state)
    param_grid = {
        'n_estimators': [200],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'max_features': ['sqrt']
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)
    best_rf = grid_search.best_estimator_

    # 全部特征直接使用
    selected_idx = np.arange(len(FEATURES))

    return scaler, best_rf, selected_idx

# 训练一次模型
scaler, best_rf, selected_idx = train_model()

# ===================== Flask 路由 =====================
@app.route('/')
def home():
    # 渲染 index.html，传 features 和 annotations
    return render_template('index.html', features=FEATURES, annotations=annotations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = []

        for feature in FEATURES:
            value = float(request.form.get(feature, 0.0))

            # 特殊处理 Maximum baseline coronary artery Z-score
            if feature == "Maximum baseline coronary artery Z-score":
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

        # 构造 DataFrame 并标准化
        input_df = pd.DataFrame([input_features], columns=FEATURES)
        input_scaled = scaler.transform(input_df)
        input_selected = input_scaled[:, selected_idx]  # 全部特征直接用

        # 预测概率
        prob = best_rf.predict_proba(input_selected)[:, 1][0]

        # 固定阈值
        threshold = 0.3242
        risk = "High Risk!" if prob > threshold else "Low Risk!"
        color = "red" if prob > threshold else "green"

        return render_template(
            'result.html',
            prediction=round(prob, 4),
            risk_level=risk,
            risk_color=color
        )

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
