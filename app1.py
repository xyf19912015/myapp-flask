# -*- coding: utf-8 -*-
"""
Created on Fri May  8 22:13:21 2026

@author: Administrator
"""
# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

random_state = 42
np.random.seed(random_state)

# ===================== 数据与特征 =====================
FEATURES = [
    "Maximum baseline coronary artery Z-score",
    "Number of involved coronary arteries",
    "Day of initial IVIG treatment, days",
    "Age, years",
    "IVIG-resistant Kawasaki disease"
]

def load_data():
    url = 'https://raw.githubusercontent.com/xyf19912015/myapp-flask/master/Development_setHCHmodel.csv'
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

def load_data():
    data = pd.read_csv(DATA_PATH, encoding='gbk')
    X = data[FEATURES]
    y = data['PCAA']
    return X, y

# ===================== 训练模型 =====================
def train_model():
    X, y = load_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    adasyn = ADASYN(sampling_strategy=0.5, random_state=random_state)
    X_resampled, y_resampled = adasyn.fit_resample(X_scaled, y)

    # LASSO 特征选择（如果想保留 LASSO 可以用，否则可以跳过）
    lasso = Lasso(alpha=0.01)
    lasso.fit(X_resampled, y_resampled)
    coefs = lasso.coef_
    selected_idx = np.where(coefs != 0)[0]
    X_selected = X_resampled[:, selected_idx]

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
    grid_search.fit(X_selected, y_resampled)
    best_rf = grid_search.best_estimator_

    return scaler, best_rf, selected_idx

scaler, best_rf, selected_idx = train_model()

# ===================== Flask 路由 =====================
@app.route('/')
def home():
    return render_template('index.html', features=FEATURES)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for feat in FEATURES:
            value = float(request.form.get(feat, 0.0))
            input_data.append(value)

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        input_selected = input_scaled[:, selected_idx]

        prob = best_rf.predict_proba(input_selected)[:, 1][0]
        threshold = 0.3242  # 可以使用约登指数阈值或自定义
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
