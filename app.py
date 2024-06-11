# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:19:12 2024

@author: user
"""

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix

def calculate_youden_index(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]
    return best_threshold

def cross_validated_youden_index(X, y, model, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    thresholds = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        best_threshold = calculate_youden_index(y_test, y_proba)
        thresholds.append(best_threshold)
    
    return np.mean(thresholds)

# 训练和评估模型
def train_model():
    url = 'https://raw.githubusercontent.com/xyf19912015/myapp-flask/master/KDlast3.csv'
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')), encoding='gbk')

    X = data.drop('PCAA', axis=1)
    y = data['PCAA']

    selected_features = select_features(X, y)
    X = X[selected_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(sampling_strategy=0.5, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    xgb_classifier = XGBClassifier(random_state=random_state)

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

    grid_search = GridSearchCV(estimator=xgb_classifier, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    best_xgb = grid_search.best_estimator_

    best_threshold = cross_validated_youden_index(X_resampled, y_resampled, best_xgb)

    return scaler, best_xgb, selected_features, X.columns, best_threshold

scaler, best_xgb, selected_features, original_columns, best_threshold = train_model()

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

    aligned_input_df = pd.DataFrame(np.zeros((1, len(original_columns))), columns=original_columns)
    aligned_input_df[selected_features] = input_df

    input_scaled = scaler.transform(aligned_input_df[selected_features])

    prediction_proba = best_xgb.predict_proba(input_scaled)[:, 1][0]
    prediction = (prediction_proba > best_threshold).astype(int)
    prediction_rounded = round(prediction_proba, 4)

    return render_template('result.html', prediction=prediction_rounded, youden_index=best_threshold)

if __name__ == '__main__':
    app.run(debug=True)
