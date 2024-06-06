# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:19:12 2024

@author: user
"""

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
    y_pred_proba = best_xgb.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    youden_index = tpr - fpr
    best_threshold = thresholds[np.argmax(youden_index)]

    # Apply the best threshold to classify
    y_pred_test = (y_pred_proba > best_threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()

    # Calculate Youden's Index
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    youden_index_value = sensitivity + specificity - 1

    # Determine risk level
    risk = 'high' if prediction_rounded > youden_index_value else 'low'

    return render_template('result.html', prediction=prediction_rounded, youden_index=round(youden_index_value, 4), risk=risk)
