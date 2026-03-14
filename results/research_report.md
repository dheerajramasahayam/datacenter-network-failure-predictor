# Research Output: Sequential ML vs Classical ML on Network Telemetry\n\nValidation Performed on: Network-And-Services/optical-failure-dataset\n\n## 1. Class-Imbalance Resolution & Architecture\nThe LSTM handles structural temporal flow and utilizes `BCEWithLogitsLoss(pos_weight)` natively weighting positive failure hits.\nBaseline models (LR & RF) fall back on `class_weight='balanced'` over flattened static temporal vectors.\n\n## 2. Experimental Results (1 Unknown Test Holdout)\n- Logistic Regression F1: 0.1791\n- Random Forest F1: 0.2275\n- LSTM (Temporal Sequence) F1: 0.2308\n\n## 3. LSTM Classification Details\n```\n              precision    recall  f1-score   support

           0       1.00      0.75      0.86     63238
           1       0.13      0.96      0.23      2485

    accuracy                           0.76     65723
   macro avg       0.56      0.85      0.54     65723
weighted avg       0.96      0.76      0.83     65723
\n```\n