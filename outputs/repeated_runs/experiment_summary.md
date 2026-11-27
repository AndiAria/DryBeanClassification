# Final Experiment Summary: Dry Bean Classification

## 1. Experiment Overview

- Project: Dry Bean Classification
- Algorithms: Decision Tree and Random Forest
- Number of experiment runs: 5
- Random states: [42, 43, 44, 45, 46]
- Model stages: Baseline and Tuned
- Total evaluation records: 20

## 2. Dataset Information

- Raw observations: 13,611
- Clean observations: 13,543
- Exact duplicate rows removed: 68
- Number of features: 16
- Target variable: Class
- Number of classes: 7
- Class labels: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA

## 3. Training and Testing Configuration

- Training/testing split: Based on actual split in each run
- Stratification: As configured in the experiment
- Cross-validation: 5-Fold
- Hyperparameter tuning metric: Macro F1

## 4. Five-Run Model Performance

| model | stage | mean_accuracy | std_accuracy | mean_f1_macro | std_f1_macro |
| --- | --- | --- | --- | --- | --- |
| Decision Tree | Baseline | 0.8928 | 0.0064 | 0.9077 | 0.0049 |
| Decision Tree | Tuned | 0.9015 | 0.0031 | 0.9155 | 0.0032 |
| Random Forest | Baseline | 0.9183 | 0.0045 | 0.9301 | 0.0038 |
| Random Forest | Tuned | 0.9174 | 0.0042 | 0.9290 | 0.0031 |

## 5. Cross-Validation Results

| model | mean_cv_macro_f1 | std_cv_macro_f1 |
| --- | --- | --- |
| Random Forest | 0.9357 | 0.0010 |
| Decision Tree | 0.9195 | 0.0015 |

## 6. Baseline vs Tuned Comparison

| model | baseline_mean_f1_macro | tuned_mean_f1_macro | f1_macro_improvement | improvement_percentage_points |
| --- | --- | --- | --- | --- |
| Decision Tree | 0.9077 | 0.9155 | 0.0078 | 0.7767 |
| Random Forest | 0.9301 | 0.9290 | -0.0011 | -0.1117 |

## 7. Best Model Selection

- Best algorithm: Random Forest
- Mean CV Macro F1: 0.93574
- CV Macro F1 standard deviation: 0.00101
- Mean testing Accuracy: 0.91739
- Mean testing Macro F1: 0.92901
- Testing Macro F1 standard deviation: 0.00306

## 8. Model Stability

- Most stable model: Random Forest
- Lowest testing Macro F1 standard deviation: 0.00306
- A lower standard deviation indicates less variation across the five tested splits.

## 9. Research Conclusion

The model selected based on the highest mean 5-fold Cross-Validation Macro F1 is Random Forest.
Its mean testing Accuracy across five runs is 0.91739, while its mean testing Macro F1 is 0.92901.
The five-run testing results are used to examine performance variation across different data splits.
The final interpretation considers both average performance and standard deviation.

## 10. Experimental Notes and Limitations

The results apply to the dataset, preprocessing, feature configuration, model candidates, random states, and evaluation procedures used in this experiment.
The five runs use different splits of the same dataset. Therefore, their results describe sensitivity to data partitioning and should not be interpreted as five independent external datasets.
Model selection is based on mean Cross-Validation Macro F1, while testing metrics are reported separately.