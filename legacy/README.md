# Legacy Project Artifacts

This directory contains the original, historical implementation artifacts for the ECG Heart Disease Prediction project (circa March 2025).

## Quarantined Contents

| File | Type | Description |
| :--- | :--- | :--- |
| `test.py` | Python Script | Original inference script using static bounding boxes and contour extraction. |
| `test1.ipynb` | Jupyter Notebook | Notebook with stored evaluation plots and error traces. |
| `test.ipynb` | Jupyter Notebook | Unexecuted notebook copy. |
| `finalmodel.pkl` | Serialized Model | `sklearn.decomposition.PCA` transformer (3,060 features to 400 components). |
| `Heart_Disease_Prediction_using_ECG (4).pkl` | Serialized Model | `GridSearchCV` wrapping `VotingClassifier` ensemble ($N_{\text{features\_in}} = 400$). |
| `model_test.pkl` | Serialized Model | Exact byte-for-byte duplicate of `Heart_Disease_Prediction_using_ECG (4).pkl`. |
| `test_image.jpg` | JPEG Image | Sample input 12-lead ECG scan ($2213 \times 1572$). |
| `Normal(1).jpg` | JPEG Image | Reference template for Normal ECG rhythm. |
| `MI(1).jpg` | JPEG Image | Reference template for acute Myocardial Infarction. |
| `PMI(1).jpg` | JPEG Image | Reference template for Previous Myocardial Infarction. |
| `HB(6).jpg` | JPEG Image | Reference template for Abnormal Heartbeat. |

## Important Notice

1. **Security Warning:** As documented in official scikit-learn security guidelines, unpickling untrusted `.pkl` files can lead to arbitrary code execution. These models are kept strictly for reference and audit purposes.
2. **Dimension Discrepancy:** In `test.py`, extracted features total 3,060 columns while the classifier expects 400. In `test1.ipynb`, calling `.predict()` on `finalmodel.pkl` fails because it is a PCA transformer, not an estimator.
3. **Immutability:** Do not edit, execute, or overwrite these files. All active development is located in `ecg_lab/`.
