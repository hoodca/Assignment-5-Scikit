# Assignment-5-Scikit
# Breast Cancer Classification Model Comparison

## Project Purpose
This project uses Scikit-learn's built-in **Breast Cancer Wisconsin (Diagnostic)** dataset to train and evaluate three machine learning classification models. The goal is to compare model performance using standard classification metrics and identify the model that best balances predictive quality and reliability.

## Dataset
- **Source:** `sklearn.datasets.load_breast_cancer`
- **Samples:** 569
- **Features:** 30 numeric features computed from digitized images of breast mass cell nuclei
- **Target:** Binary classification (`malignant` vs `benign`)

## Standard Data Division Process
The project follows the standard supervised learning workflow:
1. Load and inspect data
2. Split data into training and testing subsets using `train_test_split`
   - `test_size=0.2` (80/20 split)
   - `stratify=y` to preserve class proportions
   - `random_state=42` for reproducibility
3. Train models on training data only
4. Evaluate on held-out test data
5. Use 5-fold stratified cross-validation on the training set for stability checks

## Class Design and Implementation
The implementation is object-oriented and centered around one main class plus one data container:

### 1) `ModelResult` (Data Class)
A structured container for model metrics.

#### Attributes
- `model_name`: Name of the model
- `accuracy`: Test accuracy
- `precision`: Test precision
- `recall`: Test recall
- `f1`: F1 score on test set
- `roc_auc`: ROC-AUC score on test set
- `cv_accuracy_mean`: Mean cross-validation accuracy on training set
- `cv_accuracy_std`: Standard deviation of cross-validation accuracy
- `tn`, `fp`, `fn`, `tp`: Confusion matrix components

### 2) `BreastCancerModelComparator`
Encapsulates the complete ML pipeline from data loading through evaluation.

#### Constructor Attributes
- `test_size`: Fraction of data used for testing
- `random_state`: Seed for reproducible splits and model randomness
- `feature_names`: Dataset feature names
- `target_names`: Class names
- `X`, `y`: Full feature matrix and target vector
- `X_train`, `X_test`, `y_train`, `y_test`: Train/test splits
- `models`: Dictionary of model pipelines
- `results`: List of `ModelResult` objects

#### Methods
- `load_data()`
  - Loads the breast cancer dataset into data structures used by the class
- `split_data()`
  - Performs stratified train/test split
- `build_models()`
  - Defines the three models as pipelines:
    1. Logistic Regression (+ StandardScaler)
    2. Support Vector Machine (+ StandardScaler)
    3. Random Forest
- `evaluate_models()`
  - Fits each model and computes:
    - Accuracy, Precision, Recall, F1, ROC-AUC
    - Confusion matrix values (TN, FP, FN, TP)
    - 5-fold stratified cross-validation mean/std accuracy
  - Returns a sorted pandas DataFrame of results
- `run()`
  - End-to-end execution wrapper for all steps

### Script Entry Point
- `main()`
  - Creates `BreastCancerModelComparator`
  - Runs the full comparison
  - Prints a formatted table of metrics

## How to Run
1. Open a terminal in the project folder:
   - `cd breast_cancer_ml`
2. Install dependencies:
   - `pip install scikit-learn pandas`
3. Execute:
   - `python breast_cancer_models.py`

## Metrics Used
- **Accuracy:** Overall correctness
- **Precision:** Correctness of positive predictions
- **Recall:** Ability to find true positives
- **F1 Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Class-separation quality across thresholds
- **Cross-Validation Accuracy (Mean/Std):** Stability and generalization signal

## Model Comparison Summary
Using the current run (`random_state=42`, 80/20 split), **Logistic Regression** and **SVM** are essentially tied on test-set classification quality: both reached **Accuracy = 0.9825**, **Precision = 0.9861**, **Recall = 0.9861**, and **F1 = 0.9861**. However, Logistic Regression shows a small edge in ranking quality and stability with **ROC-AUC = 0.9954** (vs. SVM `0.9950`) and higher 5-fold CV mean accuracy (**0.9780** vs. `0.9692`) with lower CV variance (**0.0098** vs. `0.0146`). Based on those metrics, Logistic Regression is the best overall choice in this run. **Random Forest** is competitive and interpretable via feature importance, but here it trails on core metrics (**F1 = 0.9583**, **Accuracy = 0.9474**) and has more false positives/false negatives.

### Pros and Cons by Model
- **Logistic Regression**
  - Pros: Strong all-around performance, stable CV behavior, fast training, high interpretability of coefficients
  - Cons: Assumes mostly linear decision boundaries in transformed feature space
- **Support Vector Machine**
  - Pros: Excellent classification performance, good handling of complex boundaries
  - Cons: Slightly less stable CV in this run, less interpretable, can be slower at scale
- **Random Forest**
  - Pros: Captures nonlinear interactions, robust to feature scaling, useful feature-importance outputs
  - Cons: Lower performance than the other two models in this experiment, larger model complexity

## Limitations
- Evaluation uses a single train/test split for final test metrics
- Hyperparameter tuning is limited (default or lightly tuned settings)
- Dataset is relatively small and may not represent all real-world clinical populations
- This is an educational project and not a clinical diagnostic system

## Notes
I am aware that some of the code may include content that has not been covered in the course, which is why I am including this reference section to explain the source of said code:
- Mckinney, Wes. Python for Data Analysis : Data Wrangling with Pandas, NumPy, and IPython. 2nd ed., Sebastopol, Ca, O’reilly Media, Inc., October, 2017.
- Müller, Andreas C, and Sarah Guido. Introduction to Machine Learning with Python: A Guide for Data Scientists. Beijing, O’reilly, 2017.
