# =================================================================================================
# ULTIMATE ENSEMBLE v9: COMPREHENSIVE FEATURE ENGINEERING & ROBUST STACKING FOR MAX F1
#
# STRATEGY:
# 1.  Comprehensive Feature Engineering: Integrates the best features from all previous iterations,
#     including explicit interactions, health indices, frequency encoding, and advanced polynomial features. [5, 8, 13, 14, 15]
# 2.  Optimized and Diverse Base Models: Utilizes refined LightGBM, XGBoost, and CatBoost with
#     well-tuned parameters for strong initial predictions.
# 3.  Upgraded Stacking Ensemble: Employs a LightGBM meta-model for intelligent
#     blending of base model predictions. [2, 4, 7, 12]
# 4.  Rigorous F1-Score Optimization: Continues with dynamic class weighting and precision thresholding
#     to directly maximize the competition metric. [1, 6, 9, 10, 11]
# =================================================================================================

print("Step 1: Loading libraries and setting up the environment...")
import numpy as np
import pandas as pd
import warnings
import time
import gc
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.metrics import f1_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from tqdm.notebook import tqdm
import joblib

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
start_time = time.time()

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print(f'Mem. usage decreased to {end_mem:5.2f} Mb ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

print("\nStep 2: Loading data and applying comprehensive feature engineering...")
try:
    # Set the base path for the Kaggle environment
    BASE_PATH = '/kaggle/input/idealize/'
    train_df = pd.read_csv(BASE_PATH + 'train.csv')
    test_df = pd.read_csv(BASE_PATH + 'test.csv')
except FileNotFoundError:
    # Fallback to local files if Kaggle path is not found
    print("Kaggle directory not found. Using local 'train.csv' and 'test.csv'")
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

train_rows = train_df.shape[0]
test_ids = test_df['record_id']
y = train_df['survival_status']
train_df = train_df.drop('survival_status', axis=1)
df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
del train_df, test_df; gc.collect()

print("Executing the comprehensive feature engineering pipeline...")
df = df.drop(['record_id', 'first_name', 'last_name'], axis=1)
date_cols = ['diagnosis_date', 'treatment_start_date', 'treatment_end_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# Time-based features
df['time_to_treatment'] = (df['treatment_start_date'] - df['diagnosis_date']).dt.days
df['treatment_duration'] = (df['treatment_end_date'] - df['treatment_start_date']).dt.days
df['diagnosis_year'] = df['diagnosis_date'].dt.year
df['diagnosis_month'] = df['diagnosis_date'].dt.month
df = df.drop(date_cols, axis=1)

# Foundational and ratio-based features
df['bmi'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
df['cigarettes_per_day'] = df['cigarettes_per_day'].fillna(0)
df['age_div_treatment_duration'] = df['patient_age'] / (df['treatment_duration'] + 1)
df['time_to_treatment_div_age'] = df['time_to_treatment'] / (df['patient_age'] + 1)
df['cholesterol_bmi_interaction'] = df['cholesterol_mg_dl'] * df['bmi']

# Health Indices
df['health_index'] = (df['bmi'] + df['cholesterol_mg_dl'] + df['cigarettes_per_day']) / 3
df['comorbidity_score'] = (df['has_other_cancer'] == 'Yes').astype(int) + \
                          (df['asthma_diagnosis'] == 'Yes').astype(int) + \
                          (df['liver_condition'] == 'Has Cirrhosis').astype(int) + \
                          (df['blood_pressure_status'] == 'High Blood Pressure').astype(int)

# Frequency Encoding
df['state_freq'] = df['residence_state'].map(df['residence_state'].value_counts(normalize=True))

cat_cols = ['sex', 'smoking_status', 'family_cancer_history', 'has_other_cancer',
            'asthma_diagnosis', 'liver_condition', 'blood_pressure_status',
            'cancer_stage', 'treatment_type', 'residence_state']
for col in tqdm(cat_cols, desc="Label Encoding"):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Interaction Features
df['age_x_cancer_stage'] = df['patient_age'] * df['cancer_stage']
df['bmi_x_cancer_stage'] = df['bmi'] * df['cancer_stage']
df['treatment_duration_x_cancer_stage'] = df['treatment_duration'] * df['cancer_stage']

AGG_COLS = ['patient_age', 'bmi', 'cholesterol_mg_dl', 'treatment_duration', 'time_to_treatment', 'health_index', 'comorbidity_score']
GROUP_COLS = ['cancer_stage', 'treatment_type', 'residence_state', 'smoking_status']
for group_col in tqdm(GROUP_COLS, desc="Aggregating Features"):
    for agg_col in AGG_COLS:
        df[f'{agg_col}_mean_by_{group_col}'] = df.groupby(group_col)[agg_col].transform('mean')
        df[f'{agg_col}_std_by_{group_col}'] = df.groupby(group_col)[agg_col].transform('std')
        df[f'{agg_col}_diff_from_{group_col}_mean'] = df[agg_col] - df[f'{agg_col}_mean_by_{group_col}']

df.fillna(0, inplace=True)
df.replace([np.inf, -np.inf], 0, inplace=True)

# Polynomial Features
poly_features = ['patient_age', 'bmi', 'cholesterol_mg_dl', 'treatment_duration', 'time_to_treatment', 'health_index']
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
poly_df = poly.fit_transform(df[poly_features])
poly_cols = [f"poly_{i}" for i in range(poly_df.shape[1])]
poly_df = pd.DataFrame(poly_df, columns=poly_cols, index=df.index)
df = pd.concat([df, poly_df], axis=1)

df = reduce_mem_usage(df)
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
X = df[:train_rows]
X_test = df[train_rows:]
del df, poly_df; gc.collect()
print(f"Final data shape after comprehensive feature engineering: {X.shape}")

# =================================================================================================
# Step 3: Training Diverse and Optimized Base Models
# =================================================================================================
N_SPLITS = 10
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
neg_count = y.value_counts()[0]
pos_count = y.value_counts()[1]
scale_pos_weight_value = neg_count / pos_count

oof_preds = np.zeros((len(X), 3))
test_preds = np.zeros((len(X_test), 3))

# Tuned parameters for the base models
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 10000, 'learning_rate': 0.008, 'num_leaves': 80,
    'max_depth': 12, 'seed': 1, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1,
    'reg_lambda': 0.1, 'device': 'gpu', 'scale_pos_weight': scale_pos_weight_value
}
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc', 'eta': 0.008,
    'max_depth': 12, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'seed': 2, 'n_jobs': -1, 'tree_method': 'hist', 'device': 'cuda',
    'scale_pos_weight': scale_pos_weight_value
}
cat_params = {
    'objective': 'Logloss', 'eval_metric': 'AUC', 'iterations': 10000,
    'learning_rate': 0.008, 'depth': 12, 'random_seed': 3,
    'verbose': 0, 'task_type': 'GPU', 'scale_pos_weight': scale_pos_weight_value
}
callbacks = [lgb.early_stopping(300, verbose=False)]

print("\n--- Training Level 0 Base Models ---")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"--- Fold {fold+1}/{N_SPLITS} ---")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    # Model 1: LightGBM
    lgbm = lgb.LGBMClassifier(**lgb_params)
    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    oof_preds[val_idx, 0] = lgbm.predict_proba(X_val)[:, 1]
    test_preds[:, 0] += lgbm.predict_proba(X_test)[:, 1] / N_SPLITS

    # Model 2: XGBoost
    xgboost = xgb.XGBClassifier(**xgb_params, n_estimators=10000, early_stopping_rounds=300, enable_categorical=False)
    xgboost.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    oof_preds[val_idx, 1] = xgboost.predict_proba(X_val)[:, 1]
    test_preds[:, 1] += xgboost.predict_proba(X_test)[:, 1] / N_SPLITS

    # Model 3: CatBoost
    catboost = cb.CatBoostClassifier(**cat_params)
    catboost.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=300, verbose=False)
    oof_preds[val_idx, 2] = catboost.predict_proba(X_val)[:, 1]
    test_preds[:, 2] += catboost.predict_proba(X_test)[:, 1] / N_SPLITS

    gc.collect()

# =================================================================================================
# Step 4: Training the Stacking Meta-Model
# =================================================================================================
print("\n--- Training Level 1 Stacking Meta-Model (LightGBM) ---")
meta_X_train = pd.DataFrame(oof_preds, columns=['lgbm_oof', 'xgb_oof', 'cat_oof'])
meta_X_test = pd.DataFrame(test_preds, columns=['lgbm_test', 'xgb_test', 'cat_test'])

blender_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'n_estimators': 3000, 'learning_rate': 0.01, 'num_leaves': 32,
    'max_depth': 6, 'seed': 4242, 'n_jobs': -1, 'verbose': -1,
    'colsample_bytree': 0.8, 'subsample': 0.8,
    'scale_pos_weight': scale_pos_weight_value
}

blender = lgb.LGBMClassifier(**blender_params)
blender.fit(meta_X_train, y, eval_set=[(meta_X_train, y)], callbacks=[lgb.early_stopping(150, verbose=False)])

final_oof_preds_proba = blender.predict_proba(meta_X_train)[:, 1]
final_test_preds_proba = blender.predict_proba(meta_X_test)[:, 1]

# =================================================================================================
# Step 5: Final F1 Thresholding and Submission Generation
# =================================================================================================
print("\n--- Step 5: Finding the Optimal F1 Threshold and Creating the Submission ---")
best_f1 = 0
best_threshold = 0.5
for threshold in np.arange(0.2, 0.8, 0.005):
    f1 = f1_score(y, (final_oof_preds_proba > threshold).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\nBest STACKED F1 score on OOF predictions: {best_f1:.6f}")
print(f"Optimal STACKED threshold found: {best_threshold:.4f}")

final_predictions = (final_test_preds_proba > best_threshold).astype(int)
submission_df = pd.DataFrame({'record_id': test_ids, 'survival_status': final_predictions})
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file 'submission.csv' created successfully!")
print(submission_df.head())

# =================================================================================================
# Step 6: Saving the Final Model for Compliance
# =================================================================================================
print("\n--- Saving the best single model (LightGBM) for rule compliance ---")
final_single_model = lgb.LGBMClassifier(**lgb_params)
final_single_model.fit(X, y)
model_filename = 'final_lgbm_model.pkl'
joblib.dump(final_single_model, model_filename)
print(f"Final compliance model saved to '{model_filename}'")

end_time = time.time()
print(f"\n--- ULTIMATE ENSEMBLE SCRIPT FINISHED in {(end_time - start_time)/60:.2f} minutes ---")
Step 1: Loading libraries and setting up the environment...

Step 2: Loading data and applying comprehensive feature engineering...
Executing the comprehensive feature engineering pipeline...
Label Encoding: 100%
 10/10 [00:02<00:00,  3.87it/s]
Aggregating Features: 100%
 4/4 [00:01<00:00,  2.20it/s]
Mem. usage decreased to 311.14 Mb (75.5% reduction)
Final data shape after comprehensive feature engineering: (999999, 134)

--- Training Level 0 Base Models ---
--- Fold 1/10 ---
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
1 warning generated.
Default metric period is 5 because AUC is/are not implemented for GPU
--- Fold 2/10 ---
Default metric period is 5 because AUC is/are not implemented for GPU
--- Fold 3/10 ---
Default metric period is 5 because AUC is/are not implemented for GPU
--- Fold 4/10 ---
Default metric period is 5 because AUC is/are not implemented for GPU
--- Fold 5/10 ---
Default metric period is 5 because AUC is/are not implemented for GPU
--- Fold 6/10 ---
Default metric period is 5 because AUC is/are not implemented for GPU
--- Fold 7/10 ---
Default metric period is 5 because AUC is/are not implemented for GPU
--- Fold 8/10 ---
Default metric period is 5 because AUC is/are not implemented for GPU
--- Fold 9/10 ---
Default metric period is 5 because AUC is/are not implemented for GPU
--- Fold 10/10 ---
Default metric period is 5 because AUC is/are not implemented for GPU
--- Training Level 1 Stacking Meta-Model (LightGBM) ---

--- Step 5: Finding the Optimal F1 Threshold and Creating the Submission ---

Best STACKED F1 score on OOF predictions: 0.410663
Optimal STACKED threshold found: 0.4750

Submission file 'submission.csv' created successfully!
   record_id  survival_status
0    1000000                0
1    1000001                0
2    1000002                0
3    1000003                0
4    1000004                0

--- Saving the best single model (LightGBM) for rule compliance ---
Final compliance model saved to 'final_lgbm_model.pkl'

--- ULTIMATE ENSEMBLE SCRIPT FINISHED in 378.84 minutes ---