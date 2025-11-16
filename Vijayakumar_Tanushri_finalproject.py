# Importing the packages and libraries - required for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Machine learning libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Deep learning libraries
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam

print("All libraries imported successfully.")

# To find the library versions
import sys, sklearn, tensorflow as tf
print(sys.version)
print("sklearn:", sklearn.__version__)
print("tf:", tf.__version__)

# Loading data and describing it
df = pd.read_csv("gallstone.csv")
df.describe()
print(df.shape)
df.head()

print("\nColumn names:", list(df.columns))
print("\nData types:")
print(df.dtypes)
print("Missing values per column:\n", df.isna().sum())


# Data Preprocessing and Visualisation

# detect numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# simple missing-value handling by filling numeric NA with median
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# Separating dataset into features and output label
TARGET_COL = "Gallstone Status"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

# Check class imbalance
print("Checking for Data Imbalance:")
counts = y.value_counts()
majority = counts.max()
minority = counts.min()
imbalance_ratio = minority / majority * 100

print(f"Number of majority class samples: {majority}")
print(f"Number of minority class samples: {minority}")
print(f"Imbalance ratio (minority/majority): {imbalance_ratio:.2f}%")

# Visualizing the distribution of the target variable
plt.figure(figsize=(5,4))
sns.countplot(x=y)
plt.title("Class Distribution")
plt.show()


# Data Visualisation

# Checking for Correlation between attributes
plt.figure(figsize=(10,8))
corr = X.corr()
sns.heatmap(corr, annot=False, linewidths = 0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# Visualize the distribution of values for each attribute (histograms)
X.hist(figsize=(12,10))
plt.tight_layout()
plt.show()

# Generate a pairplot to visualize multiple pairwise bivariate distributions
sampled_cols = list(X.columns[:4]) + [TARGET_COL]
sns.pairplot(df[sampled_cols], hue=TARGET_COL)
plt.show()

# Train Test Data Split
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# Normalize the training dataset
scaler = StandardScaler()
X_train_all_std = scaler.fit_transform(X_train_all)
X_test_all_std = scaler.transform(X_test_all)
pd.DataFrame(X_train_all_std, columns=X_train_all.columns).describe()

# Metric Functions (manual except CM, ROC/AUC/BS/BSS)
def manual_metrics_from_cm(cm):
    TP, FN = cm[0][0], cm[0][1]
    FP, TN = cm[1][0], cm[1][1]
    P = TP + FN
    N = TN + FP
    total = P + N

    tpr = TP/P   # recall or sensitivity
    tnr = TN/N   # specificity
    fpr = FP/N
    fnr = FN/P
    
    precision = TP/(TP + FP)
    
    accuracy = (TP + TN)/(total)
    error_rate = 1 - accuracy
    
    f1 = 2 * precision * tpr / (precision + tpr)
    bacc = (tpr + tnr) / 2
    tss = tpr - fpr
    hss = (2 * (TP * TN - FP * FN)) / ((TP + FN) * (FN + TN) + (TP + FP) * (FP + TN))
    
    metrics = [TP, TN, FP, FN, tpr, tnr, fpr, fnr, precision, f1, accuracy, error_rate, bacc, tss, hss]
    
    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "P": P, "N": N,
        "TPR": tpr, "TNR": tnr, "FPR": fpr, "FNR": fnr,
        "Precision": precision, "Recall": tpr, "F1": f1,
        "Accuracy": accuracy, "Error_rate": error_rate,
        "Balanced_Accuracy": bacc,
        "TSS": tss, "HSS": hss
    }

def brier_and_bss(y_true, y_prob):
    bs = brier_score_loss(y_true, y_prob)
    p_ref = np.mean(y_true)
    bs_ref = np.mean((p_ref - y_true)**2)
    bss = 1 - bs / (bs_ref + 1e-12)
    return bs, bss
   
# Parameter tuning for Random Forest
rf_param_grid = {
    "n_estimators": [50, 100, 150],
    "min_samples_split": [2, 4, 6, 10]
}
rf_base = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_base, rf_param_grid, cv=5, n_jobs=-1)
rf_grid.fit(X_train_all_std, y_train_all)
best_rf_params = rf_grid.best_params_
print("Best RF params:", best_rf_params)

# Parameter Tuning for SVM
svm_param_grid = {
    "C": [0.1, 1, 3, 10],
    "kernel": ["linear"]
}
svm_base = SVC(probability=True, random_state=42)
svm_grid = GridSearchCV(svm_base, svm_param_grid, cv=5, n_jobs=-1)
svm_grid.fit(X_train_all_std, y_train_all)
best_svm_params = svm_grid.best_params_
print("Best SVM params:", best_svm_params)


# LSTM Model Construction
def build_lstm(input_dim):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(input_dim, 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model


# 10-Fold Stratified Cross-Validation for RF, SVM, and LSTM

# 10-Fold Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

metric_columns = [
    "TP","TN","FP","FN","P","N",
    "TPR","TNR","FPR","FNR",
    "Precision","Recall","F1_measure",
    "Accuracy","Error_rate","Balanced_Accuracy",
    "TSS","HSS","Brier_score","BSS","AUC"
]

rf_metrics_list = []
svm_metrics_list = []
lstm_metrics_list = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_train_all_std, y_train_all), start=1):
    X_train, X_test = X_train_all_std[train_idx], X_train_all_std[test_idx]
    y_train, y_test = y_train_all.iloc[train_idx], y_train_all.iloc[test_idx]

    # 1. RANDOM FOREST
    rf = RandomForestClassifier(
        n_estimators=best_rf_params["n_estimators"],
        min_samples_split=best_rf_params["min_samples_split"],
        random_state=42
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=[1, 0])
    m_rf = manual_metrics_from_cm(cm_rf)
    bs_rf, bss_rf = brier_and_bss(y_test.values, y_prob_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    m_rf["Brier_score"] = bs_rf
    m_rf["BSS"] = bss_rf
    m_rf["AUC"] = auc_rf
    rf_metrics_list.append(m_rf)

    # 2. SUPPORT VECTOR MACHINE
    svm = SVC(
        C=best_svm_params["C"],
        kernel=best_svm_params["kernel"],
        probability=True,
        random_state=42
    )
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    y_prob_svm = svm.predict_proba(X_test)[:, 1]

    cm_svm = confusion_matrix(y_test, y_pred_svm, labels=[1, 0])
    m_svm = manual_metrics_from_cm(cm_svm)
    bs_svm, bss_svm = brier_and_bss(y_test.values, y_prob_svm)
    auc_svm = roc_auc_score(y_test, y_prob_svm)

    m_svm["Brier_score"] = bs_svm
    m_svm["BSS"] = bss_svm
    m_svm["AUC"] = auc_svm
    svm_metrics_list.append(m_svm)

    # 3. LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

    lstm = build_lstm(input_dim=X_train.shape[1])
    lstm.fit(
        X_train_lstm, y_train,
        epochs=30, batch_size=32, verbose=0,
        validation_data=(X_test_lstm, y_test)
    )

    y_prob_lstm = lstm.predict(X_test_lstm).ravel()
    y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

    cm_lstm = confusion_matrix(y_test, y_pred_lstm, labels=[1, 0])
    m_lstm = manual_metrics_from_cm(cm_lstm)
    bs_lstm, bss_lstm = brier_and_bss(y_test.values, y_prob_lstm)
    auc_lstm = roc_auc_score(y_test, y_prob_lstm)

    m_lstm["Brier_score"] = bs_lstm
    m_lstm["BSS"] = bss_lstm
    m_lstm["AUC"] = auc_lstm
    lstm_metrics_list.append(m_lstm)

    # PRINT TABLE FOR THIS ITERATION
    metrics_all_df = pd.DataFrame(
        [m_rf, m_svm, m_lstm],
        columns=metric_columns,
        index=["RF", "SVM", "LSTM"]
    )

    print(f"\nIteration {fold_idx}:")
    print(f"Metrics for all Algorithms in Iteration {fold_idx}\n")
    print(metrics_all_df.round(2).T)
    print("\n")

print("10-Fold Cross-Validation Complete")

# Create DataFrames for all folds and compute averages
iter_index = [f"iter{i}" for i in range(1, 11)]

rf_df   = pd.DataFrame(rf_metrics_list,  index=iter_index, columns=metric_columns)
svm_df  = pd.DataFrame(svm_metrics_list, index=iter_index, columns=metric_columns)
lstm_df = pd.DataFrame(lstm_metrics_list,index=iter_index, columns=metric_columns)

print("\nRandom Forest Metrics per Fold")
print(rf_df.T.round(2).to_string())

print("\nSVM Metrics per Fold")
print(svm_df.T.round(2).to_string())

print("\nLSTM Metrics per Fold")
print(lstm_df.T.round(2).to_string())

# Average performance across 10 folds
rf_avg   = rf_df.mean().to_frame(name="RF")
svm_avg  = svm_df.mean().to_frame(name="SVM")
lstm_avg = lstm_df.mean().to_frame(name="LSTM")

avg_all = pd.concat([rf_avg, svm_avg, lstm_avg], axis=1)

print("\n Average Performance Across 10 Folds: ")
print(avg_all.round(3))

# 10. Evaluate ROC & AUC on the Hold-Out Test Set

rf_final = RandomForestClassifier(
    n_estimators=best_rf_params["n_estimators"],
    min_samples_split=best_rf_params["min_samples_split"],
    random_state=42
).fit(X_train_all_std, y_train_all)

svm_final = SVC(
    C=best_svm_params["C"],
    kernel=best_svm_params["kernel"],
    probability=True,
    random_state=42
).fit(X_train_all_std, y_train_all)

input_dim = X_train_all_std.shape[1]
lstm_final = build_lstm(input_dim)
X_train_lstm = X_train_all_std.reshape((X_train_all_std.shape[0], X_train_all_std.shape[1], 1))
X_test_lstm  = X_test_all_std.reshape((X_test_all_std.shape[0],  X_test_all_std.shape[1],  1))
lstm_final.fit(X_train_lstm, y_train_all, epochs=50, batch_size=32, verbose=0,
               validation_data=(X_test_lstm, y_test_all))

# Predictions for ROC
rf_probs  = rf_final.predict_proba(X_test_all_std)[:, 1]
svm_probs = svm_final.predict_proba(X_test_all_std)[:, 1]
lstm_probs= lstm_final.predict(X_test_lstm).ravel()

fpr_rf,  tpr_rf,  _ = roc_curve(y_test_all, rf_probs)
fpr_svm, tpr_svm, _ = roc_curve(y_test_all, svm_probs)
fpr_lstm,tpr_lstm,_ = roc_curve(y_test_all, lstm_probs)

auc_rf   = roc_auc_score(y_test_all, rf_probs)
auc_svm  = roc_auc_score(y_test_all, svm_probs)
auc_lstm = roc_auc_score(y_test_all, lstm_probs)

plt.figure(figsize=(6,6))
plt.plot(fpr_rf,  tpr_rf,  label=f'RF (AUC={auc_rf:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC={auc_svm:.2f})')
plt.plot(fpr_lstm,tpr_lstm,label=f'LSTM (AUC={auc_lstm:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# Individual ROC Curves for Each Model

# Random Forest ROC
plt.figure(figsize=(6,6))
plt.plot(fpr_rf, tpr_rf, label=f'RF (AUC={auc_rf:.2f})', color='darkorange', linewidth=2)
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve - Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# SVM ROC
plt.figure(figsize=(6,6))
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC={auc_svm:.2f})', color='steelblue', linewidth=2)
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve - Support Vector Machine (SVM)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# LSTM ROC
plt.figure(figsize=(6,6))
plt.plot(fpr_lstm, tpr_lstm, label=f'LSTM (AUC={auc_lstm:.2f})', color='green', linewidth=2)
plt.plot([0,1],[0,1],'k--')
plt.title("ROC Curve - LSTM Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Model Comparison Summary
print("\nModel Comparison Summary (Mean Across 10 Folds)")
print(avg_all.round(3))

best_model = avg_all.loc["Accuracy"].idxmax()
print(f"\nBest Overall Model Based on Mean Accuracy: {best_model}")
