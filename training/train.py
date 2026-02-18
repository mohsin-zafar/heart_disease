"""
Heart Disease Detection - Training Script
==========================================
This script trains multiple ML models for heart disease prediction,
compares their performance, and saves the best model.

Author: Heart Disease Detection Project
Date: 2024
"""

# ============================================
# STEP 1: IMPORT LIBRARIES
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import os

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 60)
print("HEART DISEASE DETECTION - MODEL TRAINING")
print("=" * 60)

# ============================================
# STEP 2: LOAD DATASET
# ============================================
print("\n📁 STEP 1: Loading Dataset...")
print("-" * 40)

# For Google Colab - upload your dataset first
# from google.colab import files
# uploaded = files.upload()

# Load the dataset
# Change this path based on your environment
try:
    # Try loading from current directory (Google Colab)
    df = pd.read_csv('heart_disease_dataset.csv')
except FileNotFoundError:
    try:
        # Try loading from dataset folder (local)
        df = pd.read_csv('../dataset/heart_disease_dataset.csv')
    except FileNotFoundError:
        # Try loading from parent directory
        df = pd.read_csv('dataset/heart_disease_dataset.csv')

print(f"✅ Dataset loaded successfully!")
print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================
print("\n📊 STEP 2: Exploratory Data Analysis")
print("-" * 40)

# Display first few rows
print("\n🔍 First 5 rows of the dataset:")
print(df.head())

# Dataset info
print("\n📋 Dataset Information:")
print(f"   • Total Samples: {len(df)}")
print(f"   • Total Features: {df.shape[1] - 1}")
print(f"   • Target Column: heart_disease")

# Check for missing values
print("\n🔎 Missing Values Check:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("   ✅ No missing values found!")
else:
    print(missing[missing > 0])

# Statistical summary
print("\n📈 Statistical Summary:")
print(df.describe().round(2))

# Target distribution
print("\n🎯 Target Variable Distribution:")
target_counts = df['heart_disease'].value_counts()
print(f"   • No Disease (0): {target_counts[0]} ({target_counts[0]/len(df)*100:.1f}%)")
print(f"   • Disease (1): {target_counts[1]} ({target_counts[1]/len(df)*100:.1f}%)")

# ============================================
# STEP 4: DATA VISUALIZATION
# ============================================
print("\n📊 STEP 3: Data Visualization")
print("-" * 40)

# Create figure for visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Heart Disease Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Target Distribution
ax1 = axes[0, 0]
colors = ['#2ecc71', '#e74c3c']
target_counts.plot(kind='bar', ax=ax1, color=colors, edgecolor='black')
ax1.set_title('Target Distribution', fontweight='bold')
ax1.set_xlabel('Heart Disease')
ax1.set_ylabel('Count')
ax1.set_xticklabels(['No Disease', 'Disease'], rotation=0)
for i, v in enumerate(target_counts):
    ax1.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 2. Age Distribution
ax2 = axes[0, 1]
df['age'].hist(bins=20, ax=ax2, color='#3498db', edgecolor='black', alpha=0.7)
ax2.set_title('Age Distribution', fontweight='bold')
ax2.set_xlabel('Age')
ax2.set_ylabel('Frequency')

# 3. Age vs Heart Disease
ax3 = axes[0, 2]
df.boxplot(column='age', by='heart_disease', ax=ax3)
ax3.set_title('Age by Heart Disease Status', fontweight='bold')
ax3.set_xlabel('Heart Disease')
ax3.set_ylabel('Age')
plt.suptitle('')

# 4. Correlation with Target
ax4 = axes[1, 0]
correlations = df.corr()['heart_disease'].drop('heart_disease').sort_values()
correlations.plot(kind='barh', ax=ax4, color=['#e74c3c' if x < 0 else '#2ecc71' for x in correlations])
ax4.set_title('Feature Correlation with Target', fontweight='bold')
ax4.set_xlabel('Correlation Coefficient')

# 5. Gender Distribution
ax5 = axes[1, 1]
gender_disease = pd.crosstab(df['sex'], df['heart_disease'])
gender_disease.plot(kind='bar', ax=ax5, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax5.set_title('Heart Disease by Gender', fontweight='bold')
ax5.set_xlabel('Gender (0=Female, 1=Male)')
ax5.set_ylabel('Count')
ax5.set_xticklabels(['Female', 'Male'], rotation=0)
ax5.legend(['No Disease', 'Disease'])

# 6. Chest Pain Type Distribution
ax6 = axes[1, 2]
cp_disease = pd.crosstab(df['chest_pain_type'], df['heart_disease'])
cp_disease.plot(kind='bar', ax=ax6, color=['#2ecc71', '#e74c3c'], edgecolor='black')
ax6.set_title('Heart Disease by Chest Pain Type', fontweight='bold')
ax6.set_xlabel('Chest Pain Type')
ax6.set_ylabel('Count')
ax6.legend(['No Disease', 'Disease'])

plt.tight_layout()
plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ EDA visualization saved as 'eda_visualization.png'")

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', 
            linewidths=0.5, square=True)
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")

# ============================================
# STEP 5: FEATURE ANALYSIS
# ============================================
print("\n🔬 STEP 4: Feature Analysis")
print("-" * 40)

# Feature names
feature_names = df.columns.tolist()
feature_names.remove('heart_disease')
print(f"\n📋 Features ({len(feature_names)}):")
for i, feat in enumerate(feature_names, 1):
    print(f"   {i:2d}. {feat}")

# Feature statistics by target
print("\n📊 Feature Statistics by Heart Disease Status:")
print(df.groupby('heart_disease').mean().round(2).T)

# ============================================
# STEP 6: DATA PREPARATION
# ============================================
print("\n⚙️ STEP 5: Data Preparation")
print("-" * 40)

# Separate features and target
X = df.drop('heart_disease', axis=1)
y = df['heart_disease']

print(f"   • Features shape: {X.shape}")
print(f"   • Target shape: {y.shape}")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\n📊 Train-Test Split (80/20):")
print(f"   • Training samples: {len(X_train)}")
print(f"   • Testing samples: {len(X_test)}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("   ✅ Features scaled using StandardScaler")

# ============================================
# STEP 7: MODEL TRAINING
# ============================================
print("\n🤖 STEP 6: Model Training")
print("-" * 40)

# Dictionary to store models and results
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=100),
    'Support Vector Machine': SVC(random_state=RANDOM_STATE, probability=True)
}

# Dictionary to store results
results = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': [],
    'ROC-AUC': [],
    'Specificity': []
}

# Train and evaluate each model
trained_models = {}
for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    
    # Store results
    results['Model'].append(name)
    results['Accuracy'].append(accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1 Score'].append(f1)
    results['ROC-AUC'].append(roc_auc)
    results['Specificity'].append(specificity)
    
    print(f"   ✅ {name} trained successfully!")
    print(f"      • Accuracy: {accuracy:.4f}")
    print(f"      • Recall: {recall:.4f}")
    print(f"      • Precision: {precision:.4f}")
    print(f"      • F1 Score: {f1:.4f}")
    print(f"      • ROC-AUC: {roc_auc:.4f}")

# ============================================
# STEP 8: MODEL COMPARISON
# ============================================
print("\n📊 STEP 7: Model Comparison")
print("-" * 40)

# Create results dataframe
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Model')
results_df = results_df.round(4)

print("\n🏆 Model Performance Comparison:")
print(results_df.to_string())

# Save results to CSV
results_df.to_csv('model_comparison.csv')
print("\n✅ Model comparison saved to 'model_comparison.csv'")

# Visualization: Model Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart comparison
ax1 = axes[0]
results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']].plot(
    kind='bar', ax=ax1, colormap='viridis', edgecolor='black', width=0.8
)
ax1.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
ax1.set_xlabel('Model')
ax1.set_ylabel('Score')
ax1.set_ylim([0, 1.1])
ax1.legend(loc='lower right')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
for container in ax1.containers:
    ax1.bar_label(container, fmt='%.2f', fontsize=7)

# ROC-AUC comparison
ax2 = axes[1]
colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
for idx, (name, model) in enumerate(trained_models.items()):
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax2.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

ax2.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate', fontsize=11)
ax2.set_ylabel('True Positive Rate', fontsize=11)
ax2.set_title('ROC Curves Comparison', fontweight='bold', fontsize=12)
ax2.legend(loc='lower right', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Model comparison visualization saved as 'model_comparison.png'")

# ============================================
# STEP 9: CONFUSION MATRICES
# ============================================
print("\n📊 STEP 8: Confusion Matrices")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, (name, model) in enumerate(trained_models.items()):
    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Confusion matrices saved as 'confusion_matrices.png'")

# ============================================
# STEP 10: SELECT BEST MODEL
# ============================================
print("\n🏆 STEP 9: Selecting Best Model")
print("-" * 40)

# Select best model based on ROC-AUC (primary metric for medical diagnosis)
# You can change this to Recall if sensitivity is more important
best_model_idx = results_df['ROC-AUC'].argmax()
best_model_name = results_df.index[best_model_idx]
best_model = trained_models[best_model_name]

print(f"\n🥇 BEST MODEL: {best_model_name}")
print(f"\n   Performance Metrics:")
print(f"   • Accuracy:    {results_df.loc[best_model_name, 'Accuracy']:.4f}")
print(f"   • Precision:   {results_df.loc[best_model_name, 'Precision']:.4f}")
print(f"   • Recall:      {results_df.loc[best_model_name, 'Recall']:.4f}")
print(f"   • F1 Score:    {results_df.loc[best_model_name, 'F1 Score']:.4f}")
print(f"   • ROC-AUC:     {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")
print(f"   • Specificity: {results_df.loc[best_model_name, 'Specificity']:.4f}")

# Detailed classification report
print(f"\n📋 Classification Report for {best_model_name}:")
y_pred_best = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best, target_names=['No Disease', 'Disease']))

# ============================================
# STEP 11: FEATURE IMPORTANCE
# ============================================
print("\n📊 STEP 10: Feature Importance Analysis")
print("-" * 40)

# Get feature importance (works best for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
elif hasattr(best_model, 'coef_'):
    importances = np.abs(best_model.coef_[0])
else:
    importances = None

if importances is not None:
    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
             color='steelblue', edgecolor='black')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Feature Importance - {best_model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Feature importance plot saved as 'feature_importance.png'")
    
    print("\n📋 Top 5 Most Important Features:")
    for idx, row in feature_importance.tail(5).iloc[::-1].iterrows():
        print(f"   • {row['Feature']}: {row['Importance']:.4f}")

# ============================================
# STEP 12: SAVE MODEL AND SCALER
# ============================================
print("\n💾 STEP 11: Saving Model and Scaler")
print("-" * 40)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the best model
model_filename = 'model/model.pkl'
joblib.dump(best_model, model_filename)
print(f"✅ Best model saved as '{model_filename}'")

# Save the scaler (important for preprocessing new data)
scaler_filename = 'model/scaler.pkl'
joblib.dump(scaler, scaler_filename)
print(f"✅ Scaler saved as '{scaler_filename}'")

# Save feature names
feature_names_file = 'model/feature_names.pkl'
joblib.dump(feature_names, feature_names_file)
print(f"✅ Feature names saved as '{feature_names_file}'")

# Save model info
model_info = {
    'model_name': best_model_name,
    'accuracy': results_df.loc[best_model_name, 'Accuracy'],
    'precision': results_df.loc[best_model_name, 'Precision'],
    'recall': results_df.loc[best_model_name, 'Recall'],
    'f1_score': results_df.loc[best_model_name, 'F1 Score'],
    'roc_auc': results_df.loc[best_model_name, 'ROC-AUC'],
    'specificity': results_df.loc[best_model_name, 'Specificity'],
    'features': feature_names
}
joblib.dump(model_info, 'model/model_info.pkl')
print(f"✅ Model info saved as 'model/model_info.pkl'")

# ============================================
# STEP 13: MEDICAL INTERPRETATION
# ============================================
print("\n🏥 STEP 12: Medical Interpretation Insights")
print("-" * 40)

print("""
📋 KEY MEDICAL INSIGHTS:

1. IMPORTANT RISK FACTORS (Based on Feature Analysis):
   • Chest Pain Type: Different types indicate varying risk levels
   • ST Depression: Higher values suggest cardiac stress
   • Max Heart Rate: Lower values may indicate cardiac issues
   • Number of Major Vessels: More blocked vessels = higher risk
   • Thalassemia: Blood disorder affecting oxygen transport

2. MODEL INTERPRETATION:
   • High Recall: Minimizes missed disease cases (false negatives)
   • High Precision: Reduces unnecessary follow-ups (false positives)
   • ROC-AUC: Overall discriminative ability between classes

3. CLINICAL RECOMMENDATIONS:
   • Use this model as a SCREENING TOOL, not for final diagnosis
   • Always combine with clinical judgment and additional tests
   • Patients flagged as "Disease" should undergo further evaluation
   • Regular updates with new data will improve accuracy

⚠️ DISCLAIMER: This is a machine learning model for educational purposes.
   Medical decisions should always be made by qualified healthcare professionals.
""")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)

print(f"""
📊 SUMMARY:
   • Dataset: {len(df)} samples, {len(feature_names)} features
   • Best Model: {best_model_name}
   • ROC-AUC Score: {results_df.loc[best_model_name, 'ROC-AUC']:.4f}
   • Recall: {results_df.loc[best_model_name, 'Recall']:.4f}

📁 FILES GENERATED:
   • model/model.pkl (Best trained model)
   • model/scaler.pkl (Feature scaler)
   • model/feature_names.pkl (Feature names)
   • model/model_info.pkl (Model metadata)
   • model_comparison.csv (All model results)
   • model_comparison.png (Performance chart)
   • confusion_matrices.png (Confusion matrices)
   • correlation_heatmap.png (Feature correlations)
   • eda_visualization.png (EDA plots)
   • feature_importance.png (Feature importance)

🚀 NEXT STEPS:
   1. Download model.pkl and scaler.pkl from the 'model' folder
   2. Place them in your project's 'model/' directory
   3. Run the backend and frontend applications
""")

print("\n🎉 Heart Disease Detection Model Training Complete!")
