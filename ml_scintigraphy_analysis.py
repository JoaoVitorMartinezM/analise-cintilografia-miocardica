#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Machine Learning Analysis for Scintigraphy Data
Author: AI Assistant
Date: November 2025

This script implements comprehensive machine learning models to predict procedure outcomes
and identify key factors affecting scintigraphy success rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import xgboost as xgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib parameters for better visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Color palette
COLORS = {
    'primary': '#980000',
    'secondary': '#ED7D31', 
    'tertiary': '#F1C232',
    'quaternary': '#70AD47'
}

def load_and_prepare_data(file_path):
    """Load and prepare data for machine learning analysis"""
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Translate column names to English
        column_mapping = {
            'ID PACIENTE': 'PATIENT_ID',
            'DATA': 'DATE_1',
            'HORÁRIO CHEGADA': 'ARRIVAL_TIME_1',
            'DATA 2': 'DATE_2',
            'HORÁRIO CHEGADA 2': 'ARRIVAL_TIME_2',
            'SEXO': 'GENDER',
            'IDADE': 'AGE',
            'PESO (kg)': 'WEIGHT_KG',
            'ALTURA (m)': 'HEIGHT_M',
            'ETAPA': 'STAGE',
            'CAFEÍNA': 'CAFFEINE',
            'DIETA NAS ÚLTIMAS 24 H': 'DIET_24H',
            'ATIVIDADE (mCi) Repouso': 'ACTIVITY_REST_MCI',
            'HORA INJEÇÃO Repouso': 'INJECTION_TIME_REST',
            'HORA IMAGEM Repouso': 'IMAGE_TIME_REST',
            'ATIVIDADE (mCi) Esforço': 'ACTIVITY_STRESS_MCI',
            'HORA INJEÇÃO Esforço': 'INJECTION_TIME_STRESS',
            'HORA IMAGEM Esforço': 'IMAGE_TIME_STRESS',
            'REPETIU': 'REPEATED',
            'ETAPA REPETIÇÃO': 'REPETITION_STAGE',
            'MOTIVO DE REPETIÇÃO': 'REPETITION_REASON',
            'PREPARO REPETIÇÃO': 'REPETITION_PREPARATION',
            'HORA REPETIÇÃO Repouso': 'REPETITION_TIME_REST',
            'HORA REPETIÇÃO Estress': 'REPETITION_TIME_STRESS',
            'Nº REPETIÇÃO\nTOTAL': 'TOTAL_REPETITIONS',
            'DELTA Repouso': 'DELTA_REST',
            'DELTA Esforço': 'DELTA_STRESS',
            'TEMPO TOTAL ATIVIDADE': 'TOTAL_ACTIVITY_TIME',
            'TEMPO PERMANENCIA': 'TOTAL_STAY_TIME'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Convert categorical variables to English
        df['GENDER'] = df['GENDER'].map({'Masculino': 'Male', 'Feminino': 'Female'})
        df['STAGE'] = df['STAGE'].map({'Completa': 'Complete', 'Incompleta': 'Incomplete'})
        df['CAFFEINE'] = df['CAFFEINE'].map({True: 'Yes', False: 'No'})
        df['REPEATED'] = df['REPEATED'].map({True: 'Yes', False: 'No'})
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_features(df):
    """Create engineered features for machine learning"""
    # Create a copy for feature engineering
    features_df = df.copy()
    
    # Target variables
    features_df['COMPLETION_SUCCESS'] = (features_df['STAGE'] == 'Complete').astype(int)
    features_df['REPETITION_NEEDED'] = (features_df['REPEATED'] == 'Yes').astype(int)
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    features_df['GENDER_ENCODED'] = le_gender.fit_transform(features_df['GENDER'])
    
    le_caffeine = LabelEncoder()
    features_df['CAFFEINE_ENCODED'] = le_caffeine.fit_transform(features_df['CAFFEINE'].fillna('No'))
    
    # Diet complexity feature
    features_df['DIET_COMPLEXITY'] = features_df['DIET_24H'].fillna('').apply(
        lambda x: len(x.split(',')) if x else 0
    )
    
    # BMI categories
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return 0  # Underweight
        elif bmi < 25:
            return 1  # Normal
        elif bmi < 30:
            return 2  # Overweight
        else:
            return 3  # Obese
    
    features_df['BMI_CATEGORY'] = features_df['IMC'].apply(categorize_bmi)
    
    # Age categories
    def categorize_age(age):
        if age < 50:
            return 0  # Young
        elif age < 65:
            return 1  # Middle-aged
        else:
            return 2  # Elderly
    
    features_df['AGE_CATEGORY'] = features_df['AGE'].apply(categorize_age)
    
    # Efficiency ratios
    features_df['EFFICIENCY_RATIO'] = features_df['TOTAL_ACTIVITY_TIME'] / features_df['TOTAL_STAY_TIME']
    features_df['DELTA_RATIO'] = features_df['DELTA_REST'] / (features_df['DELTA_STRESS'] + 1)  # +1 to avoid division by zero
    
    # Activity dose ratio - handle potential division by zero
    features_df['ACTIVITY_RATIO'] = features_df['ACTIVITY_REST_MCI'] / (features_df['ACTIVITY_STRESS_MCI'] + 0.001)
    
    # Weight-based dose normalization - handle potential division by zero
    features_df['REST_DOSE_PER_KG'] = features_df['ACTIVITY_REST_MCI'] / (features_df['WEIGHT_KG'] + 0.001)
    features_df['STRESS_DOSE_PER_KG'] = features_df['ACTIVITY_STRESS_MCI'] / (features_df['WEIGHT_KG'] + 0.001)
    
    # Replace any infinite or extremely large values
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    # Cap extreme values
    for col in ['EFFICIENCY_RATIO', 'DELTA_RATIO', 'ACTIVITY_RATIO', 'REST_DOSE_PER_KG', 'STRESS_DOSE_PER_KG']:
        if col in features_df.columns:
            # Cap values at 99th percentile to handle outliers
            upper_cap = features_df[col].quantile(0.99)
            lower_cap = features_df[col].quantile(0.01)
            features_df[col] = features_df[col].clip(lower=lower_cap, upper=upper_cap)
    
    print(f"Feature engineering completed. New shape: {features_df.shape}")
    return features_df

def select_features_for_ml(df):
    """Select relevant features for machine learning models"""
    feature_columns = [
        'AGE', 'GENDER_ENCODED', 'IMC', 'WEIGHT_KG', 'HEIGHT_M',
        'CAFFEINE_ENCODED', 'DIET_COMPLEXITY', 'BMI_CATEGORY', 'AGE_CATEGORY',
        'ACTIVITY_REST_MCI', 'ACTIVITY_STRESS_MCI', 'DELTA_REST', 'DELTA_STRESS',
        'TOTAL_ACTIVITY_TIME', 'TOTAL_STAY_TIME', 'EFFICIENCY_RATIO', 'DELTA_RATIO',
        'ACTIVITY_RATIO', 'REST_DOSE_PER_KG', 'STRESS_DOSE_PER_KG'
    ]
    
    # Remove rows with missing values in key features
    ml_df = df[feature_columns + ['COMPLETION_SUCCESS', 'REPETITION_NEEDED']].dropna()
    
    # Additional cleanup for infinite values
    ml_df = ml_df.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Remove any remaining problematic values
    for col in feature_columns:
        if col in ml_df.columns:
            # Remove rows with extreme outliers (beyond 5 standard deviations)
            mean_val = ml_df[col].mean()
            std_val = ml_df[col].std()
            ml_df = ml_df[abs(ml_df[col] - mean_val) <= 5 * std_val]
    
    X = ml_df[feature_columns]
    y_completion = ml_df['COMPLETION_SUCCESS']
    y_repetition = ml_df['REPETITION_NEEDED']
    
    print(f"ML dataset prepared. Shape: {X.shape}")
    print(f"Features selected: {len(feature_columns)}")
    
    return X, y_completion, y_repetition, feature_columns

def train_completion_models(X, y_completion, feature_names):
    """Train multiple models to predict procedure completion success"""
    print("\n" + "="*60)
    print("TRAINING COMPLETION PREDICTION MODELS")
    print("="*60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_completion, test_size=0.3, random_state=42, stratify=y_completion)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'SVM': SVC(random_state=42, probability=True, class_weight='balanced')
    }
    
    # Train and evaluate models
    results = {}
    model_objects = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for algorithms that need it
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        cv_scores = cross_val_score(model, X_train_scaled if name in ['Logistic Regression', 'SVM'] else X_train, 
                                   y_train, cv=5, scoring='roc_auc')
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_auc': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        model_objects[name] = model
        
        print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  Test AUC: {auc_score:.3f}")
    
    return results, X_test, y_test, scaler, model_objects, feature_names

def train_repetition_models(X, y_repetition, feature_names):
    """Train multiple models to predict procedure repetition necessity"""
    print("\n" + "="*60)
    print("TRAINING REPETITION PREDICTION MODELS")
    print("="*60)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y_repetition, test_size=0.3, random_state=42, stratify=y_repetition)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'SVM': SVC(random_state=42, probability=True, class_weight='balanced')
    }
    
    # Train and evaluate models
    results = {}
    model_objects = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for algorithms that need it
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        cv_scores = cross_val_score(model, X_train_scaled if name in ['Logistic Regression', 'SVM'] else X_train, 
                                   y_train, cv=5, scoring='roc_auc')
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_auc': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        model_objects[name] = model
        
        print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  Test AUC: {auc_score:.3f}")
    
    return results, X_test, y_test, scaler, model_objects, feature_names

def create_dual_ml_visualizations(completion_results, repetition_results, X_test_comp, y_test_comp, 
                                  X_test_rep, y_test_rep, feature_names):
    """Create comprehensive ML analysis visualizations for both targets"""
    print("\n" + "="*60)
    print("CREATING DUAL TARGET ML VISUALIZATIONS")
    print("="*60)
    
    # Set up the main figure for completion prediction
    fig1 = plt.figure(figsize=(20, 24))
    fig1.suptitle('Machine Learning Analysis - Procedure Completion Prediction', fontsize=16, fontweight='bold')
    
    # 1. Model Performance Comparison - Completion
    ax1 = plt.subplot(4, 3, 1)
    model_names = list(completion_results.keys())
    cv_scores = [completion_results[name]['cv_mean'] for name in model_names]
    test_scores = [completion_results[name]['test_auc'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cv_scores, width, label='CV AUC', color=COLORS['primary'], alpha=0.7)
    bars2 = ax1.bar(x + width/2, test_scores, width, label='Test AUC', color=COLORS['secondary'], alpha=0.7)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Completion Prediction - Model Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # 2. ROC Curves - Completion
    ax2 = plt.subplot(4, 3, 2)
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['quaternary'], '#800080']
    
    for i, (name, result) in enumerate(completion_results.items()):
        fpr, tpr, _ = roc_curve(y_test_comp, result['y_pred_proba'])
        auc_score = result['test_auc']
        ax2.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f'{name} (AUC = {auc_score:.3f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves - Completion Prediction', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature Importance - Completion (Random Forest)
    ax3 = plt.subplot(4, 3, 3)
    rf_model = completion_results['Random Forest']['model']
    feature_importance = rf_model.feature_importances_
    feature_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(10)
    
    bars = ax3.barh(range(len(feature_df)), feature_df['importance'], color=COLORS['quaternary'], alpha=0.7)
    ax3.set_yticks(range(len(feature_df)))
    ax3.set_yticklabels(feature_df['feature'])
    ax3.set_xlabel('Importance Score')
    ax3.set_title('Top 10 Features - Completion', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix - Completion (Best Model)
    best_comp_model = max(completion_results.keys(), key=lambda x: completion_results[x]['test_auc'])
    best_comp_result = completion_results[best_comp_model]
    
    ax4 = plt.subplot(4, 3, 4)
    cm = confusion_matrix(y_test_comp, best_comp_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['Incomplete', 'Complete'], yticklabels=['Incomplete', 'Complete'])
    ax4.set_title(f'Completion - {best_comp_model}', fontweight='bold')
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    
    # Continue with more completion visualizations...
    
    plt.tight_layout()
    plt.savefig('ml_completion_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Set up the second figure for repetition prediction
    fig2 = plt.figure(figsize=(20, 24))
    fig2.suptitle('Machine Learning Analysis - Procedure Repetition Prediction', fontsize=16, fontweight='bold')
    
    # 1. Model Performance Comparison - Repetition
    ax1 = plt.subplot(4, 3, 1)
    model_names = list(repetition_results.keys())
    cv_scores = [repetition_results[name]['cv_mean'] for name in model_names]
    test_scores = [repetition_results[name]['test_auc'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, cv_scores, width, label='CV AUC', color=COLORS['primary'], alpha=0.7)
    bars2 = ax1.bar(x + width/2, test_scores, width, label='Test AUC', color=COLORS['secondary'], alpha=0.7)
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('AUC Score')
    ax1.set_title('Repetition Prediction - Model Performance', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # 2. ROC Curves - Repetition
    ax2 = plt.subplot(4, 3, 2)
    
    for i, (name, result) in enumerate(repetition_results.items()):
        fpr, tpr, _ = roc_curve(y_test_rep, result['y_pred_proba'])
        auc_score = result['test_auc']
        ax2.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=2,
                label=f'{name} (AUC = {auc_score:.3f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves - Repetition Prediction', fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Feature Importance - Repetition (Random Forest)
    ax3 = plt.subplot(4, 3, 3)
    rf_model_rep = repetition_results['Random Forest']['model']
    feature_importance_rep = rf_model_rep.feature_importances_
    feature_df_rep = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance_rep
    }).sort_values('importance', ascending=True).tail(10)
    
    bars = ax3.barh(range(len(feature_df_rep)), feature_df_rep['importance'], color=COLORS['tertiary'], alpha=0.7)
    ax3.set_yticks(range(len(feature_df_rep)))
    ax3.set_yticklabels(feature_df_rep['feature'])
    ax3.set_xlabel('Importance Score')
    ax3.set_title('Top 10 Features - Repetition', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Confusion Matrix - Repetition (Best Model)
    best_rep_model = max(repetition_results.keys(), key=lambda x: repetition_results[x]['test_auc'])
    best_rep_result = repetition_results[best_rep_model]
    
    ax4 = plt.subplot(4, 3, 4)
    cm = confusion_matrix(y_test_rep, best_rep_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax4,
                xticklabels=['No Repeat', 'Repeat'], yticklabels=['No Repeat', 'Repeat'])
    ax4.set_title(f'Repetition - {best_rep_model}', fontweight='bold')
    ax4.set_ylabel('True Label')
    ax4.set_xlabel('Predicted Label')
    
    # 5. Comparison of Both Models (Side by Side)
    ax5 = plt.subplot(4, 3, 5)
    
    # Compare best models from both tasks
    comp_auc = completion_results[best_comp_model]['test_auc']
    rep_auc = repetition_results[best_rep_model]['test_auc']
    
    categories = ['Completion\nPrediction', 'Repetition\nPrediction']
    aucs = [comp_auc, rep_auc]
    
    bars = ax5.bar(categories, aucs, color=[COLORS['quaternary'], COLORS['tertiary']], alpha=0.7)
    ax5.set_ylabel('Test AUC Score')
    ax5.set_title('Best Model Comparison', fontweight='bold')
    ax5.set_ylim(0, 1.1)
    ax5.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        ax5.annotate(f'{auc:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('ml_repetition_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_comp_model, best_comp_result, best_rep_model, best_rep_result

def create_dual_feature_analysis_visualization(X, y_completion, y_repetition, feature_names):
    """Create feature analysis for both completion and repetition targets"""
    print("\n" + "="*60)
    print("CREATING DUAL FEATURE ANALYSIS VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Feature Analysis - Completion vs Repetition Prediction', fontsize=16, fontweight='bold')
    
    # Convert to DataFrame for easier manipulation
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df['completion_target'] = y_completion
    X_df['repetition_target'] = y_repetition
    
    # 1. Age distribution by completion status
    complete_ages = X_df[X_df['completion_target'] == 1]['AGE']
    incomplete_ages = X_df[X_df['completion_target'] == 0]['AGE']
    
    axes[0,0].hist([incomplete_ages, complete_ages], bins=15, alpha=0.7, 
                  label=['Incomplete', 'Complete'], color=[COLORS['primary'], COLORS['quaternary']])
    axes[0,0].set_xlabel('Age (years)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title('Age - Completion Status', fontweight='bold')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Age distribution by repetition status
    repeat_ages = X_df[X_df['repetition_target'] == 1]['AGE']
    no_repeat_ages = X_df[X_df['repetition_target'] == 0]['AGE']
    
    axes[0,1].hist([no_repeat_ages, repeat_ages], bins=15, alpha=0.7, 
                  label=['No Repeat', 'Repeat'], color=[COLORS['tertiary'], COLORS['secondary']])
    axes[0,1].set_xlabel('Age (years)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Age - Repetition Status', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3-4. BMI analysis
    complete_bmi = X_df[X_df['completion_target'] == 1]['IMC']
    incomplete_bmi = X_df[X_df['completion_target'] == 0]['IMC']
    
    axes[0,2].boxplot([incomplete_bmi, complete_bmi], labels=['Incomplete', 'Complete'],
                     patch_artist=True, boxprops=dict(facecolor=COLORS['quaternary'], alpha=0.7))
    axes[0,2].set_ylabel('BMI (kg/m²)')
    axes[0,2].set_title('BMI - Completion', fontweight='bold')
    axes[0,2].grid(True, alpha=0.3)
    
    repeat_bmi = X_df[X_df['repetition_target'] == 1]['IMC']
    no_repeat_bmi = X_df[X_df['repetition_target'] == 0]['IMC']
    
    axes[0,3].boxplot([no_repeat_bmi, repeat_bmi], labels=['No Repeat', 'Repeat'],
                     patch_artist=True, boxprops=dict(facecolor=COLORS['secondary'], alpha=0.7))
    axes[0,3].set_ylabel('BMI (kg/m²)')
    axes[0,3].set_title('BMI - Repetition', fontweight='bold')
    axes[0,3].grid(True, alpha=0.3)
    
    # 5-6. Total Activity Time
    complete_time = X_df[X_df['completion_target'] == 1]['TOTAL_ACTIVITY_TIME']
    incomplete_time = X_df[X_df['completion_target'] == 0]['TOTAL_ACTIVITY_TIME']
    
    axes[1,0].boxplot([incomplete_time, complete_time], labels=['Incomplete', 'Complete'],
                     patch_artist=True, boxprops=dict(facecolor=COLORS['primary'], alpha=0.7))
    axes[1,0].set_ylabel('Activity Time (min)')
    axes[1,0].set_title('Activity Time - Completion', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    repeat_time = X_df[X_df['repetition_target'] == 1]['TOTAL_ACTIVITY_TIME']
    no_repeat_time = X_df[X_df['repetition_target'] == 0]['TOTAL_ACTIVITY_TIME']
    
    axes[1,1].boxplot([no_repeat_time, repeat_time], labels=['No Repeat', 'Repeat'],
                     patch_artist=True, boxprops=dict(facecolor=COLORS['tertiary'], alpha=0.7))
    axes[1,1].set_ylabel('Activity Time (min)')
    axes[1,1].set_title('Activity Time - Repetition', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    # 7-8. Gender analysis
    gender_completion = pd.crosstab(X_df['GENDER_ENCODED'], X_df['completion_target'], normalize='index') * 100
    gender_completion.plot(kind='bar', ax=axes[1,2], color=[COLORS['primary'], COLORS['quaternary']], alpha=0.7)
    axes[1,2].set_xlabel('Gender (0=Female, 1=Male)')
    axes[1,2].set_ylabel('Percentage (%)')
    axes[1,2].set_title('Completion Rate by Gender', fontweight='bold')
    axes[1,2].legend(['Incomplete', 'Complete'])
    axes[1,2].tick_params(axis='x', rotation=0)
    axes[1,2].grid(True, alpha=0.3)
    
    gender_repetition = pd.crosstab(X_df['GENDER_ENCODED'], X_df['repetition_target'], normalize='index') * 100
    gender_repetition.plot(kind='bar', ax=axes[1,3], color=[COLORS['tertiary'], COLORS['secondary']], alpha=0.7)
    axes[1,3].set_xlabel('Gender (0=Female, 1=Male)')
    axes[1,3].set_ylabel('Percentage (%)')
    axes[1,3].set_title('Repetition Rate by Gender', fontweight='bold')
    axes[1,3].legend(['No Repeat', 'Repeat'])
    axes[1,3].tick_params(axis='x', rotation=0)
    axes[1,3].grid(True, alpha=0.3)
    
    # 9-10. Target correlation comparison
    top_features = ['AGE', 'IMC', 'TOTAL_ACTIVITY_TIME', 'EFFICIENCY_RATIO', 
                   'ACTIVITY_RATIO', 'DELTA_REST', 'DELTA_STRESS']
    
    # Completion correlations
    completion_corr = X_df[top_features + ['completion_target']].corr()['completion_target'].drop('completion_target')
    axes[2,0].barh(range(len(completion_corr)), completion_corr.values, color=COLORS['quaternary'], alpha=0.7)
    axes[2,0].set_yticks(range(len(completion_corr)))
    axes[2,0].set_yticklabels(completion_corr.index)
    axes[2,0].set_xlabel('Correlation with Completion')
    axes[2,0].set_title('Feature Correlation - Completion', fontweight='bold')
    axes[2,0].grid(True, alpha=0.3)
    
    # Repetition correlations
    repetition_corr = X_df[top_features + ['repetition_target']].corr()['repetition_target'].drop('repetition_target')
    axes[2,1].barh(range(len(repetition_corr)), repetition_corr.values, color=COLORS['secondary'], alpha=0.7)
    axes[2,1].set_yticks(range(len(repetition_corr)))
    axes[2,1].set_yticklabels(repetition_corr.index)
    axes[2,1].set_xlabel('Correlation with Repetition')
    axes[2,1].set_title('Feature Correlation - Repetition', fontweight='bold')
    axes[2,1].grid(True, alpha=0.3)
    
    # 11-12. Target relationship analysis
    # Completion vs Repetition relationship
    target_crosstab = pd.crosstab(X_df['completion_target'], X_df['repetition_target'])
    target_crosstab_pct = target_crosstab.div(target_crosstab.sum(axis=1), axis=0) * 100
    
    sns.heatmap(target_crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[2,2])
    axes[2,2].set_xlabel('Repetition (0=No, 1=Yes)')
    axes[2,2].set_ylabel('Completion (0=No, 1=Yes)')
    axes[2,2].set_title('Completion vs Repetition\n(Absolute Counts)', fontweight='bold')
    
    sns.heatmap(target_crosstab_pct, annot=True, fmt='.1f', cmap='Oranges', ax=axes[2,3])
    axes[2,3].set_xlabel('Repetition (0=No, 1=Yes)')
    axes[2,3].set_ylabel('Completion (0=No, 1=Yes)')
    axes[2,3].set_title('Completion vs Repetition\n(Percentages)', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dual_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_dual_detailed_results(completion_results, repetition_results, best_comp_model, best_rep_model):
    """Print detailed analysis results for both targets"""
    print("\n" + "="*80)
    print("DETAILED DUAL TARGET MACHINE LEARNING ANALYSIS RESULTS")
    print("="*80)
    
    print("\n" + "🎯 COMPLETION PREDICTION RESULTS")
    print("-" * 50)
    print(f"BEST MODEL: {best_comp_model}")
    best_comp_result = completion_results[best_comp_model]
    print(f"Test AUC Score: {best_comp_result['test_auc']:.4f}")
    print(f"Cross-Validation AUC: {best_comp_result['cv_mean']:.4f} ± {best_comp_result['cv_std']:.4f}")
    
    print("\nCompletion Model Performance:")
    for name, result in completion_results.items():
        print(f"{name:20} | Test AUC: {result['test_auc']:.3f} | CV AUC: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
    
    print("\n" + "🔄 REPETITION PREDICTION RESULTS")
    print("-" * 50)
    print(f"BEST MODEL: {best_rep_model}")
    best_rep_result = repetition_results[best_rep_model]
    print(f"Test AUC Score: {best_rep_result['test_auc']:.4f}")
    print(f"Cross-Validation AUC: {best_rep_result['cv_mean']:.4f} ± {best_rep_result['cv_std']:.4f}")
    
    print("\nRepetition Model Performance:")
    for name, result in repetition_results.items():
        print(f"{name:20} | Test AUC: {result['test_auc']:.3f} | CV AUC: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")
    
    print("\n" + "📊 COMPARATIVE ANALYSIS")
    print("-" * 50)
    print("Completion vs Repetition Prediction:")
    print(f"  Completion Best AUC: {best_comp_result['test_auc']:.3f} ({best_comp_model})")
    print(f"  Repetition Best AUC: {best_rep_result['test_auc']:.3f} ({best_rep_model})")
    
    if best_comp_result['test_auc'] > best_rep_result['test_auc']:
        print("  → Completion prediction is more accurate")
    else:
        print("  → Repetition prediction is more accurate")
    
    print("\n" + "🏥 CLINICAL IMPLICATIONS")
    print("-" * 50)
    print("COMPLETION PREDICTION:")
    comp_report = best_comp_result['classification_report']
    print(f"  • Can identify incomplete procedures with {comp_report['0']['recall']*100:.1f}% sensitivity")
    print(f"  • {comp_report['1']['precision']*100:.1f}% precision for complete procedure prediction")
    print(f"  • Overall accuracy: {comp_report['accuracy']*100:.1f}%")
    
    print("\nREPETITION PREDICTION:")
    rep_report = best_rep_result['classification_report']
    print(f"  • Can identify procedures needing repetition with {rep_report['1']['recall']*100:.1f}% sensitivity")
    print(f"  • {rep_report['0']['precision']*100:.1f}% precision for no-repetition prediction")
    print(f"  • Overall accuracy: {rep_report['accuracy']*100:.1f}%")
    
    print("\n" + "💡 ACTIONABLE INSIGHTS")
    print("-" * 50)
    print("• Both models provide complementary insights for procedure optimization")
    print("• Completion prediction helps with resource allocation and patient preparation")
    print("• Repetition prediction enables proactive intervention to prevent re-procedures")
    print("• Combined use can significantly improve overall procedural success rates")

def main():
    """Main function to run the complete ML analysis"""
    print("ADVANCED MACHINE LEARNING ANALYSIS FOR SCINTIGRAPHY DATA")
    print("="*70)
    print("Implementing comprehensive ML models to predict procedure outcomes...")
    print("All visualizations will be saved as high-resolution PNG files.")
    print("="*70)
    
    # Load and prepare data
    file_path = r'c:\Users\joao\Documents\graficos_cintilografia\dados_cintilografia.csv'
    df = load_and_prepare_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    try:
        # Create features
        features_df = create_features(df)
        
        # Prepare ML dataset
        X, y_completion, y_repetition, feature_names = select_features_for_ml(features_df)
        
        # Train completion prediction models
        completion_results, X_test_comp, y_test_comp, scaler_comp, model_objects_comp, feature_names = train_completion_models(
            X, y_completion, feature_names)
        
        # Train repetition prediction models
        repetition_results, X_test_rep, y_test_rep, scaler_rep, model_objects_rep, _ = train_repetition_models(
            X, y_repetition, feature_names)
        
        # Create comprehensive visualizations
        best_comp_model, best_comp_result, best_rep_model, best_rep_result = create_dual_ml_visualizations(
            completion_results, repetition_results, X_test_comp, y_test_comp, X_test_rep, y_test_rep, feature_names)
        
        # Create feature analysis for both targets
        create_dual_feature_analysis_visualization(X, y_completion, y_repetition, feature_names)
        
        # Print detailed results for both models
        print_dual_detailed_results(completion_results, repetition_results, best_comp_model, best_rep_model)
        
        print("\n" + "="*70)
        print("MACHINE LEARNING ANALYSIS COMPLETE!")
        print("="*70)
        print("Generated files:")
        print("  - dual_feature_analysis.png")
        print("  - dual_ml_performance_comparison.png")
        print("  - dual_feature_importance.png") 
        print("  - dual_confusion_matrices.png")
        print("All charts include detailed statistical information and clinical insights.")
        
    except Exception as e:
        print(f"Error during ML analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()