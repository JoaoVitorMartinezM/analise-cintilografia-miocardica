#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scintigraphy Data Analysis and Visualization
Author: AI Assistant
Date: November 2025

This script analyzes scintigraphy procedure data and generates comprehensive visualizations
with insights for medical practice improvement.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set matplotlib parameters for better visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_and_clean_data(file_path):
    """Load and clean the scintigraphy data"""
    try:
        # Load the data
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        
        # Display basic info about the dataset
        print("\nDataset Info:")
        print(f"Number of patients: {len(df)}")
        print(f"Number of variables: {len(df.columns)}")
        
        # Basic data exploration
        print("\nData types:")
        print(df.dtypes.value_counts())
        
        print("\nMissing values:")
        missing_data = df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        print(missing_data)
        
        # Clean column names by translating to English
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
        
        # Convert gender to English
        df['GENDER'] = df['GENDER'].map({'Masculino': 'Male', 'Feminino': 'Female'})
        
        # Convert stage to English
        df['STAGE'] = df['STAGE'].map({'Completa': 'Complete', 'Incompleta': 'Incomplete'})
        
        # Convert boolean columns
        df['CAFFEINE'] = df['CAFFEINE'].map({True: 'Yes', False: 'No'})
        df['REPEATED'] = df['REPEATED'].map({True: 'Yes', False: 'No'})
        
        print(f"\nData cleaning completed. Final shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_demographic_analysis(df):
    """Create demographic analysis visualizations"""
    print("\n" + "="*50)
    print("CREATING DEMOGRAPHIC ANALYSIS")
    print("="*50)
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Demographic Analysis - Scintigraphy Patients', fontsize=16, fontweight='bold')
    
    # 1. Gender Distribution
    gender_counts = df['GENDER'].value_counts()
    colors = ['#980000', '#ED7D31']
    wedges, texts, autotexts = axes[0,0].pie(gender_counts.values, 
                                            labels=gender_counts.index, 
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90)
    axes[0,0].set_title('Gender Distribution\n(n={})'.format(len(df)), fontweight='bold')
    
    # 2. Age Distribution
    axes[0,1].hist(df['AGE'], bins=15, color='#F1C232', alpha=0.7, edgecolor='black')
    axes[0,1].axvline(df['AGE'].mean(), color='#980000', linestyle='--', 
                     label=f'Mean: {df["AGE"].mean():.1f} years')
    axes[0,1].axvline(df['AGE'].median(), color='#ED7D31', linestyle='--', 
                     label=f'Median: {df["AGE"].median():.1f} years')
    axes[0,1].set_xlabel('Age (years)')
    axes[0,1].set_ylabel('Number of Patients')
    axes[0,1].set_title('Age Distribution', fontweight='bold')
    axes[0,1].legend()
    
    # 3. BMI Distribution by Gender
    male_bmi = df[df['GENDER'] == 'Male']['IMC']
    female_bmi = df[df['GENDER'] == 'Female']['IMC']
    
    axes[1,0].hist([male_bmi, female_bmi], bins=15, alpha=0.7, 
                  label=['Male', 'Female'], color=['#980000', '#ED7D31'])
    axes[1,0].set_xlabel('BMI (kg/m²)')
    axes[1,0].set_ylabel('Number of Patients')
    axes[1,0].set_title('BMI Distribution by Gender', fontweight='bold')
    axes[1,0].legend()
    
    # Add BMI category lines
    axes[1,0].axvline(18.5, color='green', linestyle=':', alpha=0.7, label='Underweight')
    axes[1,0].axvline(25, color='orange', linestyle=':', alpha=0.7, label='Overweight')
    axes[1,0].axvline(30, color='red', linestyle=':', alpha=0.7, label='Obese')
    
    # 4. Age vs BMI Scatter Plot
    scatter = axes[1,1].scatter(df['AGE'], df['IMC'], 
                               c=df['GENDER'].map({'Male': '#980000', 'Female': '#ED7D31'}),
                               alpha=0.6, s=50)
    axes[1,1].set_xlabel('Age (years)')
    axes[1,1].set_ylabel('BMI (kg/m²)')
    axes[1,1].set_title('Age vs BMI Distribution', fontweight='bold')
    
    # Add trend line
    z = np.polyfit(df['AGE'], df['IMC'], 1)
    p = np.poly1d(z)
    axes[1,1].plot(df['AGE'], p(df['AGE']), color='#70AD47', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Trend: BMI = {z[0]:.3f}*Age + {z[1]:.1f}')
    axes[1,1].legend()
    
    # Add custom legend for gender colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#980000', label='Male'),
                      Patch(facecolor='#ED7D31', label='Female')]
    axes[1,1].legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"Gender distribution:")
    print(f"  - Male: {len(df[df['GENDER'] == 'Male'])} ({len(df[df['GENDER'] == 'Male'])/len(df)*100:.1f}%)")
    print(f"  - Female: {len(df[df['GENDER'] == 'Female'])} ({len(df[df['GENDER'] == 'Female'])/len(df)*100:.1f}%)")
    print(f"\nAge statistics:")
    print(f"  - Mean: {df['AGE'].mean():.1f} years")
    print(f"  - Median: {df['AGE'].median():.1f} years")
    print(f"  - Range: {df['AGE'].min()}-{df['AGE'].max()} years")
    print(f"\nBMI statistics:")
    print(f"  - Mean: {df['IMC'].mean():.1f} kg/m²")
    print(f"  - Median: {df['IMC'].median():.1f} kg/m²")
    print(f"  - Range: {df['IMC'].min():.1f}-{df['IMC'].max():.1f} kg/m²")

def create_procedure_analysis(df):
    """Create procedure analysis visualizations"""
    print("\n" + "="*50)
    print("CREATING PROCEDURE ANALYSIS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Procedure Analysis - Completion and Repetition Patterns', fontsize=16, fontweight='bold')
    
    # 1. Completion Status
    stage_counts = df['STAGE'].value_counts()
    colors = ['#70AD47', '#980000']
    wedges, texts, autotexts = axes[0,0].pie(stage_counts.values, 
                                            labels=stage_counts.index, 
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90)
    axes[0,0].set_title('Procedure Completion Status\n(n={})'.format(len(df)), fontweight='bold')
    
    # 2. Repetition Rates
    repetition_counts = df['REPEATED'].value_counts()
    colors = ['#F1C232', '#ED7D31']
    wedges, texts, autotexts = axes[0,1].pie(repetition_counts.values, 
                                            labels=repetition_counts.index, 
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90)
    axes[0,1].set_title('Procedure Repetition Rates\n(n={})'.format(len(df)), fontweight='bold')
    
    # 3. Repetition Reasons
    repeat_reasons = df[df['REPEATED'] == 'Yes']['REPETITION_REASON'].value_counts()
    if len(repeat_reasons) > 0:
        repeat_reasons.plot(kind='bar', ax=axes[1,0], color='#ED7D31', alpha=0.7)
        axes[1,0].set_title('Reasons for Procedure Repetition', fontweight='bold')
        axes[1,0].set_xlabel('Repetition Reason')
        axes[1,0].set_ylabel('Number of Cases')
        axes[1,0].tick_params(axis='x', rotation=45)
    else:
        axes[1,0].text(0.5, 0.5, 'No repetition reasons data', 
                      ha='center', va='center', transform=axes[1,0].transAxes)
        axes[1,0].set_title('Reasons for Procedure Repetition', fontweight='bold')
    
    # 4. Completion vs Repetition Analysis
    completion_repetition = pd.crosstab(df['STAGE'], df['REPEATED'])
    completion_repetition.plot(kind='bar', ax=axes[1,1], color=['#F1C232', '#980000'])
    axes[1,1].set_title('Completion Status vs Repetition Rate', fontweight='bold')
    axes[1,1].set_xlabel('Completion Status')
    axes[1,1].set_ylabel('Number of Patients')
    axes[1,1].legend(title='Repeated')
    axes[1,1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('procedure_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    completion_rate = len(df[df['STAGE'] == 'Complete']) / len(df) * 100
    repetition_rate = len(df[df['REPEATED'] == 'Yes']) / len(df) * 100
    
    print(f"Procedure completion rate: {completion_rate:.1f}%")
    print(f"Procedure repetition rate: {repetition_rate:.1f}%")
    if len(repeat_reasons) > 0:
        print(f"Most common repetition reason: {repeat_reasons.index[0]} ({repeat_reasons.iloc[0]} cases)")

def create_timing_analysis(df):
    """Create timing and efficiency analysis"""
    print("\n" + "="*50)
    print("CREATING TIMING ANALYSIS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Timing and Efficiency Analysis', fontsize=16, fontweight='bold')
    
    # 1. Total Activity Time Distribution
    axes[0,0].hist(df['TOTAL_ACTIVITY_TIME'], bins=20, color='#70AD47', alpha=0.7, edgecolor='black')
    axes[0,0].axvline(df['TOTAL_ACTIVITY_TIME'].mean(), color='#980000', linestyle='--', 
                     label=f'Mean: {df["TOTAL_ACTIVITY_TIME"].mean():.1f} min')
    axes[0,0].axvline(df['TOTAL_ACTIVITY_TIME'].median(), color='#ED7D31', linestyle='--', 
                     label=f'Median: {df["TOTAL_ACTIVITY_TIME"].median():.1f} min')
    axes[0,0].set_xlabel('Total Activity Time (minutes)')
    axes[0,0].set_ylabel('Number of Patients')
    axes[0,0].set_title('Total Activity Time Distribution', fontweight='bold')
    axes[0,0].legend()
    
    # 2. Total Stay Time Distribution
    axes[0,1].hist(df['TOTAL_STAY_TIME'], bins=20, color='#F1C232', alpha=0.7, edgecolor='black')
    axes[0,1].axvline(df['TOTAL_STAY_TIME'].mean(), color='#980000', linestyle='--', 
                     label=f'Mean: {df["TOTAL_STAY_TIME"].mean():.1f} min')
    axes[0,1].axvline(df['TOTAL_STAY_TIME'].median(), color='#ED7D31', linestyle='--', 
                     label=f'Median: {df["TOTAL_STAY_TIME"].median():.1f} min')
    axes[0,1].set_xlabel('Total Stay Time (minutes)')
    axes[0,1].set_ylabel('Number of Patients')
    axes[0,1].set_title('Total Stay Time Distribution', fontweight='bold')
    axes[0,1].legend()
    
    # 3. Delta Times (Rest vs Stress)
    axes[1,0].scatter(df['DELTA_REST'], df['DELTA_STRESS'], alpha=0.6, color='#ED7D31')
    axes[1,0].set_xlabel('Delta Rest Time (minutes)')
    axes[1,0].set_ylabel('Delta Stress Time (minutes)')
    axes[1,0].set_title('Rest vs Stress Delta Times', fontweight='bold')
    
    # Add correlation line
    correlation = np.corrcoef(df['DELTA_REST'], df['DELTA_STRESS'])[0,1]
    z = np.polyfit(df['DELTA_REST'], df['DELTA_STRESS'], 1)
    p = np.poly1d(z)
    axes[1,0].plot(df['DELTA_REST'], p(df['DELTA_REST']), color='#70AD47', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Correlation: {correlation:.3f}')
    axes[1,0].legend()
    
    # 4. Efficiency Analysis: Activity Time vs Stay Time
    scatter = axes[1,1].scatter(df['TOTAL_ACTIVITY_TIME'], df['TOTAL_STAY_TIME'], 
                               c=df['STAGE'].map({'Complete': '#70AD47', 'Incomplete': '#980000'}),
                               alpha=0.6, s=50)
    axes[1,1].set_xlabel('Total Activity Time (minutes)')
    axes[1,1].set_ylabel('Total Stay Time (minutes)')
    axes[1,1].set_title('Activity Time vs Stay Time (Efficiency)', fontweight='bold')
    
    # Add efficiency line (ideal would be y=x)
    min_time = min(df['TOTAL_ACTIVITY_TIME'].min(), df['TOTAL_STAY_TIME'].min())
    max_time = max(df['TOTAL_ACTIVITY_TIME'].max(), df['TOTAL_STAY_TIME'].max())
    axes[1,1].plot([min_time, max_time], [min_time, max_time], 'k--', alpha=0.5, 
                   label='Perfect Efficiency (Activity = Stay)')
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#70AD47', label='Complete'),
                      Patch(facecolor='#980000', label='Incomplete')]
    axes[1,1].legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig('timing_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"Timing statistics:")
    print(f"  - Mean total activity time: {df['TOTAL_ACTIVITY_TIME'].mean():.1f} minutes")
    print(f"  - Mean total stay time: {df['TOTAL_STAY_TIME'].mean():.1f} minutes")
    print(f"  - Average efficiency: {(df['TOTAL_ACTIVITY_TIME']/df['TOTAL_STAY_TIME']).mean()*100:.1f}%")
    print(f"  - Rest-Stress delta correlation: {correlation:.3f}")

def create_diet_analysis(df):
    """Create diet and preparation analysis"""
    print("\n" + "="*50)
    print("CREATING DIET ANALYSIS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Diet and Preparation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Caffeine Consumption
    caffeine_counts = df['CAFFEINE'].value_counts()
    colors = ['#F1C232', '#ED7D31']
    wedges, texts, autotexts = axes[0,0].pie(caffeine_counts.values, 
                                            labels=caffeine_counts.index, 
                                            autopct='%1.1f%%',
                                            colors=colors,
                                            startangle=90)
    axes[0,0].set_title('Caffeine Consumption\n(n={})'.format(len(df)), fontweight='bold')
    
    # 2. Most Common Diet Items
    # Parse diet data
    all_foods = []
    for diet in df['DIET_24H'].dropna():
        if diet and isinstance(diet, str):
            foods = [food.strip() for food in diet.split(',')]
            all_foods.extend(foods)
    
    if all_foods:
        food_counts = pd.Series(all_foods).value_counts().head(10)
        food_counts.plot(kind='barh', ax=axes[0,1], color='#70AD47', alpha=0.7)
        axes[0,1].set_title('Top 10 Most Consumed Foods (24h before)', fontweight='bold')
        axes[0,1].set_xlabel('Number of Patients')
    else:
        axes[0,1].text(0.5, 0.5, 'No diet data available', 
                      ha='center', va='center', transform=axes[0,1].transAxes)
        axes[0,1].set_title('Diet Analysis', fontweight='bold')
    
    # 3. Caffeine vs Completion Rate
    caffeine_completion = pd.crosstab(df['CAFFEINE'], df['STAGE'])
    caffeine_completion_pct = caffeine_completion.div(caffeine_completion.sum(axis=1), axis=0) * 100
    caffeine_completion_pct.plot(kind='bar', ax=axes[1,0], color=['#980000', '#70AD47'])
    axes[1,0].set_title('Completion Rate by Caffeine Consumption', fontweight='bold')
    axes[1,0].set_xlabel('Caffeine Consumption')
    axes[1,0].set_ylabel('Percentage (%)')
    axes[1,0].legend(title='Completion Status')
    axes[1,0].tick_params(axis='x', rotation=0)
    
    # 4. Diet Complexity vs Repetition
    df['DIET_ITEMS_COUNT'] = df['DIET_24H'].fillna('').apply(lambda x: len(x.split(',')) if x else 0)
    
    # Group by diet complexity
    diet_groups = df.groupby('DIET_ITEMS_COUNT')['REPEATED'].apply(lambda x: (x == 'Yes').mean() * 100)
    valid_groups = diet_groups[diet_groups.index <= 15]  # Limit to reasonable range
    
    if len(valid_groups) > 1:
        valid_groups.plot(kind='line', ax=axes[1,1], marker='o', color='#ED7D31', linewidth=2)
        axes[1,1].set_title('Repetition Rate by Diet Complexity', fontweight='bold')
        axes[1,1].set_xlabel('Number of Food Items (24h before)')
        axes[1,1].set_ylabel('Repetition Rate (%)')
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Insufficient data for diet complexity analysis', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Diet Complexity Analysis', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('diet_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    caffeine_yes_rate = len(df[df['CAFFEINE'] == 'Yes']) / len(df) * 100
    print(f"Caffeine consumption rate: {caffeine_yes_rate:.1f}%")
    if all_foods:
        print(f"Most common food consumed: {food_counts.index[0]} ({food_counts.iloc[0]} patients)")
    print(f"Average diet complexity: {df['DIET_ITEMS_COUNT'].mean():.1f} items")

def create_activity_analysis(df):
    """Create radioactive activity dose analysis"""
    print("\n" + "="*50)
    print("CREATING ACTIVITY ANALYSIS")
    print("="*50)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Radioactive Activity Dose Analysis', fontsize=16, fontweight='bold')
    
    # 1. Rest Activity Distribution
    axes[0,0].hist(df['ACTIVITY_REST_MCI'], bins=20, color='#70AD47', alpha=0.7, edgecolor='black')
    axes[0,0].axvline(df['ACTIVITY_REST_MCI'].mean(), color='#980000', linestyle='--', 
                     label=f'Mean: {df["ACTIVITY_REST_MCI"].mean():.1f} mCi')
    axes[0,0].axvline(df['ACTIVITY_REST_MCI'].median(), color='#ED7D31', linestyle='--', 
                     label=f'Median: {df["ACTIVITY_REST_MCI"].median():.1f} mCi')
    axes[0,0].set_xlabel('Rest Activity (mCi)')
    axes[0,0].set_ylabel('Number of Patients')
    axes[0,0].set_title('Rest Phase Activity Distribution', fontweight='bold')
    axes[0,0].legend()
    
    # 2. Stress Activity Distribution
    axes[0,1].hist(df['ACTIVITY_STRESS_MCI'], bins=20, color='#F1C232', alpha=0.7, edgecolor='black')
    axes[0,1].axvline(df['ACTIVITY_STRESS_MCI'].mean(), color='#980000', linestyle='--', 
                     label=f'Mean: {df["ACTIVITY_STRESS_MCI"].mean():.1f} mCi')
    axes[0,1].axvline(df['ACTIVITY_STRESS_MCI'].median(), color='#ED7D31', linestyle='--', 
                     label=f'Median: {df["ACTIVITY_STRESS_MCI"].median():.1f} mCi')
    axes[0,1].set_xlabel('Stress Activity (mCi)')
    axes[0,1].set_ylabel('Number of Patients')
    axes[0,1].set_title('Stress Phase Activity Distribution', fontweight='bold')
    axes[0,1].legend()
    
    # 3. Rest vs Stress Activity Correlation
    scatter = axes[1,0].scatter(df['ACTIVITY_REST_MCI'], df['ACTIVITY_STRESS_MCI'], 
                               c=df['STAGE'].map({'Complete': '#70AD47', 'Incomplete': '#980000'}),
                               alpha=0.6, s=50)
    axes[1,0].set_xlabel('Rest Activity (mCi)')
    axes[1,0].set_ylabel('Stress Activity (mCi)')
    axes[1,0].set_title('Rest vs Stress Activity Correlation', fontweight='bold')
    
    # Add correlation line
    correlation = np.corrcoef(df['ACTIVITY_REST_MCI'], df['ACTIVITY_STRESS_MCI'])[0,1]
    z = np.polyfit(df['ACTIVITY_REST_MCI'], df['ACTIVITY_STRESS_MCI'], 1)
    p = np.poly1d(z)
    axes[1,0].plot(df['ACTIVITY_REST_MCI'], p(df['ACTIVITY_REST_MCI']), color='#F1C232', linestyle='--', alpha=0.8, linewidth=2,
                   label=f'Correlation: {correlation:.3f}')
    axes[1,0].legend()
    
    # Custom legend for stages
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#70AD47', label='Complete'),
                      Patch(facecolor='#980000', label='Incomplete')]
    axes[1,0].legend(handles=legend_elements, loc='upper right')
    
    # 4. Activity vs BMI Analysis
    axes[1,1].scatter(df['IMC'], df['ACTIVITY_REST_MCI'], alpha=0.6, color='#70AD47', label='Rest')
    axes[1,1].scatter(df['IMC'], df['ACTIVITY_STRESS_MCI'], alpha=0.6, color='#980000', label='Stress')
    axes[1,1].set_xlabel('BMI (kg/m²)')
    axes[1,1].set_ylabel('Activity (mCi)')
    axes[1,1].set_title('Activity Dose vs BMI', fontweight='bold')
    axes[1,1].legend()
    
    # Add trend lines
    z_rest = np.polyfit(df['IMC'], df['ACTIVITY_REST_MCI'], 1)
    z_stress = np.polyfit(df['IMC'], df['ACTIVITY_STRESS_MCI'], 1)
    p_rest = np.poly1d(z_rest)
    p_stress = np.poly1d(z_stress)
    axes[1,1].plot(df['IMC'], p_rest(df['IMC']), color='#70AD47', linestyle='--', alpha=0.8, linewidth=2)
    axes[1,1].plot(df['IMC'], p_stress(df['IMC']), color='#980000', linestyle='--', alpha=0.8, linewidth=2)
    
    plt.tight_layout()
    plt.savefig('activity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"Activity dose statistics:")
    print(f"  - Rest phase mean: {df['ACTIVITY_REST_MCI'].mean():.1f} mCi")
    print(f"  - Stress phase mean: {df['ACTIVITY_STRESS_MCI'].mean():.1f} mCi")
    print(f"  - Rest-Stress correlation: {correlation:.3f}")
    print(f"  - Rest-BMI correlation: {np.corrcoef(df['IMC'], df['ACTIVITY_REST_MCI'])[0,1]:.3f}")
    print(f"  - Stress-BMI correlation: {np.corrcoef(df['IMC'], df['ACTIVITY_STRESS_MCI'])[0,1]:.3f}")

def main():
    """Main function to run the complete analysis"""
    print("SCINTIGRAPHY DATA ANALYSIS")
    print("="*60)
    print("Loading and analyzing cardiac scintigraphy procedure data...")
    print("All visualizations will be saved as high-resolution PNG files.")
    print("="*60)
    
    # Load and clean data
    file_path = r'c:\Users\joao\Documents\graficos_cintilografia\dados_cintilografia.csv'
    df = load_and_clean_data(file_path)
    
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    try:
        # Create all visualizations
        create_demographic_analysis(df)
        create_procedure_analysis(df)
        create_timing_analysis(df)
        create_diet_analysis(df)
        create_activity_analysis(df)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  - demographic_analysis.png")
        print("  - procedure_analysis.png") 
        print("  - timing_analysis.png")
        print("  - diet_analysis.png")
        print("  - activity_analysis.png")
        print("\nAll charts have been saved with high resolution (300 DPI)")
        print("and include comprehensive statistical information.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()