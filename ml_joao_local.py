# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           accuracy_score, precision_score, recall_score, f1_score, roc_curve)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Define color palette
COLORS = {
    'primary': '#980000',    # Deep red
    'secondary': '#ED7D31',  # Orange  
    'tertiary': '#F1C232',   # Yellow
    'quaternary': '#70AD47'  # Green
}

# Feature name translation mapping
FEATURE_TRANSLATION = {
    'IDADE': 'Age',
    'PESO (kg)': 'Weight (kg)',
    'ALTURA (m)': 'Height (m)',
    'IMC': 'BMI',
    'ATIVIDADE (mCi) Repouso': 'Rest Activity (mCi)',
    'ATIVIDADE (mCi) Esforço': 'Stress Activity (mCi)',
    'Nº REPETIÇÃO\nTOTAL': 'Total Repetitions',
    'SEXO_M': 'Gender_Male',
    'ETAPA_1': 'Stage_1',
    'ETAPA_2': 'Stage_2',
    'CAFEÍNA_Sim': 'Caffeine_Yes',
    'CAFEÍNA_True': 'Caffeine_Yes',
    'CAFEÍNA': 'Caffeine'
}

def translate_feature_names(feature_series_or_list):
    """Translate feature names from Portuguese to English"""
    if hasattr(feature_series_or_list, 'index'):
        # It's a pandas Series
        translated_index = []
        for name in feature_series_or_list.index:
            # Skip ID-related features and timing variables
            excluded_features = ['ID PACIENTE', 'Unnamed: 0', 'Patient ID', 'Index',
                               'DELTA Repouso', 'DELTA Esforço', 'TEMPO TOTAL ATIVIDADE', 
                               'TEMPO PERMANENCIA', 'Rest Delta', 'Stress Delta', 
                               'Total Activity Time', 'Stay Duration']
            if name in excluded_features:
                continue
            # Check for exact match first
            elif name in FEATURE_TRANSLATION:
                translated_index.append(FEATURE_TRANSLATION[name])
            # Check for caffeine variations
            elif 'CAFEÍNA' in name:
                translated_index.append(name.replace('CAFEÍNA', 'Caffeine'))
            else:
                translated_index.append(name)
        # Filter out excluded values too
        filtered_values = []
        original_index = []
        for i, name in enumerate(feature_series_or_list.index):
            excluded_features = ['ID PACIENTE', 'Unnamed: 0', 'Patient ID', 'Index',
                               'DELTA Repouso', 'DELTA Esforço', 'TEMPO TOTAL ATIVIDADE', 
                               'TEMPO PERMANENCIA', 'Rest Delta', 'Stress Delta', 
                               'Total Activity Time', 'Stay Duration']
            if name not in excluded_features:
                filtered_values.append(feature_series_or_list.values[i])
                original_index.append(name)
        
        return pd.Series(filtered_values, index=translated_index)
    else:
        # It's a list
        translated_list = []
        for name in feature_series_or_list:
            excluded_features = ['ID PACIENTE', 'Unnamed: 0', 'Patient ID', 'Index',
                               'DELTA Repouso', 'DELTA Esforço', 'TEMPO TOTAL ATIVIDADE', 
                               'TEMPO PERMANENCIA', 'Rest Delta', 'Stress Delta', 
                               'Total Activity Time', 'Stay Duration']
            if name in excluded_features:
                continue
            elif name in FEATURE_TRANSLATION:
                translated_list.append(FEATURE_TRANSLATION[name])
            elif 'CAFEÍNA' in name:
                translated_list.append(name.replace('CAFEÍNA', 'Caffeine'))
            else:
                translated_list.append(name)
        return translated_list

plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
# Use non-interactive backend to avoid display issues
import matplotlib
matplotlib.use('Agg')

print("="*70)
print("MACHINE LEARNING ANALYSIS - SCINTIGRAPHY PROCEDURE REPETITION")
print("="*70)

# --- Leitura dos dados locais ---
try:
    df = pd.read_csv('dados_cintilografia.csv', encoding='utf-8')
    print(f"✅ Dados carregados com sucesso. Shape: {df.shape}")
    print(f"Colunas disponíveis: {list(df.columns)}")
except Exception as e:
    print(f"❌ Erro ao carregar os dados: {e}")
    exit()

# --- Análise inicial dos dados ---
print(f"\n📊 INFORMAÇÕES BÁSICAS DO DATASET:")
print(f"Número de registros: {len(df)}")
print(f"Número de variáveis: {len(df.columns)}")

# Verificar valores únicos na coluna target
if 'Repetiu' in df.columns:
    target_col = 'Repetiu'
elif 'REPETIU' in df.columns:
    target_col = 'REPETIU'
else:
    print("❌ Coluna de repetição não encontrada!")
    print("Colunas disponíveis:", df.columns.tolist())
    exit()

print(f"\nDistribuição da variável alvo '{target_col}':")
print(df[target_col].value_counts())

# --- Pré-processamento ---
print("\n🔧 INICIANDO PRÉ-PROCESSAMENTO...")

# Criar cópia para trabalhar
df_work = df.copy()

# Mapear coluna alvo para valores numéricos
if df_work[target_col].dtype == 'object':
    # Mapear diferentes formas de representar Sim/Não
    mapping = {'Sim': 1, 'SIM': 1, 'sim': 1, 'S': 1, 's': 1, 'True': 1, True: 1,
               'Não': 0, 'NÃO': 0, 'não': 0, 'N': 0, 'n': 0, 'False': 0, False: 0}
    df_work[target_col] = df_work[target_col].fillna('Não').map(mapping)
    
    # Para valores que não foram mapeados, assumir como "Não"
    df_work[target_col] = df_work[target_col].fillna(0)

print(f"Distribuição após mapeamento:")
print(df_work[target_col].value_counts())

# Identificar colunas numéricas e categóricas
numeric_cols = []
categorical_cols = []

for col in df_work.columns:
    if col == target_col:
        continue
    
    if df_work[col].dtype in ['int64', 'float64']:
        numeric_cols.append(col)
    else:
        # Tentar converter para numérico
        try:
            # Tratar vírgulas decimais
            if df_work[col].dtype == 'object':
                df_work[col] = pd.to_numeric(df_work[col].astype(str).str.replace(',', '.'), errors='coerce')
                if not df_work[col].isna().all():
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                categorical_cols.append(col)
        except:
            categorical_cols.append(col)

print(f"\nColunas numéricas identificadas ({len(numeric_cols)}): {numeric_cols}")
print(f"Colunas categóricas identificadas ({len(categorical_cols)}): {categorical_cols}")

# Processar colunas categóricas
if categorical_cols:
    # Preencher valores ausentes
    for col in categorical_cols:
        df_work[col] = df_work[col].fillna('NÃO INFORMADO')
    
    # Aplicar one-hot encoding sem prefix para evitar problemas
    df_categorical = pd.get_dummies(df_work[categorical_cols], drop_first=True)
    print(f"Variáveis categóricas após encoding: {df_categorical.shape[1]}")
else:
    df_categorical = pd.DataFrame()

# Processar colunas numéricas
if numeric_cols:
    # Remove patient ID, index columns and specific timing variables as requested
    excluded_cols = ['ID PACIENTE', 'Unnamed: 0', 'DELTA Repouso', 'DELTA Esforço', 
                    'TEMPO TOTAL ATIVIDADE', 'TEMPO PERMANENCIA']
    clinical_numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
    df_numeric = df_work[clinical_numeric_cols].fillna(df_work[clinical_numeric_cols].median())
    print(f"Variáveis numéricas processadas: {df_numeric.shape[1]} (excluindo IDs e variáveis de tempo)")
else:
    df_numeric = pd.DataFrame()

# Combinar features
if not df_categorical.empty and not df_numeric.empty:
    X = pd.concat([df_numeric, df_categorical], axis=1)
elif not df_numeric.empty:
    X = df_numeric
elif not df_categorical.empty:
    X = df_categorical
else:
    print("❌ Nenhuma feature válida encontrada!")
    exit()

y = df_work[target_col].astype(int)

print(f"\n📋 DATASET FINAL:")
print(f"Features: {X.shape[1]} variáveis")
print(f"Samples: {X.shape[0]} registros")
print(f"Target distribution:")
print(y.value_counts())

# Verificar se há variação suficiente na target
if len(y.unique()) < 2:
    print("❌ Não há variação suficiente na variável target para criar um modelo!")
    exit()

# --- Divisão treino/teste ---
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    
    print(f"\n✂️ DIVISÃO DOS DADOS:")
    print(f"Treino: {len(X_train)} amostras")
    print(f"Teste: {len(X_test)} amostras")
    print(f"Distribuição no treino: {y_train.value_counts().to_dict()}")
    print(f"Distribuição no teste: {y_test.value_counts().to_dict()}")

except ValueError as e:
    print(f"❌ Erro na divisão dos dados: {e}")
    print("Tentando divisão sem estratificação...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

# --- Balanceamento com SMOTE ---
print(f"\n⚖️ APLICANDO BALANCEAMENTO SMOTE...")

try:
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    print(f"Treino balanceado: {len(X_train_bal)} amostras")
    print(f"Distribuição balanceada: {pd.Series(y_train_bal).value_counts().to_dict()}")
    
except Exception as e:
    print(f"⚠️ Erro no SMOTE: {e}")
    print("Utilizando dados originais sem balanceamento...")
    X_train_bal, y_train_bal = X_train, y_train

# --- Treinamento do Modelo ---
print(f"\n🤖 TREINAMENTO DO MODELO RANDOM FOREST...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42, 
    class_weight='balanced',
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2
)

# Validação cruzada
try:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=skf, scoring='f1')
    print(f"F1-score médio na validação cruzada: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
except:
    print("Validação cruzada não executada devido ao tamanho pequeno dos dados")

# Treinar modelo
model.fit(X_train_bal, y_train_bal)
print("✅ Modelo treinado com sucesso!")

# --- Avaliação ---
print(f"\n📊 AVALIAÇÃO DO MODELO:")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Métricas básicas
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)

print(f"Acurácia: {acc:.3f}")
print(f"Precisão: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"AUC-ROC: {auc:.3f}")

print(f"\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

print(f"\nMatriz de Confusão:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# --- Visualizações ---
print(f"\n📈 CRIANDO VISUALIZAÇÕES...")

# 1. Importância das Features
plt.figure(figsize=(12, 8))
if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_15 = importances.nlargest(15)
    
    # Translate feature names
    top_15_translated = translate_feature_names(top_15)
    
    ax = top_15_translated.plot(kind='barh', color=COLORS['secondary'], alpha=0.8)
    plt.title('Top 15 Most Important Features for Procedure Repetition Prediction', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance_repetition_en.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show to avoid display issues

# 2. Curva ROC
plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, linewidth=2, label=f'Random Forest (AUC = {auc:.3f})', 
         color=COLORS['primary'])
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.7, label='Baseline')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Procedure Repetition Prediction', 
          fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve_repetition_en.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Matriz de Confusão Visualizada
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.7)
plt.title('Confusion Matrix - Procedure Repetition Prediction', fontsize=14, fontweight='bold', pad=20)
plt.colorbar(label='Number of Cases')

# Adicionar texto nas células
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=14, fontweight='bold')

plt.ylabel('True Value', fontsize=12)
plt.xlabel('Prediction', fontsize=12)
plt.xticks([0, 1], ['No Repetition', 'Repetition'])
plt.yticks([0, 1], ['No Repetition', 'Repetition'])
plt.tight_layout()
plt.savefig('confusion_matrix_repetition_en.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Distribuição de Probabilidades
plt.figure(figsize=(12, 6))
prob_no_repeat = y_proba[y_test == 0]
prob_repeat = y_proba[y_test == 1]

plt.hist(prob_no_repeat, bins=20, alpha=0.7, label='No Repetition', 
         color=COLORS['quaternary'], density=True)
plt.hist(prob_repeat, bins=20, alpha=0.7, label='Repetition', 
         color=COLORS['primary'], density=True)

plt.xlabel('Predicted Probability of Repetition', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Predicted Probabilities', fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('probability_distribution_repetition_en.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Análise de limiar otimizado
print(f"\n🎯 ANÁLISE DE LIMIAR OTIMIZADO:")

# Testar diferentes limiares
thresholds_test = np.arange(0.1, 1.0, 0.05)
f1_scores = []
precisions = []
recalls = []

for thresh in thresholds_test:
    y_pred_thresh = (y_proba >= thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh, zero_division=0))
    precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_thresh, zero_division=0))

# Encontrar o melhor limiar
best_thresh_idx = np.argmax(f1_scores)
best_thresh = thresholds_test[best_thresh_idx]
best_f1 = f1_scores[best_thresh_idx]

print(f"Melhor limiar encontrado: {best_thresh:.2f}")
print(f"F1-score com melhor limiar: {best_f1:.3f}")

# Avaliação com limiar otimizado
y_pred_optimized = (y_proba >= best_thresh).astype(int)

acc_opt = accuracy_score(y_test, y_pred_optimized)
prec_opt = precision_score(y_test, y_pred_optimized, zero_division=0)
rec_opt = recall_score(y_test, y_pred_optimized, zero_division=0)
f1_opt = f1_score(y_test, y_pred_optimized, zero_division=0)

print(f"\nMétricas com limiar otimizado ({best_thresh:.2f}):")
print(f"Acurácia: {acc_opt:.3f}")
print(f"Precisão: {prec_opt:.3f}")
print(f"Recall: {rec_opt:.3f}")
print(f"F1-score: {f1_opt:.3f}")

# 6. Gráfico de análise de limiar
plt.figure(figsize=(12, 6))
plt.plot(thresholds_test, f1_scores, 'o-', label='F1-Score', color=COLORS['primary'], linewidth=2)
plt.plot(thresholds_test, precisions, 'o-', label='Precision', color=COLORS['secondary'], linewidth=2)
plt.plot(thresholds_test, recalls, 'o-', label='Recall', color=COLORS['tertiary'], linewidth=2)
plt.axvline(x=best_thresh, color=COLORS['quaternary'], linestyle='--', 
            label=f'Best Threshold ({best_thresh:.2f})')

plt.xlabel('Classification Threshold', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Classification Threshold Analysis', fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('threshold_analysis_repetition_en.png', dpi=300, bbox_inches='tight')
plt.close()

# --- Conclusões ---
print(f"\n" + "="*70)
print("RESUMO DA ANÁLISE")
print("="*70)

print(f"📊 DATASET:")
print(f"  • Total de registros: {len(df)}")
print(f"  • Features utilizadas: {X.shape[1]}")
print(f"  • Taxa de repetição: {y.mean():.1%}")

print(f"\n🤖 PERFORMANCE DO MODELO:")
print(f"  • Acurácia: {acc:.1%}")
print(f"  • Precisão: {prec:.1%}")
print(f"  • Recall: {rec:.1%}")
print(f"  • F1-Score: {f1:.3f}")
print(f"  • AUC-ROC: {auc:.3f}")

print(f"\n🎯 LIMIAR OTIMIZADO:")
print(f"  • Melhor limiar: {best_thresh:.2f}")
print(f"  • F1-Score otimizado: {f1_opt:.3f}")
print(f"  • Recall otimizado: {rec_opt:.1%}")

print(f"\n📁 GENERATED FILES:")
print(f"  • feature_importance_repetition_en.png")
print(f"  • roc_curve_repetition_en.png")
print(f"  • confusion_matrix_repetition_en.png")
print(f"  • probability_distribution_repetition_en.png")
print(f"  • threshold_analysis_repetition_en.png")

print(f"\n✅ ANÁLISE CONCLUÍDA COM SUCESSO!")

print(f"\n🔍 TOP 5 MOST IMPORTANT FEATURES (English):")
if hasattr(model, 'feature_importances_'):
    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_5 = importances.nlargest(5)
    top_5_translated = translate_feature_names(top_5)
    for i, (feature, importance) in enumerate(top_5_translated.items(), 1):
        print(f"  {i}. {feature}: {importance:.3f}")
        
print(f"\n📊 ENGLISH FEATURE NAMES MAPPING:")
excluded_features = ['ID PACIENTE', 'Unnamed: 0', 'DELTA Repouso', 'DELTA Esforço', 
                    'TEMPO TOTAL ATIVIDADE', 'TEMPO PERMANENCIA']
used_features = [col for col in X.columns.tolist()[:10] if col not in excluded_features]  # Show first 10 clinical features
print("  Original → English:")
for feature in used_features:
    translated = FEATURE_TRANSLATION.get(feature, feature)
    if translated != feature:
        print(f"  • {feature} → {translated}")
    elif 'CAFEÍNA' in feature:
        print(f"  • {feature} → {feature.replace('CAFEÍNA', 'Caffeine')}")
    else:
        print(f"  • {feature} (unchanged)")