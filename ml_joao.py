# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  # pip install imblearn

# --- Leitura dos dados ---
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRooSfamgfpn6HM3tXDhRzzeYvFjdubZ0gWBGDiklSY1fstXGe9iUIAQ-btWhfL8SRRe8nBJ1sU-0x-/pub?output=csv"

try:
    df = pd.read_csv(url, encoding='utf-8', decimal=',', thousands='.', on_bad_lines='skip')
    print("Dados carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar os dados: {e}")

# --- Pré-processamento ---

# Ajuste numérico
for col in ['IMC', 'ALTURA (m)']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# Mapear coluna alvo
df['REPETIU'] = df['REPETIU'].fillna('Não').map({'Sim': 1, 'Não': 0})

# Preencher NA nas categóricas exceto DIETA
cat_cols = ['SEXO', 'ETAPA', 'CAFEÍNA']
for col in cat_cols:
    df[col] = df[col].fillna('NÃO INFORMADO')

# Processar coluna 'DIETA NAS ÚLTIMAS 24 H' em múltiplas colunas binárias
df['DIETA NAS ÚLTIMAS 24 H'] = df['DIETA NAS ÚLTIMAS 24 H'].fillna('')

df['lista_alimentos'] = df['DIETA NAS ÚLTIMAS 24 H'].str.split(',')
df['lista_alimentos'] = df['lista_alimentos'].apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else [])

df_alimentos = df['lista_alimentos'].explode().str.get_dummies().groupby(level=0).max()
df.drop(columns=['lista_alimentos'], inplace=True)

# One-hot encoding para as outras colunas categóricas
df_cat_encoded = pd.get_dummies(df[cat_cols], drop_first=True)

# Concatenar tudo
df_final = pd.concat([df, df_cat_encoded, df_alimentos], axis=1)

# Definir features
features = ['IDADE', 'PESO (kg)', 'ALTURA (m)', 'IMC'] + list(df_cat_encoded.columns) + list(df_alimentos.columns)

X = df_final[features]
y = df_final['REPETIU']

# --- Divisão treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

print("Quantidade de amostras ORIGINAIS no treino:", len(X_train))
print("Distribuição das classes no treino ORIGINAL:")
print(y_train.value_counts())

print("\nQuantidade de amostras ORIGINAIS no teste:", len(X_test))
print("Distribuição das classes no teste ORIGINAL:")
print(y_test.value_counts())

# --- Balanceamento SMOTE ---
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nQuantidade de amostras no treino BALANCEADO (após SMOTE):", len(X_train_bal))
print("Distribuição das classes no treino BALANCEADO:")
print(pd.Series(y_train_bal).value_counts())

# --- Modelo RandomForest ---
model = RandomForestClassifier(random_state=42, class_weight='balanced')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=skf, scoring='f1')
print(f"\nF1-score médio na validação cruzada: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

model.fit(X_train_bal, y_train_bal)

# --- Avaliação com limiar padrão 0.5 ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\U0001F4CA Relatório de Classificação (limiar = 0.5):")
print(classification_report(y_test, y_pred))
print("\U0001F4C9 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Acurácia: {acc:.2f}")
print(f"Precisão: {prec:.2f}")
print(f"Revocação (Recall): {rec:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"AUC-ROC: {auc:.2f}")

# --- Avaliação com limiar ajustado para melhorar recall ---
limiar = 0.30
y_pred_adjusted = (y_proba >= limiar).astype(int)

print(f"\n\U0001F4CA Relatório de Classificação (limiar ajustado = {limiar}):")
print(classification_report(y_test, y_pred_adjusted))
print("\U0001F4C9 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_adjusted))

acc_adj = accuracy_score(y_test, y_pred_adjusted)
prec_adj = precision_score(y_test, y_pred_adjusted)
rec_adj = recall_score(y_test, y_pred_adjusted)
f1_adj = f1_score(y_test, y_pred_adjusted)
auc_adj = roc_auc_score(y_test, y_proba)

print(f"Acurácia: {acc_adj:.2f}")
print(f"Precisão: {prec_adj:.2f}")
print(f"Revocação (Recall): {rec_adj:.2f}")
print(f"F1-score: {f1_adj:.2f}")
print(f"AUC-ROC: {auc_adj:.2f}")

# --- Importância das variáveis ---
importances = pd.Series(model.feature_importances_, index=X_train_bal.columns)
importances.nlargest(15).plot(kind='barh', color='skyblue')
plt.title('15 Variáveis Mais Relevantes para Prever Repetição de Exames')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

df_final.to_csv('df_final_exportado.csv', sep=';', decimal=',', index=False, encoding='utf-8-sig')


# --- Comparação percentual dos alimentos mais consumidos entre Repetiu e Não Repetiu ---

# Selecionar colunas dos alimentos
colunas_alimentos = df_alimentos.columns

# Número total de pessoas em cada grupo
n_repetiu = df_final[df_final['REPETIU'] == 1].shape[0]
n_nao_repetiu = df_final[df_final['REPETIU'] == 0].shape[0]

# Calcular percentual de consumo por alimento
perc_repetiu = df_final[df_final['REPETIU'] == 1][colunas_alimentos].sum() / n_repetiu * 100
perc_nao_repetiu = df_final[df_final['REPETIU'] == 0][colunas_alimentos].sum() / n_nao_repetiu * 100

# Unir em um DataFrame
df_percentual = pd.DataFrame({
    'Repetiu (%)': perc_repetiu,
    'Não Repetiu (%)': perc_nao_repetiu
})

# Ordenar pelos alimentos mais consumidos (soma das duas porcentagens)
df_percentual['Média (%)'] = (df_percentual['Repetiu (%)'] + df_percentual['Não Repetiu (%)']) / 2
df_percentual = df_percentual.sort_values(by='Média (%)', ascending=False).drop(columns='Média (%)')

# Plotar os top 15 alimentos
df_percentual.head(20).plot(kind='bar', figsize=(12, 6), color=['orangered', 'seagreen'])
plt.title('Top 15 Alimentos Mais Consumidos (%) - Repetiu vs Não Repetiu')
plt.ylabel('Percentual de Pessoas que Consumiram')
plt.xlabel('Alimentos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Grupo')
plt.show()

# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  # pip install imblearn

# --- Leitura dos dados ---
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRooSfamgfpn6HM3tXDhRzzeYvFjdubZ0gWBGDiklSY1fstXGe9iUIAQ-btWhfL8SRRe8nBJ1sU-0x-/pub?output=csv"

try:
    df = pd.read_csv(url, encoding='utf-8', decimal=',', thousands='.', on_bad_lines='skip')
    print("Dados carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar os dados: {e}")

# --- Pré-processamento ---

# Ajuste numérico
for col in ['IMC', 'ALTURA (m)']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# Mapear coluna alvo
df['REPETIU'] = df['REPETIU'].fillna('Não').map({'Sim': 1, 'Não': 0})

# Preencher NA nas categóricas exceto DIETA
cat_cols = ['SEXO', 'ETAPA', 'CAFEÍNA']
for col in cat_cols:
    df[col] = df[col].fillna('NÃO INFORMADO')

# Processar coluna 'DIETA NAS ÚLTIMAS 24 H' em múltiplas colunas binárias
df['DIETA NAS ÚLTIMAS 24 H'] = df['DIETA NAS ÚLTIMAS 24 H'].fillna('')

df['lista_alimentos'] = df['DIETA NAS ÚLTIMAS 24 H'].str.split(',')
df['lista_alimentos'] = df['lista_alimentos'].apply(lambda x: [item.strip() for item in x] if isinstance(x, list) else [])

df_alimentos = df['lista_alimentos'].explode().str.get_dummies().groupby(level=0).max()
df.drop(columns=['lista_alimentos'], inplace=True)

# One-hot encoding para as outras colunas categóricas
df_cat_encoded = pd.get_dummies(df[cat_cols], drop_first=True)

# Concatenar tudo
df_final = pd.concat([df, df_cat_encoded, df_alimentos], axis=1)

# Definir features
features = ['IDADE', 'PESO (kg)', 'ALTURA (m)', 'IMC'] + list(df_cat_encoded.columns) + list(df_alimentos.columns)

X = df_final[features]
y = df_final['REPETIU']

# --- Divisão treino/teste ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

print("Quantidade de amostras ORIGINAIS no treino:", len(X_train))
print("Distribuição das classes no treino ORIGINAL:")
print(y_train.value_counts())

print("\nQuantidade de amostras ORIGINAIS no teste:", len(X_test))
print("Distribuição das classes no teste ORIGINAL:")
print(y_test.value_counts())

# --- Balanceamento SMOTE ---
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nQuantidade de amostras no treino BALANCEADO (após SMOTE):", len(X_train_bal))
print("Distribuição das classes no treino BALANCEADO:")
print(pd.Series(y_train_bal).value_counts())

# --- Modelo RandomForest ---
model = RandomForestClassifier(random_state=42, class_weight='balanced')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=skf, scoring='f1')
print(f"\nF1-score médio na validação cruzada: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

model.fit(X_train_bal, y_train_bal)

# --- Avaliação com limiar padrão 0.5 ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\U0001F4CA Relatório de Classificação (limiar = 0.5):")
print(classification_report(y_test, y_pred))
print("\U0001F4C9 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Acurácia: {acc:.2f}")
print(f"Precisão: {prec:.2f}")
print(f"Revocação (Recall): {rec:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"AUC-ROC: {auc:.2f}")

# --- Avaliação com limiar ajustado para melhorar recall ---
limiar = 0.30
y_pred_adjusted = (y_proba >= limiar).astype(int)

print(f"\n\U0001F4CA Relatório de Classificação (limiar ajustado = {limiar}):")
print(classification_report(y_test, y_pred_adjusted))
print("\U0001F4C9 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_adjusted))

acc_adj = accuracy_score(y_test, y_pred_adjusted)
prec_adj = precision_score(y_test, y_pred_adjusted)
rec_adj = recall_score(y_test, y_pred_adjusted)
f1_adj = f1_score(y_test, y_pred_adjusted)
auc_adj = roc_auc_score(y_test, y_proba)

print(f"Acurácia: {acc_adj:.2f}")
print(f"Precisão: {prec_adj:.2f}")
print(f"Revocação (Recall): {rec_adj:.2f}")
print(f"F1-score: {f1_adj:.2f}")
print(f"AUC-ROC: {auc_adj:.2f}")

# --- Importância das variáveis ---
importances = pd.Series(model.feature_importances_, index=X_train_bal.columns)
importances.nlargest(15).plot(kind='barh', color='skyblue')
plt.title('15 Variáveis Mais Relevantes para Prever Repetição de Exames')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

df_final.to_csv('df_final_exportado.csv', sep=';', decimal=',', index=False, encoding='utf-8-sig')


# --- Comparação percentual dos alimentos mais consumidos entre Repetiu e Não Repetiu ---

# Selecionar colunas dos alimentos
colunas_alimentos = df_alimentos.columns

# Número total de pessoas em cada grupo
n_repetiu = df_final[df_final['REPETIU'] == 1].shape[0]
n_nao_repetiu = df_final[df_final['REPETIU'] == 0].shape[0]

# Calcular percentual de consumo por alimento
perc_repetiu = df_final[df_final['REPETIU'] == 1][colunas_alimentos].sum() / n_repetiu * 100
perc_nao_repetiu = df_final[df_final['REPETIU'] == 0][colunas_alimentos].sum() / n_nao_repetiu * 100

# Unir em um DataFrame
df_percentual = pd.DataFrame({
    'Repetiu (%)': perc_repetiu,
    'Não Repetiu (%)': perc_nao_repetiu
})

# Ordenar pelos alimentos mais consumidos (soma das duas porcentagens)
df_percentual['Média (%)'] = (df_percentual['Repetiu (%)'] + df_percentual['Não Repetiu (%)']) / 2
df_percentual = df_percentual.sort_values(by='Média (%)', ascending=False).drop(columns='Média (%)')

# Plotar os top 15 alimentos
df_percentual.head(20).plot(kind='bar', figsize=(12, 6), color=['orangered', 'seagreen'])
plt.title('Top 15 Alimentos Mais Consumidos (%) - Repetiu vs Não Repetiu')
plt.ylabel('Percentual de Pessoas que Consumiram')
plt.xlabel('Alimentos')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title='Grupo')
plt.show()


# --- Imports ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE  # pip install imblearn

# --- Leitura dos dados ---
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRooSfamgfpn6HM3tXDhRzzeYvFjdubZ0gWBGDiklSY1fstXGe9iUIAQ-btWhfL8SRRe8nBJ1sU-0x-/pub?output=csv"

try:
    df = pd.read_csv(url, encoding='utf-8', decimal=',', thousands='.', on_bad_lines='skip')
    print("Dados carregados com sucesso.")
except Exception as e:
    print(f"Erro ao carregar os dados: {e}")

# --- Pré-processamento ---
for col in ['IMC', 'ALTURA (m)']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

df['REPETIU'] = df['REPETIU'].fillna('Não').map({'Sim': 1, 'Não': 0})

cat_cols = ['SEXO', 'ETAPA', 'CAFEÍNA', 'DIETA NAS ÚLTIMAS 24 H']
for col in cat_cols:
    df[col] = df[col].fillna('NÃO INFORMADO')

df_encoded = pd.get_dummies(df[cat_cols], drop_first=True)
df_final = pd.concat([df, df_encoded], axis=1)

features = ['IDADE', 'PESO (kg)', 'ALTURA (m)', 'IMC'] + list(df_encoded.columns)
X = df_final[features]
y = df_final['REPETIU']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

print("Quantidade de amostras ORIGINAIS no treino:", len(X_train))
print("Distribuição das classes no treino ORIGINAL:")
print(y_train.value_counts())

print("\nQuantidade de amostras ORIGINAIS no teste:", len(X_test))
print("Distribuição das classes no teste ORIGINAL:")
print(y_test.value_counts())

# --- Balanceamento SMOTE ---
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("\nQuantidade de amostras no treino BALANCEADO (após SMOTE):", len(X_train_bal))
print("Distribuição das classes no treino BALANCEADO:")
print(pd.Series(y_train_bal).value_counts())

# --- Modelo RandomForest ---
model = RandomForestClassifier(random_state=42, class_weight='balanced')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_bal, y_train_bal, cv=skf, scoring='f1')
print(f"\nF1-score médio na validação cruzada: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

model.fit(X_train_bal, y_train_bal)

# --- Avaliação com limiar padrão 0.5 ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\U0001F4CA Relatório de Classificação (limiar = 0.5):")
print(classification_report(y_test, y_pred))
print("\U0001F4C9 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Acurácia: {acc:.2f}")
print(f"Precisão: {prec:.2f}")
print(f"Revocação (Recall): {rec:.2f}")
print(f"F1-score: {f1:.2f}")
print(f"AUC-ROC: {auc:.2f}")

# --- Avaliação com limiar ajustado para melhorar recall ---
limiar = 0.30
y_pred_adjusted = (y_proba >= limiar).astype(int)

print(f"\n\U0001F4CA Relatório de Classificação (limiar ajustado = {limiar}):")
print(classification_report(y_test, y_pred_adjusted))
print("\U0001F4C9 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_adjusted))

acc_adj = accuracy_score(y_test, y_pred_adjusted)
prec_adj = precision_score(y_test, y_pred_adjusted)
rec_adj = recall_score(y_test, y_pred_adjusted)
f1_adj = f1_score(y_test, y_pred_adjusted)
auc_adj = roc_auc_score(y_test, y_proba)

print(f"Acurácia: {acc_adj:.2f}")
print(f"Precisão: {prec_adj:.2f}")
print(f"Revocação (Recall): {rec_adj:.2f}")
print(f"F1-score: {f1_adj:.2f}")
print(f"AUC-ROC: {auc_adj:.2f}")

import matplotlib.pyplot as plt

# --- Importância das variáveis ---
importances = pd.Series(model.feature_importances_, index=X_train_bal.columns)
top_15 = importances.nlargest(15)

# Criar figura com tamanho maior para acomodar os textos
fig, ax = plt.subplots(figsize=(10, 8))  # ajuste o tamanho se necessário
top_15.plot(kind='barh', color='skyblue', ax=ax)

ax.set_title('15 Variáveis Mais Relevantes para Prever Repetição de Exames')
ax.invert_yaxis()

# Salvar figura com bordas ajustadas e alta resolução
plt.savefig('/content/drive/MyDrive/Colab Notebooks/Joao_Aluno_Tati_TCC/importancia_variaveis.jpeg', format='jpeg', dpi=300)

plt.show()


# Salvar figura como JPEG com 300 dpi

plt.show()

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Avaliação do modelo
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # Probabilidade da classe positiva

# Métricas principais
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n📊 Avaliação do Modelo:")
print(f"Acurácia:   {acc:.2f}")
print(f"Precisão:   {prec:.2f}")
print(f"Revocação:  {rec:.2f}")
print(f"F1-score:   {f1:.2f}")
print(f"AUC-ROC:    {auc:.2f}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
print("\nMatriz de Confusão:")
print(cm)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("Falso Positivo (1 - Especificidade)")
plt.ylabel("Verdadeiro Positivo (Sensibilidade)")
plt.title("📈 Curva ROC")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
