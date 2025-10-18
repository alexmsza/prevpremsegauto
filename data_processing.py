import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def carregar_dados(caminho_arquivo):
    """Carrega o conjunto de dados de um arquivo CSV."""
    return pd.read_csv(caminho_arquivo)

def limpar_nomes_colunas(df):
    """Limpa e padroniza os nomes das colunas."""
    colunas = df.columns
    novas_colunas = []
    for col in colunas:
        nova_col = col.lower()
        nova_col = nova_col.replace(' ', '_')
        nova_col = nova_col.replace('(', '')
        nova_col = nova_col.replace(')', '')
        nova_col = nova_col.replace('$', '')
        novas_colunas.append(nova_col)
    df.columns = novas_colunas
    return df

def tratar_outliers_iqr(df, coluna):
    """Trata outliers usando o método IQR."""
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    mediana = df[coluna].median()
    df[coluna] = np.where((df[coluna] < limite_inferior) | (df[coluna] > limite_superior), mediana, df[coluna])
    return df

def tratar_outliers_zscore(df, coluna, limiar=3):
    """Trata outliers usando o método Z-score."""
    z_scores = np.abs(stats.zscore(df[coluna]))
    mediana = df[coluna].median()
    df[coluna] = np.where(z_scores > limiar, mediana, df[coluna])
    return df

def tratar_valores_ausentes(df):
    """Trata valores ausentes imputando a mediana."""
    for coluna in df.columns:
        if df[coluna].isnull().sum() > 0:
            mediana = df[coluna].median()
            df[coluna].fillna(mediana, inplace=True)
    return df

def tratar_dados_categoricos(df):
    """
    Trata dados categóricos usando one-hot encoding.
    Este método é escolhido porque o número de categorias é esperado ser pequeno
    e evita a introdução de relações ordinais onde não existem.
    """
    # Este conjunto de dados parece não ter variáveis categóricas com base na inspeção inicial.
    # Se houvesse, usaríamos get_dummies.
    # Por exemplo:
    # if 'coluna_categorica' in df.columns:
    #     df = pd.get_dummies(df, columns=['coluna_categorica'], drop_first=True)
    print("Nenhuma variável categórica foi identificada neste conjunto de dados.")
    return df

def executar_eda(df):
    """Executa a análise exploratória de dados (EDA)."""
    print("Análise Exploratória de Dados (EDA):")
    print("\nEstatísticas Resumo:")
    print(df.describe())

    print("\nTipos de Dados:")
    print(df.info())

    # Plotando distribuições
    for coluna in df.columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[coluna], kde=True)
        plt.title(f'Distribuição de {coluna}')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
        plt.savefig(f'{coluna}_distribuicao.png')
        plt.close()

    # Plotando correlações
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlação')
    plt.savefig('matriz_de_correlacao.png')
    plt.close()

def main():
    """Função principal para executar o pipeline de processamento de dados."""
    caminho_arquivo = 'car_insurance_premium_dataset.csv'
    df = carregar_dados(caminho_arquivo)

    # 1. & 2. Limpar nomes das colunas
    df = limpar_nomes_colunas(df)
    print("Nomes de colunas limpos:")
    print(df.columns)

    # 3. Tratar outliers
    # Aplicando IQR e Z-score para cada coluna numérica
    colunas_numericas = df.select_dtypes(include=np.number).columns
    for col in colunas_numericas:
        df = tratar_outliers_iqr(df, col)
        df = tratar_outliers_zscore(df, col)

    # 4. Tratar valores ausentes
    df = tratar_valores_ausentes(df)
    print("\nValores ausentes tratados.")

    # 5. Tratar dados categóricos
    df = tratar_dados_categoricos(df)

    # 6. & 7. Executar EDA e descrever tipos de dados
    executar_eda(df)

    # Salvar os dados processados
    df.to_csv('processed_car_insurance_premium.csv', index=False)
    print("\nDados processados salvos em 'processed_car_insurance_premium.csv'")
    print("\nGráficos da EDA salvos como arquivos PNG.")

if __name__ == '__main__':
    main()
