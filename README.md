# Car Insurance Premium Prediction

Objetivo: criar o pipeline de tratamento dos dados para construção dos modelos de Machine Learning. O Modelo a ser construido será um modelo de GLM (Modelos Lineares Generalizados) para predição dos prêmios, mas não se preocupe com o modelo nesta entrega. O foco é tratar os dados.

Este conjunto de dados contém 1.000 linhas de dados sintéticos que simulam prêmios de seguro de automóveis, calculados usando uma fórmula linear. Ele incorpora características-chave como idade do motorista, experiência de direção, histórico de acidentes, quilometragem anual e ano de fabricação do carro para prever o prêmio do seguro.
O conjunto de dados é ideal para explorar modelos de regressão linear, análise de importância de características e modelagem preditiva no setor de seguros. Ele foi inspirado em fatores do mundo real que influenciam os prêmios de seguro, garantindo padrões realistas e insights significativos.

### Dados de treinamento e test dos modelos de ML
* Dados de treinamento do ML: car_insurance_premium_dataset.csv
* Dados de test do ML: car_insurance_premium_dataset_TEST.csv

## Tarefas a serem feitas
1. Aplicar lowercase em todas as colunas (uniformizar os nomes das colunas);
2. Excluir caracteres especiais dos nomes das colunas, incluindo o espaço em branco entre nome de colunas. Substitua os espaços em branco das colunas por "_";
3. Tratamento dos outliers usando IQR e ZScore para cada da uma das variáveis do dataset. Substitua os outliers por mediana;
4. Tratamento dos missing values para cada uma das variáveis do dataset. Substitua os missing values por mediana;
6. Lidar com dados categóricos. Descreva o tratamento que você aplicou.
7. Fazer EDA (Análise Exploratória de Dados);
8. Descreva os tipos de dados: float, int64, integer, ...

## Descrição dos Tipos de Dados e Tratamento de Dados Categóricos

### Tipos de Dados
Após o processamento, todos os dados foram convertidos para o tipo `float64`, conforme exibido pelo script de processamento.

### Tratamento de Dados Categóricos
O conjunto de dados não continha variáveis categóricas. Se houvesse, o tratamento seria feito usando a função `pd.get_dummies` do pandas para converter as variáveis categóricas em variáveis dummy (one-hot encoding). Este método é preferível para evitar a criação de uma relação ordinal artificial entre as categorias.