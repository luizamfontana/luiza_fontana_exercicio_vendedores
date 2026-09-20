
# Atividade - AC - Desempenho de vendedores

# Disciplina: Programação para análise de dados
# Professor: Laerte Jun Takeuti

# Estudante: Luiza Mulinari Fontana
#               202501379559

# E-mail: luizamulinarifontana@gmail.com
#       ou 202501379559@alunos.ibmec.edu.br

# Importando bibliotecas
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

# ================================================================================
# 1.INTEGRAÇÃO E QUALIDADE DOS DADOS
# ================================================================================

# 1) Leia os dois arquivos CSV em DataFrames do Pandas

df_des = pd.read_csv("vendedores_desempenho.csv") # des = desempenho
df_con = pd.read_csv("vendedores_contexto.csv")   # con = contexto

# ________________________________________________________________________________

# 2) Informe quantidade de linhas e colunas de cada base

print(f"Desempenho: (Linhas, colunas): {df_des.shape}") 
print(f"Contexto: (Linhas, colunas): {df_con.shape}") 

# ________________________________________________________________________________

# 3)  Identifique qual coluna deve ser utilizada para integrar as duas bases.

df_des.info 
# ou
df_des.head()
# ou
df_des.columns
# Colunas: id_vendedor, nome, idade, horas_treinamento, 
#   faltas, numero_clientes e vendas_mes

df_con.info 
# ou
df_con.head()
# ou
df_con.columns
# Colunas: id_vendedor, distancia_empresa_km, salario, recebe_bonus

    # Com isso, opta-se pelo ID do Vendedor, que é a coluna em comum entre as bases, 
    # além de ser uma variável identificadora, teoricamente única por vendedor.

# ________________________________________________________________________________

# 4) Identifique se a chave é única em cada base.

df_des["id_vendedor"].duplicated().sum() # =0: sem duplicatas
df_con["id_vendedor"].duplicated().sum() # =0: sem duplicatas

    # Ou seja, sim, a chave é única em cada base.

# ________________________________________________________________________________

# 5) Faça o merge das duas bases preservando todos os vendedores da base de desempenho.

df = pd.merge(df_des, df_con, on = "id_vendedor", how= "left" )

df.head()

# Left preserva todos da base de desempenho
# Right preservaria todos da base de contexto
# Inner preservaria apenas os em comum entre as bases
# Outer juntaria tudo, inclusive os que só estão em uma das bases

# ________________________________________________________________________________

# 6) Informe a quantidade de registros após o merge

print(f"DataFrame após merge: (Linhas, colunas) = {df.shape}")

# ________________________________________________________________________________

# 7) Verifique a quantidade de valores ausentes em cada variável da base integrada.

print(f"Dados faltantes por variável: {df.isnull().sum()}")

# ________________________________________________________________________________

# 8) Identifique quais variáveis possuem dados faltantes e 
# comente possíveis razões para isso.

"""
Variáveis com dados faltantes: horas_treinamento, faltas, numero_clientes,
# vendas_mes, distancia_empresa_km, salario e recebe_bonus

# POSSÍVEIS RAZÕES: Funcionários novos/em treinamento 
# (sem faltas, clientes e vendas registrados ainda), outros de férias ou em alguma 
# licença, por exemplo, ciclo periódico de faltas, treinamento, vendas, ou metas ainda 
# não fechado, etc.
"""

# ________________________________________________________________________________

# ================================================================================
# 2. ESTATÍSTICA DESCRITIVA
# ================================================================================

# 9)	Calcule a média de vendas_mes.

df["vendas_mes"].mean()
print(f"Média de vendas por mês: {df["vendas_mes"].mean():.2f}")

# ________________________________________________________________________________

# 10)	Calcule a mediana de vendas_mes.

df["vendas_mes"].median()
print(f"Mediana de vendas por mês: {df["vendas_mes"].median():.2f}")

# ________________________________________________________________________________

# 11) Informe o valor mínimo e o valor máximo de vendas_mes.

df["vendas_mes"].max()
df["vendas_mes"].min()

print(f"Vendas por mês: (máx ; mín) = ({df["vendas_mes"].max()} ; {df["vendas_mes"].min()})")

# ________________________________________________________________________________

# 12) Identifique possíveis outliers em vendas_mes usando o critério do intervalo 
# interquartil (IQR).

Q1 = df["vendas_mes"].quantile(0.25)
Q3 = df["vendas_mes"].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = []

for venda in df["vendas_mes"]:
    if venda > limite_superior or venda < limite_inferior:
        outliers.append(venda)

print(f"Limite inferior: {limite_inferior:.2f}")
print(f"Limite superior: {limite_superior:.2f}")
print(f"Quantidade de outliers: {len(outliers)}")
print(outliers)

# ________________________________________________________________________________

# 13) Compare a média de vendas entre vendedores que recebem bônus e vendedores 
# que não recebem bônus.

df.groupby("recebe_bonus")["vendas_mes"].mean()
print(f"Média de vendas por categoria 'recebe bônus': {df.groupby("recebe_bonus")["vendas_mes"].mean()}")

# A média dos que recebem bônus é consideravelmente maior:
#       53745,47 >> 45481,95

# ________________________________________________________________________________

# 14) Calcule a correlação entre as variáveis

df['recebe_bonus_num'] = df['recebe_bonus'].map({'Sim': 1, 'Não': 0})

correlacoes = df.corr(numeric_only = True)
print(correlacoes)

# ________________________________________________________________________________

# 15) Interprete o sinal das correlações encontradas.

"""
Quanto a vendas_mes, que é o que estávamos analisando, a idade, faltas, 
distancia_empresa_km e o salário têm correlação negativa.

Idade, principalmente por fatores de menos experiência; faltas, provavelmente 
atreladas à baixa dedicação ou contexto atual prejudicial ao bom desempenho;
distancia_empresa_km também prejudicando principalmente por questões de motivação
do trabalhador, horas gastas em transporte, e/ou, provavelmente, condições de vida;
e, por fim, o salário surpreende, mas pode ser justificado pelo fato de que, no meio
comercial, trabalha-se muito com metas e bônus: quem tem salário alto/suficiente, não 
precisa necessariamente bater metas todo mês, então seu desempenho não tende a ser o 
melhor no quesito vendas/mês.

O restante (horas_treinamento, numero_clientes, e recebe_bonus) possuem correlação 
positiva, o que faz sentido, e é esperado.
"""

# ________________________________________________________________________________


# ================================================================================
# 3. REGRESSÃO LINEAR MÚLTIPLA
# ================================================================================

# 16) Crie uma base para regressão contendo apenas as variáveis: vendas_mes, 
# horas_treinamento, faltas, numero_clientes, distancia_empresa_km e salario.

col = df[["vendas_mes", "horas_treinamento", "faltas", "numero_clientes",
          "distancia_empresa_km", "salario"]]


# ________________________________________________________________________________

# 17) Remova apenas as observações com dados ausentes nas variáveis que serão utilizadas 
# no modelo.

col2 = col.dropna()
print(col2)

# Testando:
col2.isnull().sum() # dados ausentes removidos

# ________________________________________________________________________________

# 18) Informe quantas observações foram utilizadas na regressão.

col2.shape
print(f"(Linhas, colunas) da base da regressão = {col2.shape}")

# ________________________________________________________________________________

# 19) Defina vendas_mes como variável dependente.

y = col2["vendas_mes"]

# ________________________________________________________________________________

# 20) Utilize como variáveis explicativas: horas_treinamento, faltas, numero_clientes, 
# distancia_empresa_km e salario.

x = col2[["horas_treinamento","faltas","numero_clientes","distancia_empresa_km","salario"]]

# ________________________________________________________________________________

# 21.	Adicione a constante ao modelo e estime uma regressão linear múltipla 
# utilizando Statsmodels.

x = sm.add_constant(x)

modelo = sm.OLS(y, x, missing="drop").fit()

# ________________________________________________________________________________

# 22.	Apresente o resumo completo da regressão.

print (modelo.summary())

# ________________________________________________________________________________

# 23.	Interprete o coeficiente de horas_treinamento, mantendo as demais variáveis constantes.

"""
 Coef(horas_treinamento) = 1022.5803 -> Indica que o aumento de uma unidade de hora de 
 treinamento aumenta em 1022.5803 unidades o valor da variável dependente vendas/mês, e 
 seu impacto é estatisticamente significativo, a nível de 0,1%, pois p < 0.001.
"""

# ________________________________________________________________________________

# 24.	Interprete o coeficiente de faltas, mantendo as demais variáveis constantes.
"""
Coef(faltas) = -1014.3678 -> Indica que o aumento de uma unidade de falta diminui
em 1014.3678 unidades o valor da variável dependente vendas/mês, e seu impacto é 
estatisticamente significativo, a nível de 5%, pois p < 0.05.

"""
# ________________________________________________________________________________

# 25.	Interprete o coeficiente de numero_clientes, mantendo as demais variáveis constantes.
"""
Coef(numero_clientes) = 990.3844 -> Indica que o aumento de uma unidade de cliente 
aumenta em 990.3844 o valor da variável dependente vendas/mês, e seu impacto é 
estatisticamente significativo, a nível de 0,1%, pois p < 0.001.
"""

# ________________________________________________________________________________

# 26.	Interprete o coeficiente de distancia_empresa_km.

"""
Coef(distancia_empresa_km) = 49.6153 -> Indica supostamente que o aumento de uma unidade 
de km de distância da empresa aumenta em 49.6153 o valor da variável dependente vendas/mês, 
porém seu impacto é considerado não-significativo estatisticamente, pois p >>> 5% ou 0.05.
"""
# ________________________________________________________________________________

# 27.	Interprete o coeficiente de salario.

"""
Coef(salario) = 0.5123 -> Indica supostamente que o aumento de uma unidade de salário
aumenta em 0.5123 o valor da variável dependente vendas/mês, porém seu impacto é 
considerado não-significativo estatisticamente, pois p >> 5% ou 0.05.
"""
# ________________________________________________________________________________

# 28.	Apresente os p-valores de todas as variáveis explicativas.

"""
    Variável                p-valor

    horas_treinamento        0.000(***)
    faltas                   0.041 (*)
    numero_clientes          0.000 (***)
    distancia_empresa_km     0.547 ()
    salario                  0.463 ()
"""
# ________________________________________________________________________________

# 29.	Identifique quais variáveis são estatisticamente significativas ao nível de 5%.

"""
Horas_treinamento, faltas e numero_clientes, pois seus valores de p são menores que 5%.
"""

# ________________________________________________________________________________

# 30.	Identifique quais variáveis não apresentam evidência estatística de associação com vendas_mes ao nível de 5%.

"""
Distancia_empresa_km e salario, por apresentarem valores de p muito superiores
ao padrão de corte de 5% = 0.05.
"""

# ________________________________________________________________________________

# 31.	Informe o valor do R² do modelo.

"""
R² = 0.632      R² ajustado = 0.614
Isso indica que o modelo explica 63,2% (ou 61,4%, analisando o R² ajustado - que
penaliza o acréscimo de variáveis no modelo, permitindo comparação entre modelos
diversos) da variação/comportamento da variável dependente Y (vendas/mês), o que é uma
porcentagem considerável para o caso desse modelo.
"""

