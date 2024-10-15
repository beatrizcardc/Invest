import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Customização de CSS para fundo verde escuro e letras brancas
st.markdown(
    """
    <style>
    body {
        background-color: #D3D3D3; /* Fundo cinza claro */
        color: black;
    }
    .stApp {
        background-color: #D3D3D3;
        color: black;
    }
    h1, h2, h3, h4, h5, h6 {
        color: black;
    }
    .stButton button {
        background-color: #004d00;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# Explicação inicial do projeto
st.write("## Projeto de Otimização de Investimentos")
st.write("Esta aplicação usa algoritmos genéticos para otimizar a alocação de investimentos em ativos variados, considerando retornos e riscos.")

# Título da aplicação
st.title("Otimização de Investimentos - Realize seus Objetivos")

# Caixas de texto explicativas
st.write("### Conceitos Importantes")
st.write("**Mutação**: A mutação é uma forma de introduzir variações em uma população de soluções.")
st.write("**Elitismo**: O elitismo preserva as melhores soluções encontradas em uma geração.")
st.write("**Sharpe Ratio**: Uma medida que compara o retorno de um investimento com seu risco.")

# Entrada do usuário: valor total do investimento
valor_total = st.number_input("Digite o valor total do investimento", value=100000)

# Adicionar controle para selecionar a taxa de mutação com explicação
taxa_mutacao = st.slider(
    "Taxa de Mutação", min_value=0.01, max_value=0.2, value=0.05, step=0.01, 
    help="A taxa de mutação ajuda a garantir a exploração de novas soluções."
)

# Adicionar controle para selecionar a taxa livre de risco
taxa_livre_risco = st.number_input("Taxa Livre de Risco (Ex: SELIC, POUPANÇA)", value=0.1075)

# Pergunta sobre o uso do elitismo
usar_elitismo = st.selectbox("Deseja usar elitismo?", options=["Sim", "Não"])
usar_elitismo = True if usar_elitismo == "Sim" else False

# Placeholder para o resultado final
resultado_placeholder = st.empty()

# Critério de parada
geracoes_sem_melhoria = st.number_input("Número de gerações sem melhora no fitness para parar", min_value=10, value=20)

# Carregar os dados do CSV atualizado diretamente do GitHub
csv_url = 'https://raw.githubusercontent.com/beatrizcardc/TechChallenge2_Otimizacao/main/Pool_Investimentos.csv'
try:
    df = pd.read_csv(csv_url)
except Exception as e:
    st.error(f"Erro ao carregar o CSV: {e}")
    st.stop()

# Funções auxiliares
def calcular_sharpe(portfolio, retornos, riscos, taxa_livre_risco):
    retorno_portfolio = np.dot(portfolio, retornos)
    risco_portfolio = np.sqrt(np.dot(portfolio, riscos ** 2))
    if risco_portfolio < 0.01:
        risco_portfolio = 0.01
    return (retorno_portfolio - taxa_livre_risco) / risco_portfolio

def gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, num_ativos):
    populacao = [genoma_inicial]
    for _ in range(num_portfolios - 1):
        populacao.append(np.random.dirichlet(np.ones(num_ativos)))
    return populacao

def selecao_torneio(populacao, fitness_scores, tamanho_torneio=3):
    selecionados = []
    for _ in range(len(populacao)):
        competidores = np.random.choice(len(populacao), tamanho_torneio, replace=False)
        vencedor = competidores[np.argmax(fitness_scores[competidores])]
        selecionados.append(populacao[vencedor])
    return selecionados

def ajustar_alocacao(portfolio, limite_max=0.25):
    portfolio = np.clip(portfolio, 0, limite_max)
    portfolio /= portfolio.sum()
    return portfolio

def mutacao(portfolio, taxa_mutacao, limite_max=0.25):
    if np.random.random() < taxa_mutacao:
        i = np.random.randint(0, len(portfolio))
        portfolio[i] += np.random.uniform(-0.1, 0.1)
        portfolio = ajustar_alocacao(portfolio, limite_max)
    return portfolio

def cruzamento(pai1, pai2):
    num_pontos_corte = np.random.randint(1, 4)
    pontos_corte = sorted(np.random.choice(range(1, len(pai1)), num_pontos_corte, replace=False))
    filho1, filho2 = pai1.copy(), pai2.copy()
    for i in range(0, len(pontos_corte) - 1, 2):
        filho1[pontos_corte[i]:pontos_corte[i+1]] = pai2[pontos_corte[i]:pontos_corte[i+1]]
        filho2[pontos_corte[i]:pontos_corte[i+1]] = pai1[pontos_corte[i]:pontos_corte[i+1]]
    return ajustar_alocacao(filho1), ajustar_alocacao(filho2)

# Algoritmo genético com critério de parada
def algoritmo_genetico(retornos, riscos, genoma_inicial, taxa_livre_risco, num_portfolios=100, geracoes=100, usar_elitismo=True, taxa_mutacao=0.05, crit_parada=20):
    populacao = gerar_portfolios_com_genoma_inicial(genoma_inicial, num_portfolios, len(retornos))
    melhor_portfolio = genoma_inicial
    melhor_sharpe = calcular_sharpe(genoma_inicial, retornos, riscos, taxa_livre_risco)
    contador_sem_melhoria = 0

    historico_sharpe = []

    for geracao in range(geracoes):
        fitness_scores = np.array([calcular_sharpe(port, retornos, riscos, taxa_livre_risco) for port in populacao])
        indice_melhor_portfolio = np.argmax(fitness_scores)
        if fitness_scores[indice_melhor_portfolio] > melhor_sharpe:
            melhor_sharpe = fitness_scores[indice_melhor_portfolio]
            melhor_portfolio = populacao[indice_melhor_portfolio]
            contador_sem_melhoria = 0
        else:
            contador_sem_melhoria += 1

        historico_sharpe.append(melhor_sharpe)

        populacao = selecao_torneio(populacao, fitness_scores)
        nova_populacao = []
        for i in range(0, len(populacao), 2):
            pai1, pai2 = populacao[i], populacao[i+1]
            filho1, filho2 = cruzamento(pai1, pai2)
            nova_populacao.append(mutacao(filho1, taxa_mutacao))
            nova_populacao.append(mutacao(filho2, taxa_mutacao))

        if usar_elitismo:
            nova_populacao[0] = melhor_portfolio

        populacao = nova_populacao

        if contador_sem_melhoria >= crit_parada:
            break

    # Exibir gráfico de evolução do Sharpe Ratio
    plt.plot(historico_sharpe)
    plt.title('Evolução do Sharpe Ratio')
    plt.xlabel('Gerações')
    plt.ylabel('Melhor Sharpe Ratio')
    st.pyplot(plt)

    return melhor_portfolio

# Gerar o portfólio otimizado
genoma_inicial = np.random.dirichlet(np.ones(34))
melhor_portfolio = algoritmo_genetico(
    retornos=df['Rentabilidade 12 meses'].values, 
    riscos=riscos_completos_final, 
    genoma_inicial=genoma_inicial, 
    taxa_livre_risco=taxa_livre_risco,
    usar_elitismo=usar_elitismo,
    taxa_mutacao=taxa_mutacao,
    crit_parada=geracoes_sem_melhoria
)

# Distribuir o valor total de investimento
distribuicao_investimento = melhor_portfolio * valor_total
distribuicao_df = pd.DataFrame({
    'Ativo': df['Ativo'],
    'Alocacao (%)': melhor_portfolio * 100,
    'Valor Investido (R$)': distribuicao_investimento
})

# Ordenar tabela pela alocação percentual
distribuicao_df = distribuicao_df.sort_values(by='Alocacao (%)', ascending=False)

# Mostrar a tabela e gráfico final
st.write("Distribuição ideal de investimento:")
st.dataframe(distribuicao_df)

# Exibir gráfico de barras da alocação
distribuicao_df.plot(kind='bar', x='Ativo', y='Alocacao (%)', legend=False)
plt.title('Distribuição Percentual por Ativo')
plt.ylabel('Alocacao (%)')
st.pyplot(plt)

# Download do CSV atualizado
csv = distribuicao_df.to_csv(index=False)
st.download_button(label="Baixar CSV Atualizado", data=csv, file_name='Pool_Investimentos_Atualizacao2.csv', mime='text/csv')
