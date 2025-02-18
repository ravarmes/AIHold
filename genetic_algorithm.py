import random
import math
from deap import base, creator, tools
import matplotlib.pyplot as plt

# ================= CONSTANTES E CONFIGURAÇÕES =================
# Tickers e preços
TICKERS_ACOES = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3", "WEGE3", "BBAS3", "SUZB3", "LREN3", "MGLU3"]
TICKERS_FIIS = ["HGLG11", "KNRI11", "MXRF11", "XPLG11", "VISC11"]
PRECOS_ACOES = {ticker: round(random.uniform(30, 30), 2) for ticker in TICKERS_ACOES}
PRECOS_FIIS = {ticker: round(random.uniform(120, 120), 2) for ticker in TICKERS_FIIS}

# Configurações de investimento
ORCAMENTO = 1000  # Valor disponível para o investimento
PROPORCAO_ACOES = 0.7  # Proporção ideal para ações
PROPORCAO_FIIS = 0.3  # Proporção ideal para FIIs

# Configurações do algoritmo genético
PESO_PROPORCOES_GERAIS = 0.6  # Peso para a proporção de ações e FIIs
PESO_DISTRIBUICAO_ATIVOS = 0.4  # Peso para a distribuição dos ativos
TAXA_MUTACAO = 0.15  # Taxa de mutação
TAMANHO_TORNEIO = 3  # Tamanho do torneio para seleção
TAMANHO_POPULACAO = 300  # Tamanho da população
NUM_GERACOES = 200  # Número de gerações

# ================= FUNÇÕES AUXILIARES =================
def calcular_investimento(individuo):
    """Calcula o investimento total em ações e FIIs."""
    investimento_acoes = sum(individuo[i] * PRECOS_ACOES[t] for i, t in enumerate(TICKERS_ACOES))
    investimento_fiis = sum(individuo[i + len(TICKERS_ACOES)] * PRECOS_FIIS[t] for i, t in enumerate(TICKERS_FIIS))
    return investimento_acoes, investimento_fiis, investimento_acoes + investimento_fiis

def exibir_configuracao_carteira(configuracao, individuo=None):
    """Exibe a configuração da carteira, inicial ou final."""
    if configuracao == "inicial":
        print("=== Configuração Inicial da Carteira ===")
        print(f"% Ações: {PROPORCAO_ACOES * 100:.2f}%")
        print(f"% FIIs: {PROPORCAO_FIIS * 100:.2f}%")
        
        proporcao_acao = PROPORCAO_ACOES * 100 / len(TICKERS_ACOES)
        proporcao_fii = PROPORCAO_FIIS * 100 / len(TICKERS_FIIS)
        
        for ticker in TICKERS_ACOES:
            print(f"{ticker}: {proporcao_acao:.2f}% dentro de Ações")
        for ticker in TICKERS_FIIS:
            print(f"{ticker}: {proporcao_fii:.2f}% dentro de FIIs")
    
    elif configuracao == "final" and individuo is not None:
        investimento_acoes, investimento_fiis, investimento_total = calcular_investimento(individuo)
        print("=== Configuração Final da Carteira ===")
        print(f"% Ações: {(investimento_acoes / investimento_total) * 100:.2f}%")
        print(f"% FIIs: {(investimento_fiis / investimento_total) * 100:.2f}%")
        
        if investimento_acoes > 0:
            for i, ticker in enumerate(TICKERS_ACOES):
                percentual_acao = (individuo[i] * PRECOS_ACOES[ticker] / investimento_acoes) * 100
                print(f"{ticker}: {percentual_acao:.2f}% dentro de Ações")
        else:
            print("Nenhum investimento em ações.")
        
        if investimento_fiis > 0:
            for i, ticker in enumerate(TICKERS_FIIS):
                percentual_fii = (individuo[i + len(TICKERS_ACOES)] * PRECOS_FIIS[ticker] / investimento_fiis) * 100
                print(f"{ticker}: {percentual_fii:.2f}% dentro de FIIs")
        else:
            print("Nenhum investimento em FIIs.")
    
    print("\n=== Preços das Ações ===")
    for ticker, preco in PRECOS_ACOES.items():
        print(f"{ticker}: R$ {preco:.2f}")
    
    print("\n=== Preços dos FIIs ===")
    for ticker, preco in PRECOS_FIIS.items():
        print(f"{ticker}: R$ {preco:.2f}")
    print("\n")

def avaliar(individuo):
    """Avalia a qualidade de um indivíduo."""
    investimento_acoes, investimento_fiis, investimento_total = calcular_investimento(individuo)
    
    if investimento_total > ORCAMENTO:
        return (0,)
    
    proporcao_real_acoes = investimento_acoes / investimento_total
    proporcao_real_fiis = investimento_fiis / investimento_total
    erro_proporcoes = abs(PROPORCAO_ACOES - proporcao_real_acoes) + abs(PROPORCAO_FIIS - proporcao_real_fiis)
    
    proporcao_ideal_acao = PROPORCAO_ACOES / len(TICKERS_ACOES)
    proporcao_ideal_fii = PROPORCAO_FIIS / len(TICKERS_FIIS)
    
    erro_distribuicao_acoes = sum(abs(proporcao_ideal_acao - (individuo[i] * PRECOS_ACOES[t] / investimento_acoes)) for i, t in enumerate(TICKERS_ACOES)) if investimento_acoes > 0 else 0
    erro_distribuicao_fiis = sum(abs(proporcao_ideal_fii - (individuo[i + len(TICKERS_ACOES)] * PRECOS_FIIS[t] / investimento_fiis)) for i, t in enumerate(TICKERS_FIIS)) if investimento_fiis > 0 else 0
    erro_distribuicao_total = erro_distribuicao_acoes + erro_distribuicao_fiis
    
    erro_ponderado_total = (PESO_PROPORCOES_GERAIS * erro_proporcoes) + (PESO_DISTRIBUICAO_ATIVOS * erro_distribuicao_total)
    investimento_total_ajustado = investimento_total / 2
    incentivo_fiis = investimento_fiis * 0.1
    
    fitness = investimento_total_ajustado * math.exp(-erro_ponderado_total) + incentivo_fiis
    return (fitness,)

# ================= ALGORITMO GENÉTICO =================
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_int", random.randint, 0, 3)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=len(TICKERS_ACOES) + len(TICKERS_FIIS))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", avaliar)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=3, indpb=TAXA_MUTACAO)
toolbox.register("select", tools.selTournament, tournsize=TAMANHO_TORNEIO)

def algoritmo_genetico(populacao, numero_geracoes):
    """Executa o algoritmo genético."""
    fitness_medio = []
    melhor_fitness = []
    melhores = []
    
    for gen in range(numero_geracoes):
        offspring = toolbox.select(populacao, len(populacao))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < TAXA_MUTACAO:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
        
        populacao[:] = offspring
        melhor_da_geracao = tools.selBest(populacao, k=1)[0]
        melhores.append(melhor_da_geracao)
        
        fitness_medio.append(sum(ind.fitness.values[0] for ind in populacao) / len(populacao))
        melhor_fitness.append(melhor_da_geracao.fitness.values[0])
    
    return tools.selBest(melhores, k=1)[0], fitness_medio, melhor_fitness

# ================= EXECUÇÃO =================
exibir_configuracao_carteira(configuracao="inicial")
pop = toolbox.population(n=TAMANHO_POPULACAO)
melhor_individuo, fitness_medio, melhor_fitness = algoritmo_genetico(pop, NUM_GERACOES)

print("Melhor Solução Global:")
print(f"Geração Encontrada: {melhor_fitness.index(max(melhor_fitness)) + 1}")
print(f"Melhor Fitness Global: {melhor_individuo.fitness.values[0]:.2f}")
print(f"Investimento Total: R${sum(melhor_individuo[i] * PRECOS_ACOES[t] for i, t in enumerate(TICKERS_ACOES)) + sum(melhor_individuo[i + len(TICKERS_ACOES)] * PRECOS_FIIS[t] for i, t in enumerate(TICKERS_FIIS)):.2f}")
print(f"Melhor Indivíduo: {melhor_individuo}")

exibir_configuracao_carteira(configuracao="final", individuo=melhor_individuo)

# Plotar resultados
plt.plot(fitness_medio, label="Fitness Médio")
plt.plot(melhor_fitness, label="Melhor Fitness")
plt.xlabel("Geração")
plt.ylabel("Fitness")
plt.legend()
plt.title("Convergência do Algoritmo Genético")
plt.show()