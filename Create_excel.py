import numpy as np
import os
import pandas as pd

# === Carrega os dados ===
pasta_simulacao = 'Simulações/Sim_react_1'
xf = np.load(os.path.join(pasta_simulacao, 'xt.npy'), allow_pickle=True)

# Extrai os sinais
p_man = xf[0, :]
q_tr = xf[1, :]

# Reconstrói o vetor de tempo
tfinal = 1500 * 26  # 1500 segundos por simulação, 26 simulações
time = np.linspace(0, tfinal, xf.shape[1])

# Cria DataFrame com os dados numéricos
df = pd.DataFrame({
    'tempo': time,
    'p_man': p_man,
    'q_tr': q_tr
})

# Formata as colunas numéricas para string com vírgula decimal
df['tempo'] = df['tempo'].map(lambda x: f"{x:.6f}".replace('.', ','))
df['p_man'] = df['p_man'].map(lambda x: f"{x:.6f}".replace('.', ','))
df['q_tr'] = df['q_tr'].map(lambda x: f"{x:.6f}".replace('.', ','))

# Salva CSV com separador ponto e vírgula (padrão BR) e sem o índice
df.to_csv('data_csv/p_man_q_tr_completo.csv', sep=',', index=False, decimal=',')

print("Arquivo CSV gerado com formato brasileiro (vírgula decimal)!")

