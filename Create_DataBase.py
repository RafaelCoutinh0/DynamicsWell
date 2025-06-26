import os
import numpy as np
from initialization_oil_production_basic import plotar_graficos

# === NOME DAS VARIÁVEIS ===
xf_names = [
    'P_man', 'q_transp',
    'P_fbhp_1', 'P_choke_1', 'q_mean_1',
    'P_fbhp_2', 'P_choke_2', 'q_mean_2',
    'P_fbhp_3', 'P_choke_3', 'q_mean_3',
    'P_fbhp_4', 'P_choke_4', 'q_mean_4'
]
zf_names = [
    'P_intake_1', 'dP_bcs_1',
    'P_intake_2', 'dP_bcs_2',
    'P_intake_3', 'dP_bcs_3',
    'P_intake_4', 'dP_bcs_4'
]
# === GERAR DADOS DE SIMULAÇÃO ===
print("Executando simulação para obter dados...")
n_pert = 10  # número de perturbações (ajustável)
i = 1
while os.path.exists(f'Simulações/Sim_{i}'):
    i += 1

# Cria a nova pasta
pasta_simulacao = f'Simulações/Sim_{i}'
os.makedirs(pasta_simulacao)

# Executa a simulação
Lista_xf_reshaped, Lista_zf_reshaped, Inputs = plotar_graficos(n_pert)

# Salva os arquivos na pasta
np.save(os.path.join(pasta_simulacao, 'xf.npy'), np.array(Lista_xf_reshaped))
np.save(os.path.join(pasta_simulacao, 'zf.npy'), np.array(Lista_zf_reshaped))
np.save(os.path.join(pasta_simulacao, 'Entradas.npy'), np.array(Inputs))