import matplotlib.pyplot as plt
import numpy as np
import os

pasta_simulacao = f'Simulações/Sim_1'
# Abrir os arquivos
xf = np.load(os.path.join(pasta_simulacao, 'xf.npy'), allow_pickle=True)
zf = np.load(os.path.join(pasta_simulacao, 'zf.npy'), allow_pickle=True)
entradas = np.load(os.path.join(pasta_simulacao, 'Entradas.npy'), allow_pickle=True)

