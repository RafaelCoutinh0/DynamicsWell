import numpy as np
import os
import matplotlib.pyplot as plt

# === Carrega os dados ===
pasta_simulacao = 'Simulações/Sim_react_1'
xf = np.load(os.path.join(pasta_simulacao, 'xf.npy'), allow_pickle=True)

# Extrai sinais
p_man = xf[0, :]
q_tr = xf[1, :]

# Cria o vetor de tempo real usado na simulação
tfinal = 1500 * 26 # 1500 segundos por simulação, 26 simulações
time_grid_total = np.linspace(0, tfinal, 100 * 26)  # 100 pontos por simulação, 26 simulações

# Parâmetros de separação
janela = 3000
passo = 4500
max_ini = int(tfinal) - janela

# Dicionários para armazenar os blocos
p_man_blocos = {}
q_tr_blocos = {}

for i, ini in enumerate(range(0, max_ini + 1, passo)):
    if i >= 9:
        break

    fim = ini + janela
    idxs = np.where((time_grid_total >= ini) & (time_grid_total <= fim))[0]
    if len(idxs) == 0:
        continue

    p_man_blocos[f'p_man_pert_{i+1}'] = p_man[idxs]
    q_tr_blocos[f'q_tr_pert_{i+1}'] = q_tr[idxs]

# Plot do bloco 3
for i in range(1, 10):
    idx = i
    p = p_man_blocos[f'p_man_pert_{idx}']
    q = q_tr_blocos[f'q_tr_pert_{idx}']
    tempo = np.linspace(0, janela, len(p))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(tempo, p, color='blue')
    ax1.set_ylabel('p_man (bar)')
    ax1.set_title(f'P_man_pert_{idx}')
    ax1.grid(True)

    ax2.plot(tempo, q, color='green')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('q_tr (m³/dia)')
    ax2.set_title(f'q_tr_pert_{idx}')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# === Cria o gráfico ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Plot da pressão no manifold
ax1.plot(time_grid_total, p_man, color='blue')
ax1.set_ylabel('p_man (bar)')
ax1.set_title('Pressão no Manifold (p_man)')
ax1.grid(True)

# Plot da vazão de transporte
ax2.plot(time_grid_total, q_tr, color='green')
ax2.set_xlabel('Tempo (s)')
ax2.set_ylabel('q_tr (m³/dia)')
ax2.set_title('Vazão de Transporte (q_tr)')
ax2.grid(True)

plt.tight_layout()
plt.show()