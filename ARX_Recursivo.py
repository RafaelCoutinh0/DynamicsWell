import numpy as np
import matplotlib.pyplot as plt
import os
from ARX_optuna import calculate_aic
from ARX_models import create_init_theta_pman, create_init_theta_qtr, groups_qtr

pasta_simulacao = 'Simulações/Sim_AC_CC_2'
x0 = np.load(os.path.join(pasta_simulacao, 'xt.npy'), allow_pickle=True)
z0 = np.load(os.path.join(pasta_simulacao, 'zt.npy'), allow_pickle=True)
t0 = np.load(os.path.join(pasta_simulacao, 'time.npy'), allow_pickle=True)
u0 = np.load(os.path.join(pasta_simulacao, 'ut.npy'), allow_pickle=True)

pasta_simulacao = 'Simulações/Sim_ARX1'
xt = np.load(os.path.join(pasta_simulacao, 'xt.npy'), allow_pickle=True)
zt = np.load(os.path.join(pasta_simulacao, 'zt.npy'), allow_pickle=True)
t = np.load(os.path.join(pasta_simulacao, 'time.npy'), allow_pickle=True)
ut = np.load(os.path.join(pasta_simulacao, 'ut.npy'), allow_pickle=True)

for i in range(len(t)):
    t[i] += t0[-1]

x0 = np.hstack((x0, xt[:, :150]))
z0 = np.hstack((z0, zt[:, :150]))
u0 = np.hstack((u0, ut[:, :150]))
t0 = np.hstack((t0, t[:150]))

xt = xt[:, 150:]
zt = zt[:, 150:]
ut = ut[:, 150:]
t = t[150:]

xt_tot = np.hstack((x0, xt))
zt_tot = np.hstack((z0, zt))
ut_tot = np.hstack((u0, ut))
t_tot = np.hstack((t0, t))

# plt.figure(figsize=(19, 6), dpi= 300)
# plt.plot(t_tot, xt_tot[0], color='red', label='Treinamento dos $\\theta_s$', linewidth=2)
# plt.plot(t, xt[0], color='Blue', label='$\\theta_s$ Recursivo', linewidth=2)
# plt.axvline(x=12000, color='r', linestyle='--', label='t = 12000 s')
# # plt.title('Pressão Manifold / Bar')
# plt.xlabel('Tempo / s', fontsize=16)
# plt.ylabel('Pressão Manifold / Bar', fontsize=16)
# plt.legend(fontsize=16)
# plt.tick_params(axis='both', labelsize=16)
# plt.show()
thetas_qtr = np.load('data/thetas_qtr_arx.npy')
thetas_pman = np.load('data/thetas_pman_arx.npy')

# xt
pman = xt[0]
q_transp = xt[1]
p_fbhp_1 = xt[2]
p_choke_1 = xt[3]
q_mean_1 = xt[4]
p_fbhp_2 = xt[5]
p_choke_2 = xt[6]
q_mean_2 = xt[7]
p_fbhp_3 = xt[8]
p_choke_3 = xt[9]
q_mean_3 = xt[10]
p_fbhp_4 = xt[11]
p_choke_4 = xt[12]
q_mean_4 = xt[13]
# zt
p_intake_1 = zt[0]
dP_bcs_1 = zt[1]
p_intake_2 = zt[2]
dP_bcs_2 = zt[3]
p_intake_3 = zt[4]
dP_bcs_3 = zt[5]
p_intake_4 = zt[6]
dP_bcs_4 = zt[7]
# ut
F_Booster = ut[0]
P_topo = ut[1]
F_bcs_1 = ut[2]
valve_1 = ut[3]
F_bcs_2 = ut[4]
valve_2 = ut[5]
F_bcs_3 = ut[6]
valve_3 = ut[7]
F_bcs_4 = ut[8]
valve_4 = ut[9]

def normalizar(x):
    media = np.mean(x)
    desvio = np.std(x)
    return (x - media) / desvio, media, desvio

def desnormalize(x_norm, media, desvio):
    return x_norm * desvio + media
# Normalizando todas as variáveis
pman, pman_media, pman_std = normalizar(pman)
q_transp, q_transp_media, q_transp_std = normalizar(q_transp)
p_fbhp_1, p_fbhp_1_media, p_fbhp_1_std = normalizar(p_fbhp_1)
p_choke_1, p_choke_1_media, p_choke_1_std = normalizar(p_choke_1)
q_mean_1, q_mean_1_media, q_mean_1_std = normalizar(q_mean_1)
p_fbhp_2, p_fbhp_2_media, p_fbhp_2_std = normalizar(p_fbhp_2)
p_choke_2, p_choke_2_media, p_choke_2_std = normalizar(p_choke_2)
q_mean_2, q_mean_2_media, q_mean_2_std = normalizar(q_mean_2)
p_fbhp_3, p_fbhp_3_media, p_fbhp_3_std = normalizar(p_fbhp_3)
p_choke_3, p_choke_3_media, p_choke_3_std = normalizar(p_choke_3)
q_mean_3, q_mean_3_media, q_mean_3_std = normalizar(q_mean_3)
p_fbhp_4, p_fbhp_4_media, p_fbhp_4_std = normalizar(p_fbhp_4)
p_choke_4, p_choke_4_media, p_choke_4_std = normalizar(p_choke_4)
q_mean_4, q_mean_4_media, q_mean_4_std = normalizar(q_mean_4)
p_intake_1, p_intake_1_media, p_intake_1_std = normalizar(p_intake_1)
dP_bcs_1, dP_bcs_1_media, dP_bcs_1_std = normalizar(dP_bcs_1)
p_intake_2, p_intake_2_media, p_intake_2_std = normalizar(p_intake_2)
dP_bcs_2, dP_bcs_2_media, dP_bcs_2_std = normalizar(dP_bcs_2)
p_intake_3, p_intake_3_media, p_intake_3_std = normalizar(p_intake_3)
dP_bcs_3, dP_bcs_3_media, dP_bcs_3_std = normalizar(dP_bcs_3)
p_intake_4, p_intake_4_media, p_intake_4_std = normalizar(p_intake_4)
dP_bcs_4, dP_bcs_4_media, dP_bcs_4_std = normalizar(dP_bcs_4)
F_Booster, F_Booster_media, F_Booster_std = normalizar(F_Booster)
P_topo, P_topo_media, P_topo_std = normalizar(P_topo)
F_bcs_1, F_bcs_1_media, F_bcs_1_std = normalizar(F_bcs_1)
valve_1, valve_1_media, valve_1_std = normalizar(valve_1)
F_bcs_2, F_bcs_2_media, F_bcs_2_std = normalizar(F_bcs_2)
valve_2, valve_2_media, valve_2_std = normalizar(valve_2)
F_bcs_3, F_bcs_3_media, F_bcs_3_std = normalizar(F_bcs_3)
valve_3, valve_3_media, valve_3_std = normalizar(valve_3)
F_bcs_4, F_bcs_4_media, F_bcs_4_std = normalizar(F_bcs_4)
valve_4, valve_4_media, valve_4_std = normalizar(valve_4)

# Exemplo de saída:
# Melhores Parametros{'n_pman': 1, 'n_pchoke': 2, 'n_dP_bcs': 1, 'n_F_Booster': 2, 'n_ptopo': 1, 'n_F_bcs': 2, 'n_valve': 2, 'P_init': 7643.804616652912, 'lam': 0.9808886573472999}
# Melhor AIC: -2098.511488012563

# Melhores Parametros Pman{'n_pman': 2, 'n_pchoke': 2, 'n_dP_bcs': 1, 'n_F_Booster': 2, 'n_ptopo': 2, 'n_F_bcs': 1, 'n_valve': 1, 'P_init': 0.6419392512124129, 'lam': 0.9835874744408079}
# Melhor AIC Pman: -3189.973384558148

n_pman, n_pchoke, n_dP_bcs, n_F_Booster,n_ptopo, n_F_bcs, n_valve = 2, 2, 1, 2, 2, 1, 1
d = max(n_pman, n_pchoke, n_dP_bcs, n_F_Booster, n_ptopo, n_F_bcs, n_valve)  # maior lag
n_params = n_pman + (n_pchoke * 4) + (n_dP_bcs * 4) + n_F_Booster + n_ptopo + (n_F_bcs * 4) + (n_valve * 4)

P = np.eye(n_params) * 0.64          # covariância inicial grande
lam = 0.98                      # fator de esquecimento
R   = 0                        # ruído de medição

pman_hat = np.zeros_like(pman)
pman_hat[:d] = pman[:d]  # inicializa com valores reais para os primeiros d
theta = thetas_pman.copy()
theta_hist = np.zeros((len(pman) - d, n_params))
for idx, k in enumerate(range(d, len(pman))):
    phi = []
    # Saída: atrasos previstos
    for i in range(1, n_pman + 1):
        phi.append(pman[k - i])
    # Entradas: valor atual e atrasos
    for p_choke in [p_choke_1, p_choke_2, p_choke_3, p_choke_4]:
        for i in range(n_pchoke):
            phi.append(p_choke[k - i])
    for dP_bcs in [dP_bcs_1, dP_bcs_2, dP_bcs_3, dP_bcs_4]:
        for i in range(n_dP_bcs):
            phi.append(dP_bcs[k - i])
    for i in range(n_F_Booster):
        phi.append(F_Booster[k - i])
    for i in range(n_ptopo):
        phi.append(P_topo[k - i])
    for f_bcs in [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4]:
        for i in range(n_F_bcs):
            phi.append(f_bcs[k - i])
    for valve in [valve_1, valve_2, valve_3, valve_4]:
        for i in range(n_valve):
            phi.append(valve[k - i])
    phi = np.array(phi)

    # Previsão e erro
    pman_hat[k] = phi @ theta
    err = pman[k] - pman_hat[k]

    # Ganho e atualização
    gain = (P @ phi) / (lam + phi @ P @ phi + R)
    theta += gain * err
    P = (P - np.outer(gain, phi) @ P) / lam

    theta_hist[idx] = theta
mse_pman = np.mean((pman[d:] - pman_hat[d:])**2)
# rmse = np.sqrt(mse)
# print(f'RMSE(Pressão Manifold): {rmse}')
aic = calculate_aic(len(pman) - d, mse_pman, n_params)
rmse_pman = np.sqrt(mse_pman)
# print(f'AIC(Pressão Manifold): {aic}')
# print(f'RMSE (Pressão Manifold): {rmse_pman}')
# plot do pman estimado vs pman experimental

# plt.figure(figsize=(16, 5))
# plt.plot(t,desnormalize(pman_hat, pman_media, pman_std), label="ARX RECURSIVO", linewidth= 3)
# plt.plot(t,desnormalize(pman, pman_media, pman_std), label="CASADI", marker='.', linewidth= 1)
# plt.xlabel("Tempo / s")
# plt.ylabel("Pressão Manifold / Bar")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# # Plot da evolução dos parâmetros
# plt.figure(figsize=(14, 6))
# for i in range(n_params):
#     plt.plot(t[d:],theta_hist[:, i], label=f'theta[{i}]')
# plt.xlabel('Tempo / s')
# plt.ylabel('Valor dos parâmetros')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Exemplo de saída:
# Melhores Parametros qtr: {'n_qtr': 3, 'n_F_Booster': 4, 'n_ptopo': 2, 'n_F_bcs': 1, 'n_valve': 1, 'P_init': 74465.24193300933, 'lam': 0.9999964029009534}
# Melhor AIC qtr: -1721.010405637254

# Melhores Parametros qtr: {'n_qtr': 1, 'n_F_Booster': 3, 'n_ptopo': 1, 'n_F_bcs': 2, 'n_valve': 2, 'P_init': 1.017536246808685, 'lam': 0.8564930170941152}
# Melhor AIC qtr: -2387.298224531064

# Melhores Parametros qtr: {'n_qtr': 1, 'n_F_Booster': 3, 'n_ptopo': 1, 'n_F_bcs': 2, 'n_valve': 2, 'P_init': 0.18152246007140704, 'lam': 0.8201136110428227}
# Melhor AIC qtr: -2405.193903225024
n_qtr, n_F_Booster, n_ptopo, n_F_bcs, n_valve = 1, 3, 1 , 2, 2
d = max(n_qtr,n_F_Booster, n_F_bcs, n_valve, n_ptopo)  # maior lag
n_params = n_qtr + n_F_Booster + (n_F_bcs * 4) + (n_valve*4) + n_ptopo

P = np.eye(n_params) * 0.18152246007140704  # covariância inicial grande
lam = 0.8201136110428227
R = 0
qtr_hat = np.zeros_like(q_transp)
qtr_hat[:d] = q_transp[:d]
theta = thetas_qtr.copy()

theta_hist = np.zeros((len(q_transp) - d, n_params))
for idx, k in enumerate(range(d, len(q_transp))):
    phi = []
    # Saída: atrasos previstos
    for i in range(1, n_qtr + 1):
        phi.append(q_transp[k - i])
    # Entradas: valor atual e atrasos
    for i in range(n_F_Booster):
        phi.append(F_Booster[k - i])
    for i in range(n_ptopo):
        phi.append(P_topo[k - i])
    for f_bcs in [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4]:
        for i in range(n_F_bcs):
            phi.append(f_bcs[k - i])
    for valve in [valve_1, valve_2, valve_3, valve_4]:
        for i in range(n_valve):
            phi.append(valve[k - i])
    phi = np.array(phi)
    # Previsão e erro
    qtr_hat[k] = phi @ theta
    err = q_transp[k] - qtr_hat[k]

    # Ganho e atualização
    gain = (P @ phi) / (lam + phi @ P @ phi + R)
    theta += gain * err
    P = (P - np.outer(gain, phi) @ P) / lam
    theta_hist[idx] = theta

mse_qtr = np.mean((q_transp[d:] - qtr_hat[d:])**2)
rmse_qtr = np.sqrt(mse_qtr)
aic = calculate_aic(len(q_transp) - d, mse_qtr, n_params)
# print(f'AIC(q_transp): {aic}')
# print(f'RMSE (q_transp): {rmse_qtr}')
#
# plt.figure(figsize=(16, 5))
# plt.plot(t, desnormalize(qtr_hat, q_transp_media, q_transp_std), label="ARX RECURSIVO", linewidth=3)
# plt.plot(t, desnormalize(q_transp, q_transp_media, q_transp_std), label="CASADI)", linewidth=3, marker='.')
# plt.xlabel("Tempo / s)")
# plt.ylabel(r"Vazão Manifold / $m^3 \cdot s^{-1}$")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
#
# plt.figure(figsize=(14, 6))
# for i in range(n_params):
#     plt.plot(t[d:], theta_hist[:, i], label=f'theta[{i}]')
# plt.xlabel('Tempo / s')
# plt.ylabel('Valor dos parâmetros')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()