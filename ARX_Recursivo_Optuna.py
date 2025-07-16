import numpy as np
import optuna
import matplotlib.pyplot as plt
import os
from ARX_models import create_init_theta_pman, create_init_theta_qtr
from ARX_optuna import calculate_aic, groups_qtr, groups

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

# Pman lags_otimos = [1, 1, 2, 1, 2]  # [pman, p_choke, dP_bcs, F_Booster, F_bcs]
n_pman, n_pchoke, n_dP_bcs, n_F_Booster,n_ptopo, n_F_bcs, n_valve = 3, 3, 3, 3, 3, 3, 3
d = 3
n_params = n_pman + (n_pchoke * 4) + (n_dP_bcs * 4) + n_F_Booster + n_ptopo + (n_F_bcs * 4) + (n_valve * 4)

def objective(trial):
    # Sugere lags ótimos
    n_pman = trial.suggest_int('n_pman', 1, 3)
    n_pchoke = trial.suggest_int('n_pchoke', 1, 3)
    n_dP_bcs = trial.suggest_int('n_dP_bcs', 1, 3)
    n_F_Booster = trial.suggest_int('n_F_Booster', 1, 3)
    n_ptopo = trial.suggest_int('n_ptopo', 1, 3)
    n_F_bcs = trial.suggest_int('n_F_bcs', 1, 3)
    n_valve = trial.suggest_int('n_valve', 1, 3)
    d = max(n_pman, n_pchoke, n_dP_bcs, n_F_Booster, n_ptopo, n_F_bcs, n_valve)
    n_params = n_pman + (n_pchoke * 4) + (n_dP_bcs * 4) + n_F_Booster + n_ptopo + (n_F_bcs * 4) + (n_valve * 4)
    # Sugere valores para os hiperparâmetros
    P_init = trial.suggest_float('P_init', 1e-2, 1e5, log=True)
    lam = trial.suggest_float('lam', 0.3, 1, log=True)
    R = 0

    # Inicialização
    P = np.eye(n_params) * P_init
    theta = create_init_theta_pman(groups, n_pman, n_pchoke, n_dP_bcs, n_F_Booster, n_ptopo, n_F_bcs, n_valve)
    pman_hat = np.zeros_like(pman)
    pman_hat[:d] = pman[:d]

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
        gain = (P @ phi) / (lam + phi @ P @ phi+ R)
        theta += gain * err
        P = (P - np.outer(gain, phi) @ P) / lam

    mse = np.mean((pman[d:] - pman_hat[d:])**2)
    aic = calculate_aic(len(pman) - d, mse, n_params)
    return aic

# Vazão Manifold lags_otimos = [3, 4, 4]  # [q_transp, F_Booster, F_bcs]
# Parâmetros do modelo qtr
n_qtr, n_F_Booster, n_ptopo, n_F_bcs, n_valve = 1, 3, 3, 3, 3
d = 3  # maior lag
n_params = n_qtr + n_F_Booster + (n_F_bcs * 4) + (n_valve*4) + n_ptopo

def objective_qtr(trial):
    n_qtr = trial.suggest_int('n_qtr', 1, 5)
    n_F_Booster = trial.suggest_int('n_F_Booster', 1, 5)
    n_ptopo = trial.suggest_int('n_ptopo', 1, 5)
    n_F_bcs = trial.suggest_int('n_F_bcs', 1, 5)
    n_valve = trial.suggest_int('n_valve', 1, 5)
    d = max(n_qtr, n_F_Booster, n_ptopo, n_F_bcs, n_valve)
    n_params = n_qtr + n_F_Booster + (n_F_bcs * 4) + (n_valve*4) + n_ptopo

    P_init = trial.suggest_float('P_init', 1e-2, 1e5, log=True)
    lam = trial.suggest_float('lam', 0.3, 1, log=True)
    R = 0

    P = np.eye(n_params) * P_init
    theta = create_init_theta_qtr(groups_qtr, n_qtr, n_F_Booster, n_ptopo, n_F_bcs, n_valve)
    qtr_hat = np.zeros_like(q_transp)
    qtr_hat[:d] = q_transp[:d]

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

    mse = np.mean((q_transp[d:] - qtr_hat[d:])**2)
    aic = calculate_aic(len(q_transp) - d, mse, n_params)
    return aic

study_qtr = optuna.create_study(direction='minimize')
study_qtr.optimize(objective_qtr, n_trials=4000)
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=2000)
# print(f"Melhores Parametros Pman{study.best_params}")
# print(f"Melhor AIC Pman: {study.best_value}")
print(f"Melhores Parametros qtr: {study_qtr.best_params}")
print(f"Melhor AIC qtr: {study_qtr.best_value}")
# Exemplo de saída:
# Melhores Parametros qtr: {'n_qtr': 3, 'n_F_Booster': 4, 'n_ptopo': 2, 'n_F_bcs': 1, 'n_valve': 1, 'P_init': 74465.24193300933, 'lam': 0.9999964029009534}
# Melhor AIC qtr: -1721.010405637254
# Melhores Parametros{'n_pman': 1, 'n_pchoke': 2, 'n_dP_bcs': 1, 'n_F_Booster': 2, 'n_ptopo': 1, 'n_F_bcs': 2, 'n_valve': 2, 'P_init': 7643.804616652912, 'lam': 0.9808886573472999}
# Melhor AIC: -2098.511488012563