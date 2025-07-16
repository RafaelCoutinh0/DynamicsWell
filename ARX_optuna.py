import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import optuna
from sklearn.metrics import mean_squared_error

def calculate_aic(n, mse, k):
    """Calcula o Critério de Informação de Akaike (AIC)."""
    return n * np.log(mse) + 2 * k
pasta_simulacao = 'Simulações/Sim_AC_CC_2'
xt = np.load(os.path.join(pasta_simulacao, 'xt.npy'), allow_pickle=True)
zt = np.load(os.path.join(pasta_simulacao, 'zt.npy'), allow_pickle=True)
t = np.load(os.path.join(pasta_simulacao, 'time.npy'), allow_pickle=True)
ut = np.load(os.path.join(pasta_simulacao, 'ut.npy'), allow_pickle=True)

pasta_simulacao = 'Simulações/Sim_ARX1'
xt1 = np.load(os.path.join(pasta_simulacao, 'xt.npy'), allow_pickle=True)
zt2 = np.load(os.path.join(pasta_simulacao, 'zt.npy'), allow_pickle=True)
t2= np.load(os.path.join(pasta_simulacao, 'time.npy'), allow_pickle=True)
ut2 = np.load(os.path.join(pasta_simulacao, 'ut.npy'), allow_pickle=True)

for i in range(len(t)):
    t2[i] += t[-1]

xt = np.hstack((xt, xt1[:, :150]))
zt = np.hstack((zt, zt2[:, :150]))
ut = np.hstack((ut, ut2[:, :150]))
t = np.hstack((t, t2[:150]))

t = t[1:-1]  # Excluir o primeiro valor de tempo para manter a consistência com os regressores
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



# Pman ( Pman, p_choke_1, p_choke_2,p_choke_3,p_choke_4, p_intake_1,p_intake_2,p_intake_3,p_intake_4, dP_bcs_1 ,dP_bcs_2 ,dP_bcs_3 ,dP_bcs_4 , F_Booster, F_bcs_1,F_bcs_2,F_bcs_3,F_bcs_4, valve_1, valve_2, valve_3, valve_4)
Pman_exp = pman[1:-1]  # Excluir o primeiro e o último valor para manter a consistência com os regressores

# Lista de variáveis de entrada (exceto a saída pman)
inputs = [
    p_choke_1, p_choke_2, p_choke_3, p_choke_4,
    dP_bcs_1, dP_bcs_2, dP_bcs_3, dP_bcs_4,
    F_Booster, F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4
]
# Variáveis já carregadas: pman, inputs (lista de arrays)
all_vars = [pman] + inputs

var_names = [ 'pman', 'p_choke_1', 'p_choke_2', 'p_choke_3', 'p_choke_4',
    'dP_bcs_1', 'dP_bcs_2', 'dP_bcs_3', 'dP_bcs_4',
    'F_Booster', 'F_bcs_1', 'F_bcs_2', 'F_bcs_3', 'F_bcs_4']
# Definição dos grupos
groups = {
    'pman': [pman],
    'p_choke': [p_choke_1, p_choke_2, p_choke_3, p_choke_4],
    'dP_bcs': [dP_bcs_1, dP_bcs_2, dP_bcs_3, dP_bcs_4],
    'F_Booster': [F_Booster],
    'p_topo': [P_topo],
    'F_bcs': [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4],
    'valve': [valve_1, valve_2, valve_3, valve_4],
}

def build_arx_matrix_groups(y, groups, lags):
    # O primeiro grupo é sempre a saída (usar apenas atrasos passados)
    min_lag = max([lag for lag in lags])  # lag para saída é o número de atrasos, para entradas é lag-1
    N = len(y) - min_lag
    X = []
    for i in range(N):
        row = []
        for idx, (group_vars, lag) in enumerate(zip(groups.values(), lags)):
            if lag > 0:
                for var in group_vars:
                    if idx == 0:
                        # Saída: usar apenas atrasos passados (nunca o valor atual)
                        for l in range(1, lag + 1):
                            row.append(var[i + min_lag - l])
                    else:
                        # Entradas: usar valor atual e atrasos
                        for l in range(lag):
                            row.append(var[i + min_lag - l])
        X.append(row)
    X = np.array(X)
    y_target = y[min_lag:]
    return X, y_target

def objective(trial):
    lags = [trial.suggest_int(f'lag_{name}', 0, 3) for name in groups.keys()]
    X, y_target = build_arx_matrix_groups(pman, groups, lags)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y_target)
    pred = model.predict(X)
    mse = np.mean((y_target - pred)**2)
    aic = calculate_aic(len(y_target), mse, X.shape[1])
    return aic

# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=2000)



# q_tr (q_transp, F_Booster, F_bcs, valve)


# Defina os grupos para q_tr
groups_qtr = {
    'q_tr': [q_transp],
    'F_Booster': [F_Booster],
    'p_topo': [P_topo],
    'F_bcs': [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4],
    'valve': [valve_1, valve_2, valve_3, valve_4],
}


def objective_qtr(trial):
    lags = [trial.suggest_int(f'lag_{name}', 0, 3) for name in groups_qtr.keys()]
    if lags == [0,0,0,0,0]:
        return 100000
    X, y_target = build_arx_matrix_groups(q_transp, groups_qtr, lags)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y_target)
    pred = model.predict(X)
    mse = np.mean((y_target - pred)**2)
    aic = calculate_aic(len(y_target), mse, X.shape[1])
    return aic


# study_qtr = optuna.create_study(direction='minimize')
# study_qtr.optimize(objective_qtr, n_trials=2000)
# print("\n")
# print("=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-")
# print("Melhores lags por grupo (Pman):")
# for name in groups.keys():
#     print(f"{name}: {study.best_trial.params[f'lag_{name}']}")
# print("Melhor AIC:", study.best_value)
# print("=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-")
# print("Melhores lags por grupo (q_tr):")
# for name in groups_qtr.keys():
#     print(f"{name}: {study_qtr.best_trial.params[f'lag_{name}']}")
# print("Melhor AIC:", study_qtr.best_value)
# print("=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-")
