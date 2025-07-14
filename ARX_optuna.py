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
    'F_bcs': [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4]
}

def build_arx_matrix_groups(y, groups, lags):
    min_lag = max(lags)
    N = len(y) - min_lag
    X = []
    for i in range(N):
        row = []
        for group_vars, lag in zip(groups.values(), lags):
            if lag > 0:  # Apenas incluir variáveis com lag positivo
                for var in group_vars:
                    for l in range(1, lag + 1):  # Ignorar lag zero
                        row.append(var[i + min_lag - l])
        X.append(row)
    X = np.array(X)
    y_target = y[min_lag:]
    return X, y_target

def objective(trial):
    lags = [trial.suggest_int(f'lag_{name}', 0, 5) for name in groups.keys()]
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
    'F_bcs': [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4],
    'valve': [valve_1, valve_2, valve_3, valve_4],

}


def objective_qtr(trial):
    lags = [trial.suggest_int(f'lag_{name}', 0, 5) for name in groups_qtr.keys()]
    if lags == [0,0,0,0]:
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
