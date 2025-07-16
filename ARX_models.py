from ARX_optuna import build_arx_matrix_groups, groups, calculate_aic
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import os
from sklearn.metrics import mean_squared_error

# =-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-
# Melhores lags por grupo (Pman):
# pman: 3
# p_choke: 3
# dP_bcs: 3
# F_Booster: 3
# p_topo: 3
# F_bcs: 3
# valve: 3
# Melhor AIC: -7032.7133977844505
# =-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-
# Melhores lags por grupo (q_tr):
# q_tr: 1
# F_Booster: 3
# p_topo: 3
# F_bcs: 3
# valve: 3
# Melhor AIC: -3017.8256166332594
# =-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-

# Exemplo de saída:
# Melhores Parametros qtr: {'n_qtr': 3, 'n_F_Booster': 4, 'n_ptopo': 2, 'n_F_bcs': 1, 'n_valve': 1, 'P_init': 74465.24193300933, 'lam': 0.9999964029009534}
# Melhor AIC qtr: -1721.010405637254
# Melhores Parametros{'n_pman': 1, 'n_pchoke': 2, 'n_dP_bcs': 1, 'n_F_Booster': 2, 'n_ptopo': 1, 'n_F_bcs': 2, 'n_valve': 2, 'P_init': 7643.804616652912, 'lam': 0.9808886573472999}
# Melhor AIC: -2098.511488012563

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
# Exemplo de saída:
# Melhores Parametros{'n_pman': 1, 'n_pchoke': 2, 'n_dP_bcs': 1, 'n_F_Booster': 2, 'n_ptopo': 1, 'n_F_bcs': 2, 'n_valve': 2, 'P_init': 7643.804616652912, 'lam': 0.9808886573472999}
# Melhor AIC: -2098.511488012563

# Melhores Parametros Pman{'n_pman': 2, 'n_pchoke': 2, 'n_dP_bcs': 1, 'n_F_Booster': 2, 'n_ptopo': 2, 'n_F_bcs': 1, 'n_valve': 1, 'P_init': 0.6419392512124129, 'lam': 0.9835874744408079}
# Melhor AIC Pman: -3189.973384558148
# Lags ótimos encontrados
lags_otimos = [2, 2, 1, 2, 2, 1, 1]  # [pman, p_choke, dP_bcs, F_Booster, F_bcs]


def create_init_theta_pman(groups, n_pman, n_pchoke, n_dP_bcs, n_F_Booster, n_ptopo, n_F_bcs, n_valve):
    """
    Cria um vetor de parâmetros iniciais para o modelo ARX de Pman.
    """
    global pman
    lags_otimos = [n_pman, n_pchoke, n_dP_bcs, n_F_Booster, n_ptopo, n_F_bcs, n_valve]
    X, y_target = build_arx_matrix_groups(pman, groups, lags_otimos)
    # Ajustar modelo ARX
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y_target)
    thetas_pman = model.coef_
    return thetas_pman
# Montar matriz de regressores e saída-alvo
X, y_target = build_arx_matrix_groups(pman, groups, lags_otimos)

# Ajustar modelo ARX
model = LinearRegression(fit_intercept=False)
model.fit(X, y_target)
pman_sim = model.predict(X)

# Calcular AIC
mse = np.mean((y_target - pman_sim)**2)
print(mse)
aic = calculate_aic(len(y_target), mse, X.shape[1])
rmse = np.sqrt(mse)
print(rmse)
# Plotar resultados
t_plot = t[-len(y_target):]
# plt.figure(figsize=(19, 6), dpi= 300)
# plt.plot(t_plot, desnormalize(pman_sim, pman_media, pman_std), label='ARX',linewidth=3)
# plt.plot(t_plot, desnormalize(y_target, pman_media, pman_std), '.', label='CASADI', linewidth=3)
# plt.ylabel(r'Pressão Manifold / Bar', fontsize=16)
# plt.xlabel(r'Tempo / s', fontsize=16)
# plt.legend(fontsize=16)
# plt.show()
#
# print(f"AIC(Pman, lags ótimos): {aic}")
# print(f"RMSE(Man, lags ótimos): {rmse}")

# Após ajustar o modelo
thetas_pman = model.coef_
np.save('data/thetas_pman_arx.npy', thetas_pman)

# Exemplo de saída:
# Melhores Parametros qtr: {'n_qtr': 3, 'n_F_Booster': 4, 'n_ptopo': 2, 'n_F_bcs': 1, 'n_valve': 1, 'P_init': 74465.24193300933, 'lam': 0.9999964029009534}
# Melhor AIC qtr: -1721.010405637254

# Melhores Parametros qtr: {'n_qtr': 1, 'n_F_Booster': 3, 'n_ptopo': 1, 'n_F_bcs': 2, 'n_valve': 2, 'P_init': 1.017536246808685, 'lam': 0.8564930170941152}
# Melhor AIC qtr: -2387.298224531064

# Melhores Parametros qtr: {'n_qtr': 1, 'n_F_Booster': 3, 'n_ptopo': 1, 'n_F_bcs': 2, 'n_valve': 2, 'P_init': 0.18152246007140704, 'lam': 0.8201136110428227}
# Melhor AIC qtr: -2405.193903225024
lags_otimos = [1, 3, 1 , 2, 2]
groups_qtr = {
    'q_tr': [q_transp],
    'F_Booster': [F_Booster],
    'p_topo': [P_topo],
    'F_bcs': [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4],
    'valve': [valve_1, valve_2, valve_3, valve_4],
}

def create_init_theta_qtr(groups_qtr, n_qtr, n_F_Booster, n_ptopo, n_F_bcs, n_valve):
    """
    Cria um vetor de parâmetros iniciais para o modelo ARX de Pman.
    """
    global q_transp
    lags_otimos_qtr = [n_qtr, n_F_Booster, n_ptopo, n_F_bcs, n_valve]
    X, y_target = build_arx_matrix_groups(q_transp, groups_qtr, lags_otimos_qtr)
    # Ajustar modelo ARX
    model = LinearRegression(fit_intercept=False)
    model.fit(X, y_target)
    thetas_qtr = model.coef_
    return thetas_qtr

# Montar matriz de regressores e saída-alvo
X, y_target = build_arx_matrix_groups(q_transp, groups_qtr, lags_otimos)

# Ajustar modelo ARX
model = LinearRegression(fit_intercept=False)
model.fit(X, y_target)
start = time.time()
pman_sim = model.predict(X)
end = time.time()
# Calcular AIC
mse = np.mean((y_target - pman_sim)**2)
rmse = np.sqrt(mse)
aic = calculate_aic(len(y_target), mse, X.shape[1])
t_plot = t[-len(y_target):]

plt.figure(figsize=(19, 6), dpi= 300)
plt.plot(t_plot, desnormalize(pman_sim, q_transp_media, q_transp_std), label='ARX',linewidth=3)
plt.plot(t_plot, desnormalize(y_target, q_transp_media, q_transp_std), '.', label='CASADI',linewidth=3)
plt.ylabel(r'Vazão Manifold / $m^3 \cdot s^{-1}$', fontsize=16)
plt.xlabel(r'Tempo / s', fontsize=16)
plt.legend(fontsize=16)
plt.show()

print(f"AIC(qtr, lags ótimos): {aic}")
print(f"RMSE(qtr, lags ótimos): {rmse}")

thetas_qtr = model.coef_
np.save('data/thetas_qtr_arx.npy', thetas_qtr)
