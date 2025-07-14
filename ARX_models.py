from ARX_optuna import build_arx_matrix_groups, groups, calculate_aic
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import os
from sklearn.metrics import mean_squared_error


pasta_simulacao = 'Simulações/Sim_AC_CC_2'

xt = np.load(os.path.join(pasta_simulacao, 'xt.npy'), allow_pickle=True)
zt = np.load(os.path.join(pasta_simulacao, 'zt.npy'), allow_pickle=True)
t = np.load(os.path.join(pasta_simulacao, 'time.npy'), allow_pickle=True)
ut = np.load(os.path.join(pasta_simulacao, 'ut.npy'), allow_pickle=True)


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

# Lags ótimos encontrados
lags_otimos = [2, 2, 1, 2, 2]  # [pman, p_choke, dP_bcs, F_Booster, F_bcs]

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
plt.figure()
plt.plot(t_plot, y_target, '.', label='Pman exp')
plt.plot(t_plot, pman_sim, label='ARX (lags ótimos)')
plt.ylabel(r'$pman(t)$')
plt.legend()
plt.show()

print(f"AIC(Pman, lags ótimos): {aic}")
print(f"RMSE(Man, lags ótimos): {rmse}")

lags_otimos = [3,4, 4]
groups_qtr = {
    'q_tr': [q_transp],
    'F_Booster': [F_Booster],
    'F_bcs': [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4],
}
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
plt.figure()
plt.plot(t_plot, y_target, '.', label='Pman exp')
plt.plot(t_plot, pman_sim, label='ARX (lags ótimos)')
plt.ylabel(r'$pman(t)$')
plt.legend()
plt.show()

print(f"AIC(qtr, lags ótimos): {aic}")
print(f"RMSE(qtr, lags ótimos): {rmse}")
