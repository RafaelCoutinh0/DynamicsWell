import time
import numpy as np
from ARX_Recursivo import (
    pman, pman_media, pman_std, q_transp, q_transp_media, q_transp_std,
    thetas_pman, thetas_qtr, F_Booster, F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4,
    P_topo, valve_1, valve_2, valve_3, valve_4, t
)
from initialization_oil_production_basic import Sim_dynamics

# --- ARX Recursivo para Pressão Manifold ---
start_arx = time.time()
# Use os melhores hiperparâmetros encontrados
n_pman, n_pchoke, n_dP_bcs, n_F_Booster, n_ptopo, n_F_bcs, n_valve = 2, 2, 1, 2, 2, 1, 1
d = max(n_pman, n_pchoke, n_dP_bcs, n_F_Booster, n_ptopo, n_F_bcs, n_valve)
n_params = n_pman + (n_pchoke * 4) + (n_dP_bcs * 4) + n_F_Booster + n_ptopo + (n_F_bcs * 4) + (n_valve * 4)
P = np.eye(n_params) * 0.64
lam = 0.98
R = 0
pman_hat = np.zeros_like(pman)
pman_hat[:d] = pman[:d]
theta = thetas_pman.copy()
for k in range(d, len(pman)):
    phi = []
    for i in range(1, n_pman + 1):
        phi.append(pman[k - i])
    for p_choke in [0, 0, 0, 0]:  # Substitua por p_choke_1~4 se necessário
        for i in range(n_pchoke):
            phi.append(p_choke)
    for dP_bcs in [0, 0, 0, 0]:  # Substitua por dP_bcs_1~4 se necessário
        for i in range(n_dP_bcs):
            phi.append(dP_bcs)
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
    pman_hat[k] = phi @ theta
    err = pman[k] - pman_hat[k]
    gain = (P @ phi) / (lam + phi @ P @ phi + R)
    theta += gain * err
    P = (P - np.outer(gain, phi) @ P) / lam
end_arx = time.time()
tempo_arx = end_arx - start_arx

# --- ARX Recursivo para qtr ---
start_arx = time.time()
n_qtr, n_F_Booster, n_ptopo, n_F_bcs, n_valve = 1, 3, 1, 2, 2
d = max(n_qtr, n_F_Booster, n_ptopo, n_F_bcs, n_valve)
n_params = n_qtr + n_F_Booster + n_ptopo + (n_F_bcs * 4) + (n_valve * 4)
P = np.eye(n_params) * 0.18
lam = 0.82
R = 0
qtr_hat = np.zeros_like(q_transp)
qtr_hat[:d] = q_transp[:d]
theta = thetas_qtr.copy()
for k in range(d, len(q_transp)):
    phi = []
    for i in range(1, n_qtr + 1):
        phi.append(q_transp[k - i])
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
    qtr_hat[k] = phi @ theta
    err = q_transp[k] - qtr_hat[k]
    gain = (P @ phi) / (lam + phi @ P @ phi + R)
    theta += gain * err
    P = (P - np.outer(gain, phi) @ P) / lam
end_arx = time.time()
tempo_arx_qtr = end_arx - start_arx

# --- Tempo do CASADI ---
start_casadi = time.time()
Lista_xf_reshaped, Lista_zf_reshaped, Inputs, grid = Sim_dynamics(n_pert=1, qtd_pts=len(t))
end_casadi = time.time()
tempo_casadi = end_casadi - start_casadi

print(f"Tempo de execução ARX Recursivo (pman): {tempo_arx:.4f} segundos")
print(f"Tempo de execução ARX Recursivo (qtr): {tempo_arx_qtr:.4f} segundos")
print(f"Tempo de execução CASADI: {tempo_casadi:.4f} segundos")