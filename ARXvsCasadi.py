import time
import numpy as np
from ARX_Recursivo import (
    pman, pman_media, pman_std, q_transp, q_transp_media, q_transp_std,
    thetas_pman,q_transp, thetas_qtr, F_Booster, F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4,
    t
)
from initialization_oil_production_basic import Sim_dynamics

# --- Tempo do ARX Recursivo ---
start_arx = time.time()

# Parâmetros do modelo ARX recursivo (Pressão Manifold)
n_pman, n_pchoke, n_dP_bcs, n_F_Booster, n_F_bcs = 1, 1, 2, 1, 2
d = 2
n_params = n_pman + (n_pchoke * 4) + (n_dP_bcs * 4) + n_F_Booster + (n_F_bcs * 4)
P = np.eye(n_params) * 0.43
lam = 1
R = 1e-6
pman_hat = np.zeros_like(pman)
pman_hat[:d] = pman[:d]
theta = thetas_pman.copy()
for k in range(d, len(pman)):
    phi = []
    for i in range(1, n_pman + 1):
        phi.append(pman[k - i])
    for i in range(1, n_pchoke + 1):
        phi.append(0)  # Substitua por p_choke_1[k - i] se necessário
        phi.append(0)
        phi.append(0)
        phi.append(0)
    for i in range(1, n_dP_bcs + 1):
        phi.append(0)  # Substitua por dP_bcs_1[k - i] se necessário
        phi.append(0)
        phi.append(0)
        phi.append(0)
    for i in range(1, n_F_Booster + 1):
        phi.append(F_Booster[k - i])
    for f_bcs in [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4]:
        for i in range(1, n_F_bcs + 1):
            phi.append(f_bcs[k - i])
    phi = np.array(phi)
    pman_hat[k] = phi @ theta
    err = pman[k] - pman_hat[k]
    gain = (P @ phi) / (lam + phi @ P @ phi)
    theta += gain * err
    P = (P - np.outer(gain, phi) @ P) / lam

end_arx = time.time()
tempo_arx = end_arx - start_arx

# --- Tempo do CASADI ---
start_casadi = time.time()
# Simula 1 perturbação, pode ajustar qtd_pts e n_pert conforme necessário
Lista_xf_reshaped, Lista_zf_reshaped, Inputs, grid = Sim_dynamics(n_pert=1, qtd_pts=len(t))
end_casadi = time.time()
tempo_casadi = end_casadi - start_casadi

print(f"Tempo de execução ARX Recursivo: {tempo_arx:.4f} segundos")
print(f"Tempo de execução CASADI: {tempo_casadi:.4f} segundos")




# --- Tempo do ARX Recursivo para qtr ---
start_arx = time.time()

n_qtr, n_F_Booster, n_F_bcs, n_valve = 1, 3, 3, 3
d = 4  # maior lag
n_params = n_qtr + n_F_Booster + (n_F_bcs * 4)
P = np.eye(n_params) * 1.97
lam = 1
qtr_hat = np.zeros_like(q_transp)
qtr_hat[:d] = q_transp[:d]
theta = thetas_qtr.copy()
for k in range(d, len(q_transp)):
    phi = []
    for i in range(1, n_qtr + 1):
        phi.append(q_transp[k - i])
    for i in range(1, n_F_Booster + 1):
        phi.append(F_Booster[k - i])
    for f_bcs in [F_bcs_1, F_bcs_2, F_bcs_3, F_bcs_4]:
        for i in range(1, n_F_bcs + 1):
            phi.append(f_bcs[k - i])
    phi = np.array(phi)
    qtr_hat[k] = phi @ theta
    err = q_transp[k] - qtr_hat[k]
    gain = (P @ phi) / (lam + phi @ P @ phi)
    theta += gain * err
    P = (P - np.outer(gain, phi) @ P) / lam

end_arx = time.time()
tempo_arx = end_arx - start_arx

# --- Tempo do CASADI ---
start_casadi = time.time()
Lista_xf_reshaped, Lista_zf_reshaped, Inputs, grid = Sim_dynamics(n_pert=1, qtd_pts=len(t))
end_casadi = time.time()
tempo_casadi = end_casadi - start_casadi

print(f"Tempo de execução ARX Recursivo (qtr): {tempo_arx:.4f} segundos")
print(f"Tempo de execução CASADI: {tempo_casadi:.4f} segundos")