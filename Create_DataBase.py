import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import random
from numpy import linspace
from initialization_oil_production_basic import F, u0, x_ss, z_ss, Sim_dynamics, integrator, dae

grid = np.linspace(0, 1500, 100)
F = integrator('F', 'idas', dae, 0, grid)

def create_data_reaction_curve():
    # Inicialização
    Lista_xf = []
    Lista_zf = []
    Inputs = []

    x0 = x_ss
    z0 = z_ss

    # Simulação inicial com condição base
    res = F(x0=x0, z0=z0, p=u0)
    Lista_xf.append(res["xf"])
    Lista_zf.append(res["zf"])
    Inputs.append(u0)
    x0 = res["xf"][:, -1]
    z0 = res["zf"][:, -1]

    # Perturbações programadas nos controles
    alteracoes = {
        0: (0, 60.0),
        3: (2, 40.0),
        6: (3, 0.8),
        9: (4, 40.0),
        12: (5, 0.8),
        15: (6, 40.0),
        18: (7, 0.8),
        21: (8, 40.0),
        24: (9, 0.8),
    }

    for i in range(25):
        u_mod = u0.copy()
        if i in alteracoes:
            idx, val = alteracoes[i]
            u_mod[idx] = val

        res = F(x0=x0, z0=z0, p=u_mod)
        x0 = res["xf"][:, -1]
        z0 = res["zf"][:, -1]
        Lista_xf.append(res["xf"])
        Lista_zf.append(res["zf"])
        Inputs.append(u_mod)

    # Reorganizando resultados
    Lista_xf = np.hstack(Lista_xf)
    Lista_zf = np.hstack(Lista_zf)
    tfinal = 1500 * 26  # 1500 segundos por simulação, 26 simulações
    grid = np.linspace(0, tfinal, 100 * 26)  # 26 é o número de simulações
    # Função auxiliar de plotagem
    def auto_plot(data, title_str, x_label, y_label, color):
        plt.figure(figsize=(10, 5))
        plt.plot(grid, data.T, color)
        plt.title(title_str)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        y_min, y_max = np.min(data), np.max(data)
        plt.ylim([y_min - 0.1 * abs(y_max), y_max + 0.1 * abs(y_max)])
        # 👉 Adiciona linhas verticais a cada 1000 horas
        for xline in range(0, int(grid[-1]) + 1, 100):
            plt.axvline(x=xline, color='gray', linestyle='--', linewidth=1)

        plt.grid()
        plt.show()

    def plot_pert(Lista_xf, Lista_zf, tfinal, var_idx=0):
        # Define os blocos com janela de 3000 segundos a cada 4500 segundos
        janela = 3000
        passo = 4500
        max_ini = int(tfinal) - janela

        for ini in range(0, max_ini + 1, passo):
            fim = ini + janela

            # Encontra os índices correspondentes no vetor de tempo
            idxs = np.where((grid >= ini) & (grid <= fim))[0]
            if len(idxs) == 0:
                continue

            tempo_plot = grid[idxs]
            dados_plot = Lista_xf[var_idx, idxs]

            plt.figure(figsize=(10, 5))
            plt.plot(tempo_plot, dados_plot, 'm')
            plt.title(f'Variável {var_idx} - Alteração {int(ini/300) + 1}')
            plt.xlabel('Tempo (s)')
            plt.ylabel('Valor')
            plt.grid()
            plt.show()

    # Plotagem dos principais sinais
    rcParams['axes.formatter.useoffset'] = False
    # auto_plot(Lista_zf[[1, 3, 5, 7], :], "Pressão de Saída BCS", 'Tempo (s)', 'Pressão (bar)', 'b')
    # auto_plot(Lista_xf[[2, 5, 8, 11], :], "Pressão de Fundo de Poço", 'Tempo (s)', 'Pressão (bar)', 'r')
    # auto_plot(Lista_xf[[3, 6, 9, 12], :], 'Pressão nas Chokes', 'Tempo (s)', 'Pressão (bar)', 'g')
    # auto_plot(Lista_xf[[4, 7, 10, 13], :], 'Vazão dos Poços', 'Tempo (s)', 'Vazão (m³/h)', 'k')
    # auto_plot(Lista_xf[[1], :], 'Vazão Manifold', 'Tempo (s)', 'Vazão (m³/h)', 'y')
    auto_plot(Lista_xf[[0], :], 'Pressão Manifold', 'Tempo (s)', 'Pressão (bar)', 'm')
    # auto_plot(Lista_zf[[0, 2, 4, 6], :], "Pressão de Entrada BCS", 'Tempo (s)', 'Pressão (bar)', 'c')
    qtd_pts_sim = res["xf"].shape[1]
    plot_pert(Lista_xf, Lista_zf,tfinal , qtd_pts_sim, var_idx=0)
    plot_pert(Lista_xf, Lista_zf,tfinal,  qtd_pts_sim, var_idx=1)


    return Lista_xf, Lista_zf, Inputs, grid
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def create_data_AC_DC():
    qtd_pts_inicial = 100
    grid = np.linspace(0, 1500, qtd_pts_inicial)
    F = integrator('F', 'idas', dae, 0, grid)
    Lista_xf = []
    Lista_zf = []
    x0 = x_ss
    z0 = z_ss
    Inputs = []

    base_valve_open = 0.5
    base_bcs = 56.
    delta = 0.1
    delta_bcs = 5
    alteracoes = {}
    pert_indices = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    for idx in pert_indices:
        if idx in [0, 2, 4, 6, 8]:
            val = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs), 35, 65)
        else:
            val = np.clip(base_valve_open + np.random.uniform(-delta, delta), 0, 1.0)
        alteracoes[len(alteracoes)] = (idx, val)
        if idx in [0, 2, 4, 6, 8]:
            val = np.clip(base_bcs + np.random.uniform(-delta_bcs, delta_bcs), 35, 65)
        else:
            val = np.clip(base_valve_open + np.random.uniform(-delta, delta), 0, 1.0)
        alteracoes[len(alteracoes)] = (idx, val)

    for i in range(18):
        u_mod = u0.copy()
        idx, val = alteracoes[i]
        u_mod[idx] = val
        for _ in range(qtd_pts_inicial):
            Inputs.append(u_mod.copy())
        res = F(x0=x0, z0=z0, p=u_mod)
        x0 = res["xf"][:, -1]  # Atualiza para o final da simulação anterior
        z0 = res["zf"][:, -1]
        Lista_xf.append(res["xf"])
        Lista_zf.append(res["zf"])

    Lista_xf = np.hstack(Lista_xf)
    Lista_zf = np.hstack(Lista_zf)
    tfinal = 1500 * 18
    grid = np.linspace(0, tfinal, 18 * qtd_pts_inicial)

    def auto_plot(data, title_str, x_label, y_label, color):
        plt.figure(figsize=(10, 5))
        plt.plot(grid, data.T, color)
        plt.title(title_str)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        y_min, y_max = np.min(data), np.max(data)
        plt.ylim([y_min - 0.1 * abs(y_max), y_max + 0.1 * abs(y_max)])
        plt.grid()
        plt.show()

    rcParams['axes.formatter.useoffset'] = False
    auto_plot(Lista_xf[[0], :], 'Pressão Manifold', 'Tempo (s)', 'Pressão (bar)', 'm')

    return Lista_xf, Lista_zf, np.array(Inputs).T, grid

# === NOME DAS VARIÁVEIS ===
xf_names = [
    'P_man', 'q_transp',
    'P_fbhp_1', 'P_choke_1', 'q_mean_1',
    'P_fbhp_2', 'P_choke_2', 'q_mean_2',
    'P_fbhp_3', 'P_choke_3', 'q_mean_3',
    'P_fbhp_4', 'P_choke_4', 'q_mean_4'
]
zf_names = [
    'P_intake_1', 'dP_bcs_1',
    'P_intake_2', 'dP_bcs_2',
    'P_intake_3', 'dP_bcs_3',
    'P_intake_4', 'dP_bcs_4'
]
# === GERAR DADOS DE SIMULAÇÃO ===
print("Executando simulação para obter dados...")
n_pert = 10  # número de perturbações (ajustável)
i = 1

# while os.path.exists(f'Simulações/Sim_{i}'):
#     i += 1
# pasta_simulacao = f'Simulações/Sim_{i}'
# os.makedirs(pasta_simulacao)
# Lista_xf_reshaped, Lista_zf_reshaped, Inputs, time = Sim_dynamics(n_pert)

# while os.path.exists(f'Simulações/Sim_react_{i}'):
#     i += 1
#
# pasta_simulacao = f'Simulações/Sim_react_{i}'
# os.makedirs(pasta_simulacao)
# Lista_xf_reshaped, Lista_zf_reshaped, Inputs, time = create_data_reaction_curve()

# while os.path.exists(f'Simulações/Sim_AC_CC_{i}'):
#     i += 1
#
# pasta_simulacao = f'Simulações/Sim_AC_CC_{i}'
# os.makedirs(pasta_simulacao)
# Lista_xf_reshaped, Lista_zf_reshaped, Inputs, time = Sim_dynamics(5)

while os.path.exists(f'Simulações/Sim_ARX{i}'):
    i += 1
pasta_simulacao = f'Simulações/Sim_ARX{i}'
os.makedirs(pasta_simulacao)

pasta_origem = 'Simulações/Sim_AC_CC_2'
xt = np.load(os.path.join(pasta_origem, 'xt.npy'), allow_pickle=True)
zt = np.load(os.path.join(pasta_origem, 'zt.npy'), allow_pickle=True)
ut = np.load(os.path.join(pasta_origem, 'ut.npy'), allow_pickle=True)

Lista_xf_reshaped, Lista_zf_reshaped, Inputs, time = Sim_dynamics(n_pert= 20 , u0 = [ut[0][-1], ut[1][-1],ut[2][-1], ut[3][-1],ut[4][-1], ut[5][-1],ut[6][-1], ut[7][-1],ut[8][-1], ut[9][-1]], x0=[xt[0][-1], xt[1][-1],xt[2][-1], xt[3][-1],xt[4][-1], xt[5][-1],xt[6][-1], xt[7][-1],xt[8][-1], xt[9][-1],xt[10][-1], xt[11][-1],xt[12][-1], xt[13][-1]], z0=[zt[0][-1], zt[1][-1],zt[2][-1], zt[3][-1],zt[4][-1], zt[5][-1],zt[6][-1], zt[7][-1]])


# Salva os arquivos na pasta
np.save(os.path.join(pasta_simulacao, 'xt.npy'), np.array(Lista_xf_reshaped))
np.save(os.path.join(pasta_simulacao, 'zt.npy'), np.array(Lista_zf_reshaped))
np.save(os.path.join(pasta_simulacao, 'ut.npy'), Inputs)
np.save(os.path.join(pasta_simulacao, 'time.npy'), np.array(time))

