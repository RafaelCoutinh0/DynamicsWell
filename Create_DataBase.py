import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
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


    return Lista_xf, Lista_zf, Inputs

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

while os.path.exists(f'Simulações/Sim_{i}'):
    i += 1
pasta_simulacao = f'Simulações/Sim_{i}'
os.makedirs(pasta_simulacao)
Lista_xf_reshaped, Lista_zf_reshaped, Inputs, time = Sim_dynamics(n_pert)

# while os.path.exists(f'Simulações/Sim_react_{i}'):
#     i += 1
#
# pasta_simulacao = f'Simulações/Sim_react_{i}'
# os.makedirs(pasta_simulacao)
# Lista_xf_reshaped, Lista_zf_reshaped, Inputs = create_data_reaction_curve()


# Salva os arquivos na pasta
np.save(os.path.join(pasta_simulacao, 'xt.npy'), np.array(Lista_xf_reshaped))
np.save(os.path.join(pasta_simulacao, 'zt.npy'), np.array(Lista_zf_reshaped))
np.save(os.path.join(pasta_simulacao, 'ut.npy'), np.array(Inputs))
np.save(os.path.join(pasta_simulacao, 'time.npy'), np.array(time))