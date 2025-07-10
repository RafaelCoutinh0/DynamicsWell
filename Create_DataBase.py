import os
import numpy as np
from initialization_oil_production_basic import plotar_graficos

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

# Cria a nova pasta
pasta_simulacao = f'Simulações/Sim_{i}'
os.makedirs(pasta_simulacao)

# Executa a simulação
Lista_xf_reshaped, Lista_zf_reshaped, Inputs = plotar_graficos(n_pert)

# Salva os arquivos na pasta
np.save(os.path.join(pasta_simulacao, 'xf.npy'), np.array(Lista_xf_reshaped))
np.save(os.path.join(pasta_simulacao, 'zf.npy'), np.array(Lista_zf_reshaped))
np.save(os.path.join(pasta_simulacao, 'Entradas.npy'), np.array(Inputs))


def create_data_reaction_curve():
    from initialization_oil_production_basic import *
    global res
    global tfinal

    Lista_xf = []
    Lista_zf = []
    Lista_xf.append(res["xf"])
    Lista_zf.append(res["zf"])
    x0 = Lista_xf[-1][:, -1]
    z0 = Lista_zf[-1][:, -1]
    map_est = []
    map_est.append(x0)
    map_est.append(z0)
    Lista_zf = np.array(Lista_zf)
    Lista_xf = np.array(Lista_xf)
    Lista_zf_reshaped = Lista_zf.reshape(8, qtd_pts)
    Lista_xf_reshaped = Lista_xf.reshape(14, qtd_pts)

    # criando as pertubações de u0
    b_booster_freq = 50.
    b_bcs_freq1 = 45.
    b_bcs_freq2 = 45.
    b_bcs_freq3 = 45.
    b_bcs_freq4 = 45.
    b_valve_open1 = 0.7
    b_valve_open2 = 0.7
    b_valve_open3 = 0.7
    b_valve_open4 = 0.7
    b_p_topo = 1e5

    grid_cont = 1
    for i in range(10):
        grid_cont += 1
        delta = 1000
        grid = linspace(tfinal, tfinal + delta, qtd_pts)
        tfinal += delta
        u0 = [booster_freq[i], p_topo[i] ** 4, bcs_freq1[i], valve_open1[i], bcs_freq2[i], valve_open2[i], bcs_freq3[i],
              valve_open3[i], bcs_freq4[i], valve_open4[i]]
        res = F(x0=x0, z0=z0, p=u0)
        x0 = res["xf"][:, -1]
        z0 = res["zf"][:, -1]
        map_est.append(x0)
        map_est.append(z0)
        Inputs = []
        Inputs.append(u0)
        Lista_xf_reshaped = np.hstack((Lista_xf_reshaped, np.array(res["xf"])))
        Lista_zf_reshaped = np.hstack((Lista_zf_reshaped, np.array(res["zf"])))

        # Plotted Graphs
        rcParams['axes.formatter.useoffset'] = False
        grid = linspace(0, tfinal, qtd_pts * grid_cont)

    def Auto_plot(i, t, xl, yl, c):
        plt.plot(grid, i.transpose(), c)
        matplotlib.pyplot.title(t)
        matplotlib.pyplot.xlabel(xl)
        matplotlib.pyplot.ylabel(yl)
        conc = np.concatenate(i)
        y_min, y_max = np.min(conc), np.max(conc)
        plt.ylim([y_min - 0.1 * abs(y_max), y_max + 0.1 * abs(y_max)])
        plt.grid()
        plt.show()

    # Auto_plot(Lista_zf_reshaped[[1, 3, 5, 7], :], "Pressão de Saída BCS", 'Time/(h)', 'Pressure/(bar)', 'b')
    # Auto_plot(Lista_xf_reshaped[[2, 5, 8, 11], :], "Pressure de Fundo de Poço", 'Time/(h)', 'Pressure/(bar)', 'r')
    # Auto_plot(Lista_xf_reshaped[[3, 6, 9, 12], :], 'Pressão da Choke', 'Time/(h)', 'Pressure/(bar)', 'g')
    # Auto_plot(Lista_xf_reshaped[[4, 7, 10, 13], :], 'Vazão dos Poços', 'Time/(h)', 'Flow Rate/(m^3/h)', 'k')
    # Auto_plot(Lista_xf_reshaped[[1], :], 'Vazão Manifold', 'Time/(h)', 'Flow Rate/(m^3/h)','y')
    # Auto_plot(Lista_xf_reshaped[[0], :], 'Pressão Manifold', 'Time/(h)', 'Pressure/(bar)', 'm')

    # p_intake é desnecessário
    # Auto_plot(Lista_zf_reshaped[[0, 2, 4, 6], :],"Pressão de Entrada BCS", 'Time/(h)', 'Pressure/(bar)', 'c')
    return Lista_xf_reshaped, Lista_zf_reshaped, Inputs