import os
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import ccf, acf
from itertools import combinations
from initialization_oil_production_basic import Sim_dynamics

# === NOME DAS VARIÁVEIS ===
xt_names = [
    'P_man', 'q_transp',
    'P_fbhp_1', 'P_choke_1', 'q_mean_1',
    'P_fbhp_2', 'P_choke_2', 'q_mean_2',
    'P_fbhp_3', 'P_choke_3', 'q_mean_3',
    'P_fbhp_4', 'P_choke_4', 'q_mean_4'
]
zt_names = [
    'P_intake_1', 'dP_bcs_1',
    'P_intake_2', 'dP_bcs_2',
    'P_intake_3', 'dP_bcs_3',
    'P_intake_4', 'dP_bcs_4'
]
ut_names = [
    'F_Booster', 'P_topo',
    'F_bcs_1', 'valve_1',
    'F_bcs_2', 'valve_2',
    'F_bcs_3', 'valve_3',
    'F_bcs_4', 'valve_4'
]
pasta_simulacao = f'Simulações/Sim_AC_CC_2'
# Abrir os arquivos
xt = np.load(os.path.join(pasta_simulacao, 'xt.npy'), allow_pickle=True)
zt = np.load(os.path.join(pasta_simulacao, 'zt.npy'), allow_pickle=True)
ut = np.load(os.path.join(pasta_simulacao, 'ut.npy'), allow_pickle=True)
print(xt.shape)
print(zt.shape)
print(ut.shape)

# === CRIAR DATAFRAME ===
def criar_dataframe(xf, zf):
    all_data = np.vstack([xf, zf, ut])
    all_names = xt_names + zt_names + ut_names
    df = pd.DataFrame(all_data.T, columns=all_names)
    return df


df = criar_dataframe(xt, zt)
# === AUTOCORRELAÇÃO ===
import matplotlib.pyplot as plt
def plot_autocorrelacoes(df):
    pasta = 'Autocorrelações'
    for col in df.columns:
        var = df[col]
        acf_var = acf(var,fft=False,nlags=len(var)-1)
        plt.figure()
        plt.plot(acf_var, '.')
        plt.title(f'Autocorrelação - {col}')
        nome_autocorrelacoes = f"autocorrelacao_{col}"
        caminho_completo = os.path.join(pasta, nome_autocorrelacoes)
        plt.savefig(caminho_completo)
        plt.close()

# === CORRELAÇÃO CRUZADA ===
def plot_correlacoes_cruzadas(df):
    pasta = 'Correlações'
    pairs = list(combinations(df.columns, 2))
    for var1, var2 in pairs:
        x = df[var1] - df[var1].mean()
        y = df[var2] - df[var2].mean()
        ccf_vars = ccf(x,y,nlags=25)
        plt.figure()
        plt.plot(ccf_vars,'.')
        plt.title(f'Correlação cruzada - {var1} vs {var2}')
        nome_correlacoes = f'{var1} vs {var2}'
        caminho_completo = os.path.join(pasta, nome_correlacoes)
        plt.savefig(caminho_completo)
        plt.close()
# === MATRIZ DE CORRELAÇÃO ===
def matriz_correlacao(df):
    nome_matriz = "matriz_correlacao.png"
    pasta = 'Correlações'
    corr = df.corr()
    plt.figure(figsize=(28, 20))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Matriz de Correlação (Lag 0)")
    plt.tight_layout()
    caminho_completo = os.path.join(pasta, nome_matriz)
    plt.savefig(caminho_completo)
    plt.close()
# === EXECUÇÃO DAS ANÁLISES ===
print("Plotando autocorrelações...")
plot_autocorrelacoes(df)

print("Plotando correlações cruzadas...")
plot_correlacoes_cruzadas(df)

print("Plotando matriz de correlação...")
matriz_correlacao(df)
