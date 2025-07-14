import numpy as np
import matplotlib.pyplot as plt
import os

pasta_simulacao = 'Simulações/Sim_AC_CC_2'
xt = np.load(os.path.join(pasta_simulacao, 'xt.npy'), allow_pickle=True)
zt = np.load(os.path.join(pasta_simulacao, 'zt.npy'), allow_pickle=True)
t = np.load(os.path.join(pasta_simulacao, 'time.npy'), allow_pickle=True)
ut = np.load(os.path.join(pasta_simulacao, 'ut.npy'), allow_pickle=True)

pasta_simulacao = 'Simulações/Sim_ARX1'
xt2 = np.load(os.path.join(pasta_simulacao, 'xt.npy'), allow_pickle=True)
zt2 = np.load(os.path.join(pasta_simulacao, 'zt.npy'), allow_pickle=True)
t2 = np.load(os.path.join(pasta_simulacao, 'time.npy'), allow_pickle=True)
ut2 = np.load(os.path.join(pasta_simulacao, 'ut.npy'), allow_pickle=True)

for i in range(len(t2)):
    t2[i] += t[-1]

xt_tot = np.hstack((xt, xt2))
zt_tot = np.hstack((zt, zt2))
ut_tot = np.hstack((ut, ut2))
t_tot = np.hstack((t, t2))


plt.plot(t_tot, xt_tot[0], color='blue', label='ARX recursivo')
plt.plot(t, xt[0], color='red', label='Treinamento de teta')
plt.axvline(x=6000, color='r', linestyle='--', label='t = 6000')
plt.title('Pressão Manifold')
plt.xlabel('Tempo (s)')
plt.ylabel('Valor')
plt.legend()
plt.show()