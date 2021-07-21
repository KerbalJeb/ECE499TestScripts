import numpy as np

p1 = np.load('tvec_p1.npy').reshape(-1, 3)
p2 = np.load('tvec_p2.npy').reshape(-1, 3)
p3 = np.load('tvec_p3.npy').reshape(-1, 3)
zp1 = np.load('tvec_zp1.npy').reshape(-1, 3)
zp2 = np.load('tvec_zp2.npy').reshape(-1, 3)

p1_mean = np.mean(p1, 0)
p2_mean = np.mean(p2, 0)
p3_mean = np.mean(p3, 0)
zp1_mean = np.mean(zp1, 0)
zp2_mean = np.mean(zp2, 0)

t12 = np.abs(p1_mean-p2_mean)
t13 = np.abs(p1_mean-p3_mean)
t23 = np.abs(p2_mean-p3_mean)

z12 = np.abs(zp1_mean-zp2_mean)
