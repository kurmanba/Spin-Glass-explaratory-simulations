import numpy as np
from collections import defaultdict


vf = [0.07, 0.07, 0.07, 0.07, 0.07]
vf_c_desired = [0.35, 0.175, 0.175/2, 0.175/2, 0.175/8]
vf_corrections_needed = np.zeros(len(vf))

albam = [0, 0, 0, 0, 0]
albam_c_desired = [50, 50, 50, 50, 50]
albam_c_needed = np.zeros(len(vf))

na_cl = [0, 0, 0, 0, 0]                  # we don't care
na_cl_c_desired = [9, 9, 9, 9, 9]
na_cl_c_needed = np.zeros(len(vf))

dv_albam = np.linspace(0.001, 0.1, 1000)
dv_na_cl = np.linspace(0.001, 0.1, 1000)


tot_obj = 1000000
temp_alb = []
temp_na_cl = []

current_best = defaultdict()

print(tot_obj)

for i, j in enumerate(dv_albam):
    for k, l in enumerate(dv_na_cl):

        tot = vf[0] + j + l
        obj_1 = np.abs(vf_c_desired[0] - vf[0]/tot)
        obj_2 = np.abs(albam_c_desired[0] - j/tot)
        # obj_3 = np.abs(na_cl_c_needed[0] - l/tot)
        temp_obj = obj_1 + obj_2

        if temp_obj < tot_obj:
            current_best["alb"] = j
            current_best["salt"] = l
            tot_obj = temp_obj

print(current_best["alb"], current_best["salt"])















