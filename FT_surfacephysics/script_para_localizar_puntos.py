import numpy as np
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pandas as pd
from localizar_puntos import localizar_puntos
from hacer_fft import hacer_fft

for num in np.arange(2,20,1):
    if num < 10:
        numero_completo = '000' + str(num)
    else:
        numero_completo = '00' + str(num)

    if num in [2, 3, 4, 7, 8]:
        mean_distance_inner, mean_distance_outer, angle_b1_b2, angle_d1_d2 = localizar_puntos(numero_completo, threshold = 40)
        #if num in [2, 3, 4]:
        #    distances_bare_crystal.append(pd.DataFrame([[mean_distance_inner, mean_distance_outer]]))
        #    angles_bare_crystal = np.append(angles_bare_crystal, [angle_b1_b2, angle_d1_d2])
        #if num in [7, 8]:
        #    distances_CO_first_adsorption = np.append(distances_CO_first_adsorption, [mean_distance_inner, mean_distance_outer])
        #    angles_CO_first_adsorption = np.append(angles_CO_first_adsorption, [angle_b1_b2, angle_d1_d2])
    elif num == 19:
        mean_distance_inner, mean_distance_outer, angle_b1_b2, angle_d1_d2 = localizar_puntos(numero_completo, threshold = 100)
        #distances_beam_damage = np.append(distances_beam_damage, [mean_distance_inner, mean_distance_outer])
        #angles_beam_damage = np.append(angles_beam_damage, [angle_b1_b2, angle_d1_d2])
    elif num in [9, 10, 11, 12]:
        mean_distance_inner, mean_distance_outer, angle_b1_b2, angle_d1_d2 = localizar_puntos(numero_completo, threshold = 50)
        #distances_CO_annealed = np.append(distances_CO_annealed, [mean_distance_inner, mean_distance_outer])
        #angles_CO_annealed = np.append(angles_CO_annealed, [angle_b1_b2, angle_d1_d2])
    elif num in [15, 16]:
        #localizar_puntos(numero_completo, threshold = 20)
        print("I skipped image number %i" % num)
    else:
        mean_distance_inner, mean_distance_outer, angle_b1_b2, angle_d1_d2 = localizar_puntos(numero_completo, threshold = 20)
        #if num in [5, 6]:
        #    distances_CO_first_adsorption = np.append(distances_CO_first_adsorption, [mean_distance_inner, mean_distance_outer])
        #    angles_CO_first_adsorption = np.append(angles_CO_first_adsorption, [angle_b1_b2, angle_d1_d2])
        #if num == 13:
        #    distances_CO_annealed = np.append(distances_CO_annealed, [mean_distance_inner, mean_distance_outer])
        #    angles_CO_annealed = np.append(angles_CO_annealed, [angle_b1_b2, angle_d1_d2])
        #if num in [14, 17, 18]:
        #    distances_beam_damage = np.append(distances_beam_damage, [mean_distance_inner, mean_distance_outer])
        #    angles_beam_damage = np.append(angles_beam_damage, [angle_b1_b2, angle_d1_d2])
