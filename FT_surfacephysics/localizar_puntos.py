
import numpy as np
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pandas as pd

def localizar_puntos(numero_imagen, threshold):

    fname = 'FoPra_Group_50_' + str(numero_imagen)
    neighborhood_size = 60

    data = scipy.misc.imread(fname+'.jpg')
    data = np.mean(data, axis = 2)

    data_max = ndimage.filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = ndimage.filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))

    puntos_xy = xy[:,[0,1]]
    distances = np.zeros((puntos_xy.shape[0], puntos_xy.shape[0]))
    for i in range(len(puntos_xy[:,0])):
        for j in range(len(puntos_xy[:,0])):
            distances[i, j] = np.sqrt((puntos_xy[i, 0] - puntos_xy[j, 0]) ** 2 + (puntos_xy[i, 1] - puntos_xy[j, 1]) ** 2)

    masks = np.zeros(distances.shape)
    puntos_finales_x = np.zeros((len(puntos_xy[:, 0]), 1))
    puntos_finales_y = np.zeros((len(puntos_xy[:, 0]), 1))

    for i in range(len(puntos_xy[:,0])):
        masks[i] = distances[i, :] < 40
        puntos_x = puntos_xy[:, 0][masks[i].astype(bool)]
        puntos_y = puntos_xy[:, 1][masks[i].astype(bool)]
        puntos_finales_x[i] = np.sum(puntos_x) / len(puntos_x)
        puntos_finales_y[i] = np.sum(puntos_y) / len(puntos_y)

    puntos_finales_x, indices = np.unique(puntos_finales_x, return_index = True)
    puntos_finales_x = puntos_finales_x.reshape(len(indices),1)
    puntos_finales_y = puntos_finales_y[indices]

    puntos_finales = np.concatenate((puntos_finales_y, puntos_finales_x), axis = 1)  # aqui le doy la vuelta a los indices

    # eliminamos los que estan fuera de la zona de interes

    mask_x = (puntos_finales[:, 0] > 455) * (puntos_finales[:, 0] < 900)  #850
    mask_y = (puntos_finales[:, 1] > 315) * (puntos_finales[:, 1] < 800)

    mask = (mask_x * mask_y).reshape(len(puntos_finales[:, 0]), 1)
    mask = np.concatenate((mask, mask), axis = 1)
    puntos_finales = puntos_finales[mask].reshape(np.sum(mask[:, 0]), 2)

    # calculo el centro de los hexagonos:
    centro = np.mean(puntos_finales, axis = 0)

    mask_interior = []
    distancia_interior = np.sqrt((puntos_finales[:, 0] - centro[0])**2 + (puntos_finales[:, 1] - centro[1])**2)
    mask_interior = distancia_interior > 70  # antes 150
    mask_interior = mask_interior.reshape(len(mask_interior), 1)
    mask_interior = np.concatenate((mask_interior, mask_interior), axis=1)
    puntos_finales = puntos_finales[mask_interior].reshape(np.sum(mask_interior[:, 0]), 2)

    df_final = pd.DataFrame({'X': puntos_finales[:, 0], 'Y': puntos_finales[:, 1]})

    # here I split the points into 2 groups: inner points and outer points
    df_final['d'] = (df_final['X'] - centro[0])**2 + (df_final['Y'] - centro[1])**2
    df_ordered = df_final.sort_values(by = ['d'], ascending = False)
    df_ordered_inner = df_ordered[6:12]
    df_ordered_outer = df_ordered[0:6]
    # finish the points classification

    # here I make the calculations
    distances_to_center_inner = np.sqrt((df_ordered_inner['X'] - centro[0])**2 + (df_ordered_inner['Y'] - centro[1])**2)
    distances_to_center_outer = np.sqrt((df_ordered_outer['X'] - centro[0])**2 + (df_ordered_outer['Y'] - centro[1])**2)
    mean_distance_inner = np.mean(distances_to_center_inner)
    mean_distance_outer = np.mean(distances_to_center_outer)

    b1_sin_centro = np.array([df_ordered_outer['X'].max(), df_ordered_outer['Y'][df_ordered_outer['X'].idxmax()]])
    b2_sin_centro = np.array([df_ordered_outer['X'][df_ordered_outer['Y'].idxmin()], df_ordered_outer['Y'].min()])
    b1 = b1_sin_centro - centro
    b2 = b2_sin_centro - centro

    angulo_b1 = np.arctan(b1[1] / b1[0]) * 180 / np.pi
    angulo_b2 = np.arctan(b2[1] / b2[0]) * 180 / np.pi
    angulo_b2 += 2 * (90 - angulo_b2)
    angulo_b1_b2 = angulo_b1 + angulo_b2
    print("Figure number:", numero_imagen)
    print("The angle between b1 and b2 is", angulo_b1_b2)

    if int(numero_imagen) in np.arange(5, 20, 1):
        d1_sin_centro = np.array([df_ordered_inner['X'].max(), df_ordered_inner['Y'][df_ordered_inner['X'].idxmax()]])
        d2_sin_centro = np.array([df_ordered_inner['X'][df_ordered_inner['Y'].idxmin()], df_ordered_inner['Y'].min()])
        d1 = d1_sin_centro - centro
        d2 = d2_sin_centro - centro

        angulo_d1 = np.arctan(d1[1] / d1[0]) * 180 / np.pi
        angulo_d2 = np.arctan(d2[1] / d2[0]) * 180 / np.pi
        angulo_d1_d2 = angulo_d1 - angulo_d2
        print("The angle between d1 and d2 is", angulo_d1_d2)
    else:
        angulo_d1_d2 = np.NaN


    #calculo de eta:
    a1 = 2.7  # Angstrom
    eta = (2 * np.pi)**(-1) * a1 * mean_distance_outer * np.cos(0.5 * angulo_b1_b2 * np.pi /180)  # Angstrom * pixel
    print("The coeficient eta is", eta)

    if int(numero_imagen) in np.arange(5, 20, 1):
        c = 2 * np.pi * eta / (mean_distance_inner * np.cos(angulo_d1_d2 * np.pi / 180))  # Angstrom
        print("The value of c is", c)

    # finish the calculations

    text_file = open('calculations', 'a')
    text_file.write("Figure number: %s \n" % numero_imagen)
    text_file.write('The angle between b1 and b2 is %f \n' % angulo_b1_b2)
    if int(numero_imagen) in np.arange(5, 20, 1):
        text_file.write('The angle between d1 and d2 is %f \n' % angulo_d1_d2)
        text_file.write('The value of c is %f \n' % c)
        text_file.write('The mean distance of the inner points is %f' % mean_distance_inner)
    text_file.write('The coefficient eta is %f \n' % eta)
    text_file.write('The mean distance of the outer points is %f \n\n' % mean_distance_outer)
    text_file.close()


    df_final.to_csv('data/csv_'+fname)

    print("\n")
    plt.figure()
    plt.imshow(data, cmap = 'gray')
    #plt.savefig('data/data_'+fname+'.png')


    plt.autoscale(False)
    plt.plot(df_ordered['X'], df_ordered['Y'], 'ro')
    plt.plot(centro[0], centro[1], 'bo')
    plt.savefig('data/result_'+fname+'.png')

    #plt.show()

    return mean_distance_inner, mean_distance_outer, angulo_b1_b2, angulo_d1_d2