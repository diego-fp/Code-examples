import numpy as np
import scipy
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import pandas as pd

def hacer_fft(numero_imagen):
    fname = 'FoPra_Group_50_' + str(numero_imagen)
    neighborhood_size = 60

    data = scipy.misc.imread(fname + '.jpg')
    data_grayscale = np.mean(data, axis = 2)

    data_fft = np.fft.fft2(data_grayscale)

    plt.figure()
    plt.imshow(np.real(data_fft), cmap = 'gray')
    plt.show()