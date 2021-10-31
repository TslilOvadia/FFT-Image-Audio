

import numpy as np
import matplotlib as matplt
PI = np.pi


def unitRoot(x,N,u):
    return np.exp((-2*PI*1j*x*u)/N)

def inverseUnitRoot(x,N,u):
    return np.exp((2*PI*1j*x*u)/N)

def DFT(signal):
    """

    :param signal: an array of float64 and shape of (N,) or (N,1)
    :return: a complex Fourier transform of the given signal.
    """
    signal = np.array(np.copy(signal))
    N  = signal.shape[0]
    row, col = np.meshgrid(np.arange(N),np.arange(N))
    frequency = np.exp(2*PI*1j/N)
    DFT_matrix = np.power(frequency, row*col)
    dft = np.matmul(DFT_matrix, signal.T)*1/N
    return dft




def IDFT(fourier_signal):
    """

    :param fourier_signal: an array of complex128 and shape of (N,) or (N,1)
    :return: a complex signal of the given Fourier transform
    """
    fourier_signal = np.copy(fourier_signal)
    N = fourier_signal.shape[0]
    row, col = np.meshgrid(np.arange(N),np.arange(N))
    frequency = np.exp(-2*PI*1j/N)
    IDFT_matrix =  np.power(frequency, row*col)
    idft = np.matmul(IDFT_matrix,fourier_signal.T )
    return idft
if __name__ == "__main__":
    pass