

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import skimage.color

###########
#CONSTANTS#
###########

RGB = 2
GRAY_SCALE = 1
DERIVEATIVE = np.array([0.5, 0, -0.5])
PI = np.pi


def unitRoot(x,N,u):
    return np.exp((-2*PI*1j*x*u)/N)

def inverseUnitRoot(x,N,u):
    return np.exp((2*PI*1j*x*u)/N)

# 1.1
def DFT(signal):
    """
    A simple implementation of DFT for a given signal.
    :param signal: an array of float64 and shape of (N,) or (N,1)
    :return: a complex Fourier transform of the given signal.
    """
    signal = np.array(np.copy(signal))
    N = signal.shape[0]
    if N == 0:
        return []
    row, col = np.meshgrid(np.arange(N),np.arange(N))
    frequency = np.exp(2*PI*1j/N)
    DFT_matrix = np.power(frequency, row*col)
    dft = np.matmul(DFT_matrix, signal)
    return dft



def IDFT(fourier_signal):
    """
    A simple Inverse Fourier transform of a given signal
    :param fourier_signal: an array of complex128 and shape of (N,) or (N,1)
    :return: a complex signal of the given Fourier transform
    """
    fourier_signal = np.copy(fourier_signal)
    N = fourier_signal.shape[0]
    if N == 0:
        return []
    row, col = np.meshgrid(np.arange(N),np.arange(N))
    frequency = np.exp(-2*PI*1j/N)
    IDFT_matrix = np.power(frequency, row*col)
    idft = np.matmul(IDFT_matrix,fourier_signal)*1/N
    return idft



def DFT2(image):
    """
    :param image: image is a grayscale image of dtype float64
    :return:
    """
    image = np.copy(image)
    size_rows, size_cols = image.shape[0], image.shape[1]
    print(size_rows, size_cols)
    result_img = np.zeros(image.shape, dtype=np.complex128)
    inner_dft = np.zeros(image.shape, dtype=np.complex128)
    for row in range(0, size_rows):
        inner_dft[row,:] = DFT(image[row,:])
    for col in range(0, size_cols):
        result_img[:, col] = np.matmul(DFT(image[:,col]),inner_dft[:,col])
    return result_img * 1/(size_rows*size_cols)


def IDFT2(fourier_image):
    """
    :param fourier_image: fourier_image is a 2D array of type
           complex128, both of shape (M,N) or (M,N,1)
    :return:
    """
    image = np.copy(fourier_image)
    size_rows, size_cols = image.shape[0], image.shape[1]
    result_img = np.zeros(image.shape, dtype=np.complex128)
    inner_dft = np.zeros(image.shape, dtype=np.complex128)
    for row in range(0, size_rows):
        inner_dft[row,:] = IDFT(image[row,:])
    for col in range(0,size_cols):
        result_img[:, col] = np.matmul(DFT(image[:,col]),inner_dft[:,col])
    return result_img


def change_rate(filename, ratio):
    """

    :param filename: a string representation of the .wav filename we want to edit
    :param ratio: the ratio of the speed of the output file to the original file's speed (assume 0.25 < ratio < 4)
    :return: save the manipulated data into a new file called 'change_rate.wav'. "
    """
    # read the file:
    audio_orig = wav.read(filename)

    # unpack the audio file's into its data and the sample rate of the original file:
    sample_rate,data_orig = audio_orig

    # manipulate the the data and save it to a new file with different speed:
    wav.write("change_rate.wav", int(ratio*sample_rate), data_orig)



def resize(data,ratio):
    """

    :param data: 1D ndarray of dtype float64 or complex128(*) representing the original sample points
    :param ratio: the ratio of the speed of the output file to the original file's speed (assume 0.25 < ratio < 4)
    :return: resize is a 1D ndarray of the dtype of data representing the new sample points.
    """

    frequencies = DFT(data)
    frequencies = np.fft.fftshift(frequencies)
    # plt.plot(frequencies)

    new_freq = np.array([])
    N = len(frequencies)
    pivot = int(N/2)
    if ratio > 1:
        # Chop the high part of the frequencies:
        pos = frequencies[pivot - int(pivot*(1/ratio)) : pivot]
        neg = frequencies[pivot: pivot + int(pivot * (1/ratio))]
        new_freq = np.hstack((neg, pos))
    elif ratio < 1:
        # Zero padding for the frequencies:
        new_freq = np.hstack((np.zeros(int(pivot/ratio)), frequencies, np.zeros(int(pivot/ratio))))

    result = IDFT(new_freq)

    return np.real(result)
    # If we want to create a slower version of the data:

def change_samples(filename, ratio):
    """
    Fast forward function that changes the duration of an audio file by reducing the number of samples using Fourier.
    This function does not change the sample rate of the given file.
    :param filename: a string representation of the .wav filename we want to edit
    :param ratio: the ratio of the speed of the output file to the original file's speed (assume 0.25 < ratio < 4)
    :return: save the manipulated data into a new file called 'change_samples.wav'. "
    """
    # read the file:
    audio_orig = wav.read(filename)

    # unpack the audio file's into its data and the sample rate of the original file:
    sample_rate, data_orig = audio_orig

    # Extract the frequencies of the audio file using the Fourier transform implementation we did:
    wav.write("change_samples.wav", sample_rate ,resize(data_orig,ratio).astype(np.float64))


def resize_spectrogram(data, ratio):
    spectrogram = stft(data)
    new_spectogram = np.array(np.shape(spectrogram.shape))
    for idx, interval in enumerate(spectrogram):
        new_spectogram[idx] = resize(interval, ratio)
    result = istft(new_spectogram)
    return result



def resize_vocoder(data, ratio):
    spectrogram = stft(data)
    new_spectogram = np.array(np.shape(spectrogram.shape))
    for idx, interval in enumerate(spectrogram):
        new_spectogram[idx] = resize(interval, ratio)
    new_spectogram = phase_vocoder(new_spectogram, ratio)
    result = istft(new_spectogram)
    return result



def conv_der(im):
    dx, dy = signal.convolve2d(im, DERIVEATIVE, mode="same"),signal.convolve2d(im, DERIVEATIVE.T, mode="same")
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)

def fourier_der(im):
    # fourier_dx = np.zeros(fourier_im.shape[0], fourier_im.shape[1])
    # fourier_dy = np.zeros(fourier_im.shape[0], fourier_im.shape[1])
    # for col in range(fourier_im.shape[0]):
    #     fourier_dx[:,col] = fourier_x_freq[:,col]*fourier_im[:,col]
    #
    # for row in range(fourier_im.shape[1]):
    #     fourier_dy[row, :] = fourier_y_freq[row,:] * fourier_im[row,:]

    fourier_im = DFT2(np.copy(im))
    fourier_y_freq, fourier_x_freq = np.meshgrid(fourier_im)
    fourier_dx = fourier_im * fourier_x_freq
    fourier_dy = fourier_im * fourier_y_freq
    fourier_derived = fourier_dx + fourier_dy
    image = IDFT2(fourier_derived)
    return image

####################################################################################
# next section is dedicated to helper functions implementation for this assignment:#
####################################################################################

def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

def read_image(filename, representation):
    """
    filename - the filename of an image on disk (could be grayscale or RGB).
    representation - representation code, either 1 or 2 defining whether the output should be a:
    grayscale image (1)
    or an RGB image (2).
    NOTE: If the input image is grayscale, we won’t call it with represen- tation = 2.
    :param filename: String - the address of the image we want to read
    :param representation: Int - as described above
    :return: an image in the correct representation
    """
    if representation != RGB and representation != GRAY_SCALE:
        return "Invalid Input. You may use representation <- {1, 2}"
    tempImage = plt.imread(filename)[:, :, :3]
    resultImage = np.array(tempImage)

    if representation == GRAY_SCALE:
        resultImage = skimage.color.rgb2gray(tempImage)
    elif representation == RGB:
        resultImage = tempImage
    if resultImage.max() > 1:
        resultImage = resultImage/255


    return resultImage.astype(np.float64)

