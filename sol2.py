

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from numpy import arange
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import skimage.color

###########
#CONSTANTS#
###########

RGB = 2
GRAY_SCALE = 1
DERIVEATIVE = np.array([[0.5], [0], [-0.5]])
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
    frequency = np.exp(-2*PI*1j/N)
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
    frequency = np.exp(2*PI*1j/N)
    IDFT_matrix = np.power(frequency, row*col)
    idft = np.matmul(IDFT_matrix,fourier_signal)*1/N
    return idft



def DFT2(image):
    """
    :param image: image is a grayscale image of dtype float64
    :return:
    """

    # Step 1: compute the DFT on the row's elements:
    fourier_image_row = DFT(image)

    # Step 2: transpose the Image we got in step #1:
    fourier_image_row = fourier_image_row.T

    # Step #3: Calculate the DFT with respect to the cols dimensions:
    fourier_image = DFT(fourier_image_row)

    return fourier_image.T



def IDFT2(fourier_image):
    """
    :param fourier_image: fourier_image is a 2D array of type
           complex128, both of shape (M,N) or (M,N,1)
    :return:
    """
    # Step 1: compute the IDFT on the row's elements:
    image_x = IDFT(fourier_image)

    # Step 2: transpose the Image we got in step #1:
    image_x = image_x.T

    # Step 3: Calculate the DFT with respect to the cols dimensions:
    image = IDFT(image_x)

    return image.T


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
    wav.write("change_rate.wav", int(ratio*sample_rate), data_orig.astype(np.float64))


def resize(data,ratio):
    """
    given a data array, ratio, this function will resize the data in accordance with the ratio parameter
    :param data: 1D ndarray of dtype float64 or complex128(*) representing the original sample points
    :param ratio: the ratio of the speed of the output file to the original file's speed (assume 0.25 < ratio < 4)
    :return: resize is a 1D ndarray of the dtype of data representing the new sample points.
    """
    # Step 1: Get Fourier transform of the data:
    frequencies = DFT(data)
    frequencies = np.fft.fftshift(frequencies)

    # Step 2: initialize some variable we will use later:
    new_freq = np.array([])
    N = len(frequencies)
    pivot = int(N/2)

    # Step 3: modify the frequencies:
    if ratio > 1:
        # Chop the high part of the frequencies:
        pos = frequencies[pivot - int(pivot*(1/ratio)): pivot]
        neg = frequencies[pivot: pivot + int(pivot * (1/ratio))]
        new_freq = np.hstack((neg, pos))
    elif ratio < 1:
        # Zero padding for the frequencies:
        new_freq = np.hstack((np.zeros(int(N*(ratio))), frequencies, np.zeros(int(N*(ratio)+1))))

    # Step 4: perform inverse Fourier transform:
    result = IDFT(new_freq)
    return np.real(result).astype(np.float64)

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
    wav.write("change_samples.wav", sample_rate ,resize(data_orig.astype(np.float64),ratio))


def resize_spectrogram(data, ratio):
    """
    given data and ratio, this function resizes the spectrogram's windows
    :param data: 1d array containing audio data we want to modify
    :param ratio: the ratio of the speed of the output file to the original file's speed (assume 0.25 < ratio < 4)
    :return: modified audio file
    """
    # Step 1: Build the spectrogram using stft:
    spectrogram = stft(data)

    # Step 2: initialize a new spectrogram with the correct shape:
    first_resize = resize(spectrogram[0,:], ratio)
    new_spectogram = np.zeros((int(spectrogram.shape[0]),len(first_resize)))
    new_spectogram[0,:] = first_resize
    # Step 3: iterate through the spectrogram's rows, and resize each row in respect to the ratio parameter:
    for row in range(1,len(spectrogram)):
        new_spectogram[row,:] = resize(spectrogram[row,:], ratio)

    # Step 4: perform istft on the spectrogram to get the modified audio:
    result = istft(new_spectogram)
    return result



def resize_vocoder(data, ratio):
    """
    given data and ratio, this function resizes the spectrogram's windows, but unlike the spectrogram_resize,
    this function will also make corrections regarding the phase of the signal.
    :param data: 1d array containing audio data we want to modify
    :param ratio: the ratio of the speed of the output file to the original file's speed (assume 0.25 < ratio < 4)
    :return: modified audio file
    """
    # Step 1: Build the spectrogram using stft:
    spectrogram = stft(data)

    # Step 2: initialize a new spectrogram with the correct shape:
    new_spectogram = np.zeros((int(spectrogram.shape[0]),resize(spectrogram[0,:], ratio).shape[0]))

    # Step 3: iterate through the spectrogram's rows, and resize each row in respect to the ratio parameter:
    for row in range(len(spectrogram)):
        new_spectogram[row,:] = resize(spectrogram[row,:], ratio)

    # Step 4: Fix the phase after resizing each row of the spectrogram:
    new_spectogram = phase_vocoder(new_spectogram, ratio)

    # Step 5: perform istft on the spectrogram to get the modified audio:
    result = istft(new_spectogram)
    return result



def conv_der(im):
    """
    Compute the derivative of the image using the convolution:
    :param im: image we want to derive
    :return: image's derivative with respect to both x and y directions:
    """

    # Step 1: calculate derivative with respect to x:
    dx = signal.convolve2d(im, DERIVEATIVE, mode="same")

    # Step 2: calculate derivative with respect to y:
    dy = signal.convolve2d(im, DERIVEATIVE.T, mode="same")

    # Step 3: calculate the magnitude of each Image's derivative and sum it to one pic:
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)

def fourier_der(im):
    """
    function to compute fourier coefficients
    :param im: image to derive using fourier
    :return:
    """
    #Step 1: compute DFT2:
    fourier_im = DFT2(im)
    shift = np.fft.fftshift(fourier_im)
    rows = fourier_im.shape[0]
    cols = fourier_im.shape[1]

    #Step 2: Get Matrices of the row indices and cols indices with respect to fft shift:
    fourier_y_freq, fourier_x_freq = np.meshgrid(np.arange(-rows//2, rows//2),np.arange(-cols//2, cols//2))

    #Step 3: Get the derivative images with respect to x and with respect to y:
    fourier_dx = np.fft.ifftshift(shift * fourier_x_freq.T*2*PI*1j/im.shape[1])
    fourier_dy = np.fft.ifftshift(shift * fourier_y_freq.T*2*PI*1j/im.shape[0])

    #Step 4: Calculate the IDFT for each Image we got from step #3:
    dx = IDFT2(fourier_dx)
    dy = IDFT2(fourier_dy)

    #Step 5: Compute the magnitude of each Image dx and dy, and sum it to one image:
    return np.sqrt(np.abs(dx)**2 + np.abs(dy)**2)

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
