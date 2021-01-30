"""pylsl library is used for this program: https://github.com/chkothe/pylsl/blob/master/pylsl/pylsl.py"""
"""The code was copied from the EXAMPLE file"""
"""The Example files are from two github files, """

import os
import sys
from tempfile import gettempdir
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from scipy.signal import butter, lfilter, lfilter_zi

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data

"""enum for the code"""

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used for muse
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256 / 2), btype='bandstop')

"""Extract epochs from a time series.
    Given a 2D array of the shape [n_samples, n_channels]
    Creates a 3D array of the shape [wlength_samples, n_channels, n_epochs]
    [data] (numpy.ndarray or list of lists): data [n_samples, n_channels]
    [samples_epoch] (int): window length in samples
    [samples_overlap] (int): Overlap between windows in samples
    Returns: (numpy.ndarray): epoched data of shape"""
def epoch(data, samples_epoch, samples_overlap=0):
    #does this conver the [data] from list to np.array? Are they different?
    if isinstance(data, list):
        data = np.array(data)

    #gets the dimension of the array
    n_samples, n_channels = data.shape
    #in case the data overlapped, probably by checking time stamp
    samples_shift = samples_epoch - samples_overlap
    #?????????what exactly is samples_epoch?????what exactly does sample_shift mean?????
    n_epochs = int(np.floor((n_samples - samples_epoch) / float(samples_shift)) + 1)

    # Markers indicate where the epoch starts, and the epoch contains samples_epoch rows
    markers = np.asarray(range(0, n_epochs + 1)) * samples_shift
    markers = markers.astype(int)

    # Divide data in epochs: this function reshapes the array into the dementions given in the parameter
    epochs = np.zeros((samples_epoch, n_channels, n_epochs))

    #brain not big enough
    for i in range(0, n_epochs):
        epochs[:, :, i] = data[markers[i]:markers[i] + samples_epoch, :]

    return epochs

"""Extract the features (band powers) from the EEG.
    [eegdata] (numpy.ndarray): array of dimension [number of samples, number of channels]
    [fs] (float): sampling frequency of eegdata
    Returns (numpy.ndarray): feature matrix of shape [number of feature points, number of different features]"""
def compute_band_powers(eegdata, fs):
    # 1. Compute the PSD
    winSampleLength, nbCh = eegdata.shape

    # Apply Hamming window; weighted cosine function?????
    w = np.hamming(winSampleLength)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # Remove offset
    dataWinCenteredHam = (dataWinCentered.T * w).T

    NFFT = nextpow2(winSampleLength)
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :])
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))

    # SPECTRAL FEATURES
    # Average of band powers
    # Delta <4
    ind_delta, = np.where(f < 4)
    meanDelta = np.mean(PSD[ind_delta, :], axis=0)
    # Theta 4-8
    ind_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(PSD[ind_theta, :], axis=0)
    # Alpha 8-12
    ind_alpha, = np.where((f >= 8) & (f <= 12))
    meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
    # Beta 12-30
    ind_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(PSD[ind_beta, :], axis=0)

    feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha,
                                     meanBeta), axis=0)

    feature_vector = np.log10(feature_vector)

    return feature_vector


def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n


def compute_feature_matrix(epochs, fs):
    """
    Call compute_feature_vector for each EEG epoch
    """
    n_epochs = epochs.shape[2]

    for i_epoch in range(n_epochs):
        if i_epoch == 0:
            feat = compute_band_powers(epochs[:, :, i_epoch], fs).T
            # Initialize feature_matrix
            feature_matrix = np.zeros((n_epochs, feat.shape[0]))

        feature_matrix[i_epoch, :] = compute_band_powers(
            epochs[:, :, i_epoch], fs).T

    return feature_matrix


def get_feature_names(ch_names):
    """Generate the name of the features.
    Args:
        ch_names (list): electrode names
    Returns:
        (list): feature names
    """
    bands = ['delta', 'theta', 'alpha', 'beta']

    feat_names = []
    for band in bands:
        for ch in range(len(ch_names)):
            feat_names.append(band + '-' + ch_names[ch])

    return feat_names


def update_buffer(data_buffer, new_data, notch=False, filter_state=None):
    """
    Concatenates "new_data" into "data_buffer", and returns an array with
    the same size as "data_buffer"
    """
    if new_data.ndim == 1:
        new_data = new_data.reshape(-1, data_buffer.shape[1])

    if notch:
        if filter_state is None:
            filter_state = np.tile(lfilter_zi(NOTCH_B, NOTCH_A),
                                   (data_buffer.shape[1], 1)).T
        new_data, filter_state = lfilter(NOTCH_B, NOTCH_A, new_data, axis=0,
                                         zi=filter_state)

    new_buffer = np.concatenate((data_buffer, new_data), axis=0)
    new_buffer = new_buffer[new_data.shape[0]:, :]

    return new_buffer, filter_state


def get_last_data(data_buffer, newest_samples):
    """
    Obtains from "buffer_array" the "newest samples" (N rows from the
    bottom of the buffer)
    """
    new_buffer = data_buffer[(data_buffer.shape[0] - newest_samples):, :]

    return new_buffer

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    # [reslove_byprop(prop, value, minimum=1, timeout=FOREVER)]
    #   this checks the bluetooth connection list and finds the [minimum] number of streams
    #   this uses multiple library calls, not sure what they do...
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=5)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
        #if there are no streams found, raise the error

    # Set active EEG stream to inlet and apply time correction
    # [StreamInlet(info, max_buflen=360, max_chunklen=0, recover=True)]
    #   [info] is the bluetooth stream: [dataType(info) = Streaminfo]
    #   [max_buflen] is the seconds of data the program will buffer, "real time applications would only buffer as much as the program needs to perform the calculations"
    #   [max_chunklen] defaults to the sender's transmission length; (?) this should be specified to make the program consistant, as real time bluetooth transmission can be unreliable
    #   [recover] can recover data; if stream does not exist due to crash or something it throws lost error
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()
    # [<StreamInlet>.time_correction(timeout=FOREVER)] uses lsl library to estimate and correct the data's time offset from the real time and apply it

    # Get the stream info and description
    # [<StreamInlet>.info(timeout=FOREVER)] gets the data of the stream using lsl library; creates a class that contains all the info
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())
    
    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            # [<StreamInlet>.pull_chunk(timeout=0.0, max_samples=1024)] pulls the data aquired and queued in storage.
            #   [timeout] indicates the time limit for pulling data from the stream
            #   [max_samples] is the maximum amount of data to pull
            # NOT ACTUALLY SURE HOW THIS FUNCTION WORKS
            #   what are these [[____ for _ in _] for _ in _]
            eeg_data, timestamp = inlet.pull_chunk(1, int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            # [:] is for everything; how does this 1D array
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data, this function and all below can be seen above
            eeg_buffer, filter_state = update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = compute_band_powers(data_epoch, fs)
            band_buffer, _ = update_buffer(band_buffer, np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)

            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
            # These metrics could also be used to drive brain-computer interfaces

            # Alpha Protocol:
            # Simple redout of alpha power, divided by delta waves in order to rule out noise
            alpha_metric = smooth_band_powers[Band.Alpha] / \
                smooth_band_powers[Band.Delta]
            print('Alpha Relaxation: ', alpha_metric)

            # Beta Protocol:
            # Beta waves have been used as a measure of mental activity and concentration
            # This beta over theta ratio is commonly used as neurofeedback for ADHD
            # beta_metric = smooth_band_powers[Band.Beta] / \
            #     smooth_band_powers[Band.Theta]
            # print('Beta Concentration: ', beta_metric)

            # Alpha/Theta Protocol:
            # This is another popular neurofeedback metric for stress reduction
            # Higher theta over alpha is supposedly associated with reduced anxiety
            # theta_metric = smooth_band_powers[Band.Theta] / \
            #     smooth_band_powers[Band.Alpha]
            # print('Theta Relaxation: ', theta_metric)

    except KeyboardInterrupt:
        print('Closing!')
