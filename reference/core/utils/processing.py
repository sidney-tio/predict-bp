import math
import os
from collections import Counter
from dataclasses import dataclass

import neurokit2 as nk
import numpy as np
import scipy
import tensorflow as tf
from core.utils.kubioscloudcli import predict as predict_kubios
from kapre import STFT, Magnitude, MagnitudeToDecibel
from scipy.interpolate import UnivariateSpline
from scipy.signal import (butter, filtfilt, find_peaks, iirnotch, periodogram,
                          resample, welch)
from scipy.sparse import spdiags
from settings import config

os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp/'


KUBIOS_MIN_SAMPLES = 11


@dataclass
class VitalSignsResult:
    bpm: float = 0
    hrv: float = 0
    si: float = 0
    sns_index: float = 0
    resp_rate: float = 0
    sp: float = 0
    dp: float = 0
    spo2: float = 0
    o2: int = 0


def isNaN(num):
    return num != num


def butter_lowpass_filter(cutoff_freq, sampling_rate, order=2):
    nyqs = 0.5 * sampling_rate
    normal_cutoff_freq = cutoff_freq / nyqs
    b, a = butter(order, normal_cutoff_freq, btype='low', analog=False)
    return b, a


def butter_highpass_filter(cutoff_freq, sampling_rate, order=2):
    nyqs = 0.5 * sampling_rate
    normal_cutoff_freq = cutoff_freq / nyqs
    b, a = butter(order, normal_cutoff_freq, btype='high', analog=False)
    return b, a


def butter_bandpass_filter(lowcutoff, highcutoff, sampling_rate, order=2):
    nyqs = 0.5 * sampling_rate
    low_normal_cutoff = lowcutoff / nyqs
    high_normal_cutoff = highcutoff / nyqs
    b, a = butter(order, [low_normal_cutoff, high_normal_cutoff], btype='band')
    return b, a


def filter_signal(data, cutoff_freq, sampling_rate, order=2, filtertype='lowpass'):
    if filtertype.lower() == 'lowpass':
        b, a = butter_lowpass_filter(cutoff_freq, sampling_rate, order=order)
    elif filtertype.lower() == 'highpass':
        b, a = butter_highpass_filter(cutoff_freq, sampling_rate, order=order)
    elif filtertype.lower() == 'bandpass':
        assert type(
            cutoff_freq) == list or np.array, 'please enter the cutoff freqency in form of array or list'
        b, a = butter_bandpass_filter(
            cutoff_freq[0], cutoff_freq[1], sampling_rate, order=order)
    elif filtertype.lower() == 'notch':
        b, a = iirnotch(cutoff_freq, Q=0.005, fs=sampling_rate)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def calc_resp_rate(rr_intervals, sampling_rate=100, calc_method='fft', filter_freq=[0.1, 0.4]):
    """The function will calculate respicatory rate (number of breaths per min). The logic is to upsample the
    detected rr_intecvals using cubic slpine interpolation, apply th-eeshold at min 6 and max 24 breathes per min,
    at the end extract cespivatiocn vate from the signal. The cale_methed can be fft, welch and peridogram. The spline
    interpolation, welch and pericdogram are available in scipy and numpy libraries."""
    rr_intervals = np.round(rr_intervals).astype(np.int32)
    indepandant_data = rr_intervals
    depandent_data = np.linspace(0, len(rr_intervals), len(rr_intervals))
    interpolate = UnivariateSpline(depandent_data, indepandant_data, k=3)
    new_data = np.linspace(0, len(rr_intervals), np.sum(rr_intervals))
    interpolated_data = interpolate(new_data)

    breating_signal = interpolated_data
    new_samplingrate = 10 * sampling_rate
    filtered_breathingsignal = filter_signal(
        breating_signal, cutoff_freq=filter_freq, sampling_rate=new_samplingrate, filtertype='bandpass')

    if calc_method.lower() == 'fft':
        len_data = len(filtered_breathingsignal)
        frequency = np.fft.fftfreq(len_data, d=1 / new_samplingrate)
        frequency = frequency[range(int(len_data / 2))]
        psd_var = np.fft.fft(filtered_breathingsignal) / len_data
        psd_var = psd_var[range(int(len_data / 2))]
        psd = np.power(np.abs(psd_var), 2)
    elif calc_method.lower() == 'welch':
        frequency, psd = welch(filtered_breathingsignal,
                               fs=new_samplingrate, nperseg=len(filtered_breathingsignal))
    elif calc_method.lower() == 'periodogram':
        frequency, psd = periodogram(
            filtered_breathingsignal, fs=new_samplingrate, nperseg=len(filtered_breathingsignal))
    else:
        raise ValueError(
            'Calculation method enterd is not valid. Please select fft, welch or periodogram.')
    ts_measure = dict()
    ts_measure['respiratory_rate'] = frequency[np.argmax(psd)] * 60
    # dict_data['respiratory_signal'] = filtered_breathingsignal
    # dict_data['respiratory_frequency'] = frequency
    # dict_data['power_spectrum_density'] = psd
    return ts_measure


def _detrend(signal, Lambda):
    """detrend(signal, Lambda) -> filtered_signal
    This function applies a detrending filter.
    This code is based on the following article "An advanced detrending method with application
    to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
    *Parameters*
      ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
      ``Lambda`` (int):
        The smoothing parameter.
    *Returns*
      ``filtered_signal`` (1d numpy array):
        The detrended signal.
    """
    signal_length = signal.shape[0]

    # observation matrix
    H = np.identity(signal_length)

    # second-order difference matrix

    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)
    return filtered_signal


def calc_vital_values(peaklist, resp_peaks, fs):
    """Calculate vital values from given list of peaks.
    Returns BPM (beat per minutes), SI (stress index), HRV (heart rate variability)
    Parameters
    ----------
    peaklist : List[int]
        A list of peak indices
    fs : float
        Frames per seconds of the input buffer (or sampling rate).
    Returns
    -------
    List[float]
        A list of vital values.
    """
    # ===========================================
    # BPM
    RR_list = []
    cnt = 0

    while (cnt < (len(peaklist) - 1)):
        # Calculate distance between beats in # of samples
        RR_interval = (peaklist[cnt + 1] - peaklist[cnt])
        # Convert sample distances to ms distances
        ms_dist = int(((RR_interval / fs) * 1000.0))
        RR_list.append(ms_dist)  # Append to list
        cnt += 1

    # 60000 ms (1 minute) / average R-R interval of signal
    bpm = 60000 / np.mean(RR_list)
    # ===========================================

    # ===========================================
    # Invoke Kubios's API to get SNS index
    # ===========================================
    if len(RR_list) < KUBIOS_MIN_SAMPLES:
        num_pad = KUBIOS_MIN_SAMPLES - len(RR_list)
        RR_list += RR_list[-num_pad:]
    if len(RR_list) < KUBIOS_MIN_SAMPLES:
        num_pad = KUBIOS_MIN_SAMPLES - len(RR_list)
        RR_list += RR_list[-num_pad:]
    sns_index = predict_kubios(RR_list)

    # ===========================================
    # Similarly, calculate the resp rate
    resp_RR_list = []
    cnt = 0

    while (cnt < (len(resp_peaks) - 1)):
        # Calculate distance between beats in # of samples
        resp_RR_interval = (resp_peaks[cnt + 1] - resp_peaks[cnt])
        # Convert sample distances to ms distances
        ms_dist = int(((resp_RR_interval / fs) * 1000.0))
        resp_RR_list.append(ms_dist)  # Append to list
        cnt += 1

    # 60000 ms (1 minute) / average R-R interval of signal
    resp_rate = 60000 / np.mean(resp_RR_list)
    if isNaN(resp_rate):
        ts_measure = calc_resp_rate(RR_list)
        resp_rate = ts_measure['respiratory_rate']
    # ===========================================

    # ===========================================
    # Calculate HRV using RMSSD method.
    # Ref: https://imotions.com/blog/heart-rate-variability/
    # HRV = sqrt(mean((RR1 - RR2) ^ 2 + (RR2 - RR3) ^ 2 + ...))
    RR_list = np.array(RR_list)
    hrv = np.sqrt(np.square(np.diff(RR_list)).mean())

    # Ref: https://www.kubios.com/hrv-analysis-methods/
    # "In order to make SI less sensitive to slow changes in mean heart rate
    # (which would increase the MxDMn and lower AMo), the very low
    # frequency trend is removed from the RR interval time series
    # by using the smoothness priors method Tarvainen"
    # where, regularization = lambda (read the paper for the details)
    RR_list = nk.signal_detrend(
        RR_list, method='tarvainen2002', regularization=300)

    # Quantize the RR-intervals with bin size = 50ms
    bin_size = 50
    RR_bin_list = np.ceil(RR_list / bin_size)
    mode, most_freqs = Counter(RR_bin_list).most_common(1)[0]
    mo = np.median(RR_list) / 1000  # ms -> seconds
    max_RR_interval = np.max(RR_list)
    min_RR_interval = np.min(RR_list)
    mxDmn = (max_RR_interval - min_RR_interval + 1e-12) / \
        1000.  # ms -> seconds
    si = 100 * (most_freqs / len(RR_list)) / (2 * mo * mxDmn)
    si = np.sqrt(si)
    # ===========================================

    return bpm, hrv, si, sns_index, resp_rate


def postprocess(preds: np.ndarray, fps: float, user_info: dict = None) -> list[float]:
    """
    """
    pulse_pred = preds[0]
    pulse_pred = _detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(
        1, [0.75 / fps * 2, 2.5 / fps * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = preds[1]
    resp_pred = _detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(
        1, [0.08 / fps * 2, 0.5 / fps * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    # calculate peaks
    pulse_peaks, _ = find_peaks(pulse_pred, distance=15)
    resp_peaks, _ = find_peaks(resp_pred, distance=70)
    bpm, hrv, si, sns_index, resp_rate = calc_vital_values(
        pulse_peaks, resp_peaks, fps)

    # calculate blood pressure if user_info is given
    bp_calc = BloodCalculator()
    sp, dp = bp_calc.predict(pulse_pred, fps)
    # if user_info is not None:
    #     user_age = user_info['age']
    #     user_weight = user_info['weight']
    #     user_height = user_info['height']
    #     user_gender = user_info['gender']  # male or female

    #     # Constants
    #     ROB = 18.5  # Resistance to blood flow
    #     Q = 4.5  # Cardiac output  4.5 L/min for female and 5 L/min for male
    #     if user_gender == 'male':
    #         Q = 5

    #     ET = (364.5 - 1.23 * bpm)  # Ejection time (ms)
    #     BSA = 0.007184 * \
    #         (math.pow(user_weight, 0.425)) * \
    #         (math.pow(user_height, 0.725))  # Body surface area
    #     SV = (-6.6 + (0.25 * (ET - 35)) -
    #           (0.62 * bpm) + (40.4 * BSA) - (0.51 * user_age))  # Stroke volume
    #     PP = SV / ((0.013 * user_weight - 0.007 *
    #                user_age - 0.004 * bpm) + 1.307)  # Pulse pressure
    #     MPP = Q * ROB

    #     SP = int(MPP + 3 / 2 * PP)  # Systolic pressure (top number)
    #     DP = int(MPP - PP / 3)  # Diastolic pressure (bottom number)

    return bpm, hrv, si, sns_index, resp_rate, sp, dp


class BloodCalculator:
    def __init__(self):
        """Ref: https://github.com/Fabian-Sc85/non-invasive-bp-estimation-using-deep-learning
        """
        custom_objects = {
            'ReLU': tf.keras.layers.ReLU,
            'STFT': STFT,
            'Magnitude': Magnitude,
            'MagnitudeToDecibel': MagnitudeToDecibel
        }

        self.model = tf.keras.models.load_model(
            config.bp_model.model_path,
            custom_objects=custom_objects
        )
        self.sampling_rate = 125
        self.max_samples = 875

    def predict(self, pulse_sig, fp):
        num_samples = int(pulse_sig.shape[0] * self.sampling_rate / fp)
        resampled_pulse = resample(
            pulse_sig, num_samples)
        resampled_len = resampled_pulse.shape[0]
        if resampled_len < self.max_samples:
            resampled_pulse = np.pad(
                resampled_pulse, [[0, self.max_samples - resampled_len]])
        resampled_pulse = resampled_pulse[:self.max_samples]
        x = resampled_pulse[None, :, None]
        outputs = self.model.predict(x)
        sp = np.squeeze(outputs[0])
        dp = np.squeeze(outputs[1])

        sp = np.clip(sp, 20, 200)
        dp = np.clip(dp, 20, 200)
        return sp, dp


class VitalSignsCalculator:
    def __init__(self, model):
        self.model = model

    def predict(self, inputs) -> VitalSignsResult:
        dXsub = inputs['dXsub']
        fps = inputs['fps']
        green_avg_list = inputs['green_avg_list']
        red_avg_list = inputs['red_avg_list']
        blue_avg_list = inputs['blue_avg_list']
        user_info = inputs.get('user_info')
        sum_red = np.sum(red_avg_list)
        sum_blue = np.sum(blue_avg_list)

        preds = self.model.predict(
            (dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=config.model.args.batch_size)
        bpm, hrv, si, sns_index, resp_rate, sp, dp = postprocess(
            preds, fps, user_info)

        counter = len(green_avg_list)

        # calculating the mean of red and blue intensities on the whole period of recording
        mean_r = sum_red / counter
        mean_b = sum_blue / counter
        var_b = 0
        var_r = 0
        for i in range(counter - 1):
            buffer_b = blue_avg_list[i]
            var_b += (buffer_b - mean_b) ** 2
            buffer_r = red_avg_list[i]
            var_r += (buffer_r - mean_r) ** 2

        # calculating the variance
        std_r = math.sqrt(var_r / (counter - 1))
        std_b = math.sqrt(var_b / (counter - 1))

        # calculating ratio between the two means and two variances
        ratio = (std_r / mean_r) / (std_b / mean_b)

        # Estimating SPo2
        spo2 = 100 - 5 * ratio
        o2 = int(spo2)

        # return breath, spo2, o2
        return VitalSignsResult(
            bpm=bpm,
            hrv=hrv,
            si=si,
            sns_index=sns_index,
            resp_rate=resp_rate,
            sp=sp,
            dp=dp,
            spo2=spo2,
            o2=o2
        )
