import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import median_filter as gp_medfilt
from cupyx.scipy import signal
from scipy.signal import firwin, detrend
from helpers import modulation_index
from scipy.signal.windows import tukey

import pywt
from cupyx.scipy.ndimage import gaussian_filter


NPERSEG = 512
DEBUG = True


def compute_raw_psd(data_values, fs):
    """
    Expects: (subject, trial, channel, time)
    Returns: (subject, channel, freq)
    """
    n_samples = data_values.shape[-1]
    win = np.hamming(n_samples);
    win_energy = np.sum(win**2)
    
    windowed = data_values * win
    fft_vals = np.fft.rfft(windowed, axis=-1)
    
    psd = (np.abs(fft_vals)**2) / (fs * win_energy)
    psd[..., 1:-1] *= 2 
    
    return np.nanmean(psd, axis=1)

def apply_median_filter(psd_values, size=(1, 1, 1)):
    """
    Expects: (subject, channel, freq)
    Returns: (subject, channel, freq) filtered
    """
    psd_cp = cp.asarray(psd_values)
    filtered_cp = gp_medfilt(psd_cp, size=size)
    return cp.asnumpy(filtered_cp)

def normalize_psd(psd_values):
    """
    Expects: (subject, channel, freq)
    Returns: Normalized array (sums to 1 across freq axis)
    """
    # Sum along freq axis (last dimension)
    total_power = np.nansum(psd_values, axis=-1, keepdims=True)
    return psd_values / total_power




def debug_print(msg):
    if DEBUG:
        print(f"[DEBUG] {msg}")

# --- 1D Methods ---

def compute_raw_fft_psd(data_values, fs):
    debug_print(f"ENTERING: compute_medfiltered_fft_psd | Input: {data_values.shape}")
    n_samples = data_values.shape[-1]
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    
    psd_mean = compute_raw_psd(data_values, fs)
    debug_print(f"After raw_psd (trial mean): {psd_mean.shape}")
    
    debug_print(f"EXITING: compute_medfiltered_fft_psd | Output: {psd_mean.shape}")
    return freqs, psd_mean



def compute_medfiltered_fft_psd(data_values, fs, filter_size=(1, 1, 1)):
    debug_print(f"ENTERING: compute_medfiltered_fft_psd | Input: {data_values.shape}")
    n_samples = data_values.shape[-1]
    freqs = np.fft.rfftfreq(n_samples, 1/fs)
    
    psd_mean = compute_raw_psd(data_values, fs)
    debug_print(f"After raw_psd (trial mean): {psd_mean.shape}")
    
    processed_psd = apply_median_filter(psd_mean, size=filter_size)
    total_power = np.nansum(processed_psd, axis=-1, keepdims=True) + 1e-12
    normalized_psd = processed_psd / total_power

    debug_print(f"EXITING: compute_medfiltered_fft_psd | Output: {normalized_psd.shape}")
    return freqs, normalized_psd

def compute_coherence_gpu(data_values, fs, add_noise=True, noise_level=1e-3):
    data_cp = cp.asarray(data_values)
    nperseg = NPERSEG 
    noverlap = nperseg // 2

    ch1 = data_cp[:, :, 0, :]
    ch2 = data_cp[:, :, 1, :]
    
    if add_noise:
        rms1 = cp.sqrt(cp.mean(ch1**2, axis=-1, keepdims=True))
        rms2 = cp.sqrt(cp.mean(ch2**2, axis=-1, keepdims=True))
        ch1 = ch1 + cp.random.normal(0, noise_level, ch1.shape, dtype=ch1.dtype) * rms1
        ch2 = ch2 + cp.random.normal(0, noise_level, ch2.shape, dtype=ch2.dtype) * rms2
    
    freqs, coh = signal.coherence(ch1, ch2, fs=fs, window='hamming', 
                                  nperseg=nperseg, noverlap=noverlap, axis=-1)
    
    m_coh = cp.nanmean(coh, axis=1)
    output = cp.asnumpy(m_coh[:, cp.newaxis, :])
    return cp.asnumpy(freqs), output

def compute_imaginary_coherence_gpu(data_values, fs, add_noise=True, noise_level=1e-3):
    data_cp = cp.asarray(data_values)
    nperseg = NPERSEG
    noverlap = nperseg // 2

    ch1 = data_cp[:, :, 0, :]
    ch2 = data_cp[:, :, 1, :]
    
    if add_noise:
        rms1 = cp.sqrt(cp.mean(ch1**2, axis=-1, keepdims=True))
        rms2 = cp.sqrt(cp.mean(ch2**2, axis=-1, keepdims=True))
        ch1 = ch1 + cp.random.normal(0, noise_level, ch1.shape, dtype=ch1.dtype) * rms1
        ch2 = ch2 + cp.random.normal(0, noise_level, ch2.shape, dtype=ch2.dtype) * rms2
    
    freqs, csd = signal.csd(ch1, ch2, fs=fs, window='hamming', 
                        nperseg=nperseg, noverlap=noverlap, axis=-1)
    
    _, psd1 = signal.welch(ch1, fs=fs, window='hamming', 
                           nperseg=nperseg, noverlap=noverlap, axis=-1)
    _, psd2 = signal.welch(ch2, fs=fs, window='hamming', 
                           nperseg=nperseg, noverlap=noverlap, axis=-1)
    
    icoh_trials = cp.imag(csd / cp.sqrt(psd1 * psd2))
    icoh = cp.nanmean(cp.abs(icoh_trials), axis=1)
    output = cp.asnumpy(icoh[:, cp.newaxis, :])
    return cp.asnumpy(freqs), output


def compute_cross_spectrum_gpu(data_values, fs):
    debug_print(f"ENTERING: compute_cross_spectrum_gpu | Input: {data_values.shape}")
    data_cp = cp.asarray(data_values)
    nperseg = NPERSEG
    noverlap = nperseg // 2

    ch1 = data_cp[:, :, 0, :]
    ch2 = data_cp[:, :, 1, :]
    
    freqs, csd = signal.csd(ch1, ch2, fs=fs, window='hamming', 
                            nperseg=nperseg, noverlap=noverlap, axis=-1)
    
    m_csd = cp.nanmean(csd, axis=1)
    debug_print(f"After CSD (trial mean): {m_csd.shape}")
    
    total_power = cp.nansum(m_csd, axis=-1, keepdims=True) + 1e-12
    m_csd = m_csd / total_power 
    output = cp.asnumpy(cp.abs(m_csd)[:, cp.newaxis, :])

    debug_print(f"EXITING: compute_cross_spectrum_gpu | Output: {output.shape}")
    return cp.asnumpy(freqs), output


def compute_granger_gpu(data_values, fs, order=50, n_freqs=256):
    debug_print(f"ENTERING: compute_granger_gpu | Input: {data_values.shape}")
    data_cp = cp.asarray(data_values)
    n_subjects, n_trials, n_ch, n_times = data_cp.shape
    freqs = cp.linspace(0, fs / 2, n_freqs)
    g12_subs = cp.zeros((n_subjects, n_freqs))
    g21_subs = cp.zeros((n_subjects, n_freqs))
    for s in range(n_subjects):
        sub_raw = data_cp[s]
        mask = ~cp.any(cp.isnan(sub_raw), axis=(1, 2))
        valid_trials = sub_raw[mask]
        if valid_trials.shape[0] == 0: continue
        n_v_trials = valid_trials.shape[0]
        v_data = valid_trials - cp.mean(valid_trials, axis=-1, keepdims=True)
        n_samples_per_trial = n_times - order
        g12_trial_acc = cp.zeros(n_freqs)
        g21_trial_acc = cp.zeros(n_freqs)
        for t in range(n_v_trials):
            y_trial = v_data[t, :, order:].T
            x_trial = cp.zeros((n_samples_per_trial, n_ch * order))
            for i in range(order):
                x_trial[:, i*n_ch:(i+1)*n_ch] = v_data[t, :, order-i-1:n_times-i-1].T
            coeffs, _, _, _ = cp.linalg.lstsq(x_trial, y_trial, rcond=None)
            residuals = y_trial - cp.matmul(x_trial, coeffs)
            sigma = cp.matmul(residuals.T, residuals) / (n_samples_per_trial - (n_ch * order))
            A_k = coeffs.T.reshape(n_ch, n_ch, order, order='F')
            exp_phase = cp.exp(-1j * 2 * cp.pi * freqs[:, None] * cp.arange(1, order + 1) / fs)
            A_f = cp.einsum('ijk,fk->fij', A_k, exp_phase)
            I = cp.eye(n_ch, dtype=cp.complex128)[None, :, :]
            H = cp.linalg.inv(I - A_f)
            S = cp.matmul(cp.matmul(H, sigma[None, :, :]), H.transpose(0, 2, 1).conj())
            sig11, sig22, sig12 = sigma[0, 0].real, sigma[1, 1].real, sigma[0, 1].real
            gam11 = sig11 - (sig12**2 / sig22)
            gam22 = sig22 - (sig12**2 / sig11)
            s11, s22 = S[:, 0, 0].real, S[:, 1, 1].real
            g12_trial_acc += cp.log(s22 / (cp.abs(s22 - gam11 * cp.abs(H[:, 1, 0])**2) + 1e-15))
            g21_trial_acc += cp.log(s11 / (cp.abs(s11 - gam22 * cp.abs(H[:, 0, 1])**2) + 1e-15))
        g12_subs[s] = g12_trial_acc / n_v_trials
        g21_subs[s] = g21_trial_acc / n_v_trials
    out_g21 = cp.asnumpy(g21_subs)
    out_g12 = cp.asnumpy(g12_subs)
    debug_print(f"EXITING: compute_granger_gpu | Returning Tuple. Shapes: {out_g21.shape}, {out_g12.shape}")
    return cp.asnumpy(freqs), (out_g21, out_g12)

# --- 2D Methods ---

def compute_psd_ratio_matrix(data_values, fs):
    debug_print(f"ENTERING: compute_psd_ratio_matrix | Input: {data_values.shape}")
    freqs, norm_psd = compute_medfiltered_fft_psd(data_values, fs)
    
    row_psd = norm_psd[..., :, np.newaxis]
    col_psd = norm_psd[..., np.newaxis, :]
    debug_print(f"Internal Reshape - Rows: {row_psd.shape} Cols: {col_psd.shape}")
    
    ratio_matrix = row_psd / (col_psd + 1e-12)
    debug_print(f"EXITING: compute_psd_ratio_matrix | Output: {ratio_matrix.shape}")
    return freqs, ratio_matrix


def get_pactools_taps(fs, fc, bandwidth, n_cycles=None):
    if n_cycles is None:
        n_cycles = 1.65 * fc / bandwidth
    half_order = int(n_cycles * fs / fc / 2)
    order = 2 * half_order
    taps = firwin(order + 1, [fc - bandwidth/2, fc + bandwidth/2], 
                  pass_zero=False, fs=fs, window='hamming')
    return taps, order

def compute_PAC_2D(data_values, fs, low_fq_width=4.0, zpac=False, n_surrogates=100):
    debug_print(f"ENTERING: compute_PAC_2D | Input: {data_values.shape} | Z-PAC: {zpac}")
    f_low, f_high = (4, 14), (30, 100)
    n_bins = 50
    n_subs, n_trials, n_ch, n_time = data_values.shape
    low_freqs = np.linspace(f_low[0], f_low[1], n_bins)
    high_freqs = np.linspace(f_high[0], f_high[1], n_bins)
    
    final_output = np.zeros((n_subs, n_ch * n_ch, n_bins, n_bins))
    raw_data = data_values.values 
    win_gpu = cp.array(tukey(n_time, alpha=0.1), dtype=cp.float64)
    
    h = cp.zeros(n_time, dtype=cp.float64)
    h[0] = 1
    h[1:n_time//2 + (1 if n_time%2 != 0 else 0)] = 2
    if n_time % 2 == 0: h[n_time // 2] = 1
    
    shift_min, shift_max = int(0.2 * fs), n_time - int(0.2 * fs)
    
    for s_idx in range(n_subs):
        sub_raw = raw_data[s_idx]
        trial_mask = ~np.any(np.isnan(sub_raw), axis=(1, 2))
        valid_trials = sub_raw[trial_mask]
        if valid_trials.shape[0] == 0:
            debug_print(f"Sub {s_idx}: No valid trials.")
            continue

        subject_detrended = detrend(valid_trials, axis=-1)
        subject_gpu = cp.array(subject_detrended, dtype=cp.float64)
        subject_gpu -= subject_gpu.mean(axis=-1, keepdims=True)
        
        phases = extract_phases_batch(subject_gpu, fs, low_freqs, low_fq_width, h)
        amplitudes = extract_amplitudes_batch(subject_gpu, fs, high_freqs, low_freqs.max(), h)
        debug_print(f"Sub {s_idx} | Phases: {phases.shape} | Amps: {amplitudes.shape}")

        def get_pac_matrix(p_in, a_in):
            p_win, a_win = p_in * win_gpu, a_in * win_gpu
            norm_a = cp.linalg.norm(a_win, axis=-1)
            res = cp.abs(cp.einsum('btin, mtjn -> bmtij', p_win, a_win))
            denom = (cp.sqrt(n_time) * norm_a)[cp.newaxis, :, :, cp.newaxis, :]
            return res / (denom + 1e-12)
        
        pac_real = get_pac_matrix(phases, amplitudes)
        debug_print(f"Sub {s_idx} | Raw PAC Shape: {pac_real.shape}") # Expect (50, 50, n_trials, 2, 2)

        if zpac:
            sum_surr = cp.zeros_like(pac_real)
            sq_sum_surr = cp.zeros_like(pac_real)
            
            for _ in range(n_surrogates):
                shift = np.random.randint(shift_min, shift_max)
                
                pac_surr = get_pac_matrix(phases, cp.roll(amplitudes, shift, axis=-1))
                
                sum_surr += pac_surr
                sq_sum_surr += pac_surr**2
            
            mu = sum_surr / n_surrogates
            std = cp.sqrt((sq_sum_surr / n_surrogates) - mu**2)
            pac_final = (pac_real - mu) / (std + 1e-12)
        else:
            pac_final = pac_real

        # 1. Average across the TRIAL dimension (axis=2)
        # Result: (n_low, n_high, n_ch, n_ch)
        subject_avg = cp.nanmean(pac_final, axis=2)
        debug_print(f"Sub {s_idx} | After Trial Mean: {subject_avg.shape}")

        # 2. Reshape to flatten channel combinations: (n_low, n_high, n_ch*n_ch)
        subject_avg = subject_avg.reshape(n_bins, n_bins, n_ch * n_ch)
        
        # 3. Transpose to move channels to the front: (n_ch*n_ch, n_low, n_high)
        # This matches your final_output expectations
        subject_avg = subject_avg.transpose(2, 0, 1)
        debug_print(f"Sub {s_idx} | Final Subject Shape: {subject_avg.shape}")

        final_output[s_idx] = cp.asnumpy(subject_avg)
        
    debug_print(f"EXITING: compute_PAC_2D | Output: {final_output.shape}")
    return (low_freqs, high_freqs), final_output

def extract_phases_batch(data_gpu, fs, freqs, width, h):
    n_bins = len(freqs)
    n_trials, n_ch, n_time = data_gpu.shape
    phases = cp.zeros((n_bins, n_trials, n_ch, n_time), dtype=cp.complex128)
    for i, f in enumerate(freqs):
        taps, order = get_pactools_taps(fs, f, width)
        taps_gpu = cp.array(taps, dtype=cp.float64)
        filtered = cp.apply_along_axis(lambda m: cp.convolve(m, taps_gpu, mode='same'), axis=-1, arr=data_gpu)
        f_sig = cp.fft.fft(filtered, axis=-1)
        analytic = cp.fft.ifft(f_sig * h, axis=-1)
        res = analytic / (cp.abs(analytic) + 1e-12)
        phases[i] = cp.roll(res, -(order // 2), axis=-1)
    return phases

def extract_amplitudes_batch(data_gpu, fs, freqs, low_fq_max, h):
    n_bins = len(freqs)
    n_trials, n_ch, n_time = data_gpu.shape
    amplitudes = cp.zeros((n_bins, n_trials, n_ch, n_time), dtype=cp.float64)
    high_width = 2.0 * low_fq_max
    for i, f in enumerate(freqs):
        taps, order = get_pactools_taps(fs, f, high_width)
        taps_gpu = cp.array(taps, dtype=cp.float64)
        filtered = cp.apply_along_axis(lambda m: cp.convolve(m, taps_gpu, mode='same'), axis=-1, arr=data_gpu)
        f_sig = cp.fft.fft(filtered, axis=-1)
        analytic = cp.fft.ifft(f_sig * h, axis=-1)
        res = cp.abs(analytic)
        amplitudes[i] = cp.roll(res, -(order // 2), axis=-1)
    return amplitudes


def compute_spectrogram_2D(data_values, fs):
    debug_print(f"ENTERING: compute_spectrogram_2D | Input: {data_values.shape}")
    n_subs, n_trials, n_ch, n_time = data_values.shape
    nperseg = NPERSEG
    noverlap = nperseg // 2
    
    raw_data = cp.asarray(data_values.values)
    dummy_sig = raw_data[0, 0, 0, :]
    freqs_cp, times_cp, _ = signal.spectrogram(dummy_sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    n_freqs, n_times = len(freqs_cp), len(times_cp)
    final_output = cp.zeros((n_subs, n_ch, n_freqs, n_times))

    for s_idx in range(n_subs):
        sub_data_cp = raw_data[s_idx]
        sub_avg_cp = cp.zeros((n_ch, n_freqs, n_times))
        for ch_idx in range(n_ch):
            sigs = sub_data_cp[:, ch_idx, :]
            f_res, t_res, Sxx_cp = signal.spectrogram(sigs, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
            sub_avg_cp[ch_idx] = cp.nanmean(Sxx_cp, axis=0)
        final_output[s_idx] = sub_avg_cp

    output_np = cp.asnumpy(final_output)
    debug_print(f"EXITING: compute_spectrogram_2D | Output: {output_np.shape}")
    return (cp.asnumpy(freqs_cp), cp.asnumpy(times_cp)), output_np

"""
# ORIGINAL WORKING ONE
def compute_vectorized_cwt(da_values, fs, f_min=4, f_max=100, n_scales=100, wavelet_name='cmor5.5-1.5'):
    debug_print(f"ENTERING: compute_vectorized_cwt | Input: {da_values.shape}")
    n_subs, n_trials, n_ch, n_time = da_values.shape
    freqs = np.linspace(f_min, f_max, n_scales)
    Ts = 1.0 / fs
    wav_obj = pywt.ContinuousWavelet(wavelet_name)
    cf = pywt.central_frequency(wav_obj)
    scales = cf / (freqs * Ts)
    
    pad_amt = min(n_time // 2, 500) 
    data_gpu = cp.asarray(da_values.values)
    data_padded = cp.pad(data_gpu, ((0,0), (0,0), (0,0), (pad_amt, pad_amt)), mode='reflect')
    
    n_padded = data_padded.shape[-1]
    n_fft = int(2**np.ceil(np.log2(n_padded + 1000)))
    
    data_fft = cp.fft.fft(data_padded, n=n_fft, axis=-1)
    coefs = cp.empty((n_scales, n_subs, n_trials, n_ch, n_time), dtype=cp.complex64)
    
    int_psi_cpu, x = pywt.integrate_wavelet(wav_obj, precision=10)
    step = x[1] - x[0]

    for i, scale in enumerate(scales):
        j_cpu = np.arange(float(scale) * (x[-1] - x[0]) + 1) / (float(scale) * step)
        j_cpu = j_cpu.astype(np.int32)
        j_cpu = j_cpu[j_cpu < int_psi_cpu.size]
        
        wav_kernel_cpu = np.conj(int_psi_cpu[j_cpu][::-1])
        wav_fft = cp.fft.fft(cp.asarray(wav_kernel_cpu), n=n_fft)
        
        conv = cp.fft.ifft(data_fft * wav_fft, axis=-1)
        
        res = - cp.sqrt(scale) * cp.diff(conv[..., :n_padded + 1], axis=-1)
        coefs[i] = res[..., pad_amt : pad_amt + n_time]

    out = coefs.transpose(1, 2, 3, 0, 4)
    debug_print(f"EXITING: compute_vectorized_cwt | Output Coefs: {out.shape}")
    return freqs, cp.arange(n_time) * Ts, out
"""
import numpy as np
import cupy as cp
import pywt
from math import ceil, floor
import numpy as np
import cupy as cp
import pywt
from math import ceil, floor

def compute_vectorized_cwt(da_values, fs, f_min=4, f_max=100, n_scales=100, wavelet_name='cmor3.5-1.5'):
    debug_print(f"ENTERING: compute_vectorized_cwt | Input: {da_values.shape}")
    n_subs, n_trials, n_ch, n_time = da_values.shape
    freqs = np.linspace(f_min, f_max, n_scales)
    Ts = 1.0 / fs
    wav_obj = pywt.ContinuousWavelet(wavelet_name)
    cf = pywt.central_frequency(wav_obj)
    scales = cf / (freqs * Ts)
    
    pad_amt = min(n_time // 2, 500) 
    data_gpu = cp.asarray(da_values.values, dtype=cp.float32)
    data_padded = cp.pad(data_gpu, ((0,0), (0,0), (0,0), (pad_amt, pad_amt)), mode='reflect')
    n_padded = data_padded.shape[-1]
    
    coefs = cp.empty((n_scales, n_subs, n_trials, n_ch, n_time), dtype=cp.complex64)
    
    int_psi_cpu, x = pywt.integrate_wavelet(wav_obj, precision=10)
    step = x[1] - x[0]

    for i, scale in enumerate(scales):
        j_cpu = np.arange(float(scale) * (x[-1] - x[0]) + 1) / (float(scale) * step)
        j_cpu = j_cpu.astype(np.int32)
        j_cpu = j_cpu[j_cpu < int_psi_cpu.size]
        
        wav_kernel_cpu = np.conj(int_psi_cpu[j_cpu][::-1])
        n_kernel = wav_kernel_cpu.size
        
        n_fft = int(2**np.ceil(np.log2(n_padded + n_kernel - 1)))
        
        wav_fft = cp.fft.fft(cp.asarray(wav_kernel_cpu, dtype=cp.complex64), n=n_fft)
        data_fft = cp.fft.fft(data_padded, n=n_fft, axis=-1)
        
        conv = cp.fft.ifft(data_fft * wav_fft, axis=-1)
        
        res = - cp.sqrt(scale) * cp.diff(conv[..., :n_fft], axis=-1)
        
        d = (n_kernel - 1) / 2.0
        start_idx = int(floor(d)) + pad_amt
        
        coefs[i] = res[..., start_idx : start_idx + n_time]
        
        del conv, res, data_fft, wav_fft
        cp.get_default_memory_pool().free_all_blocks()

    out = coefs.transpose(1, 2, 3, 0, 4)
    debug_print(f"EXITING: compute_vectorized_cwt | Output Coefs: {out.shape}")
    return freqs, cp.arange(n_time) * Ts, out

def compute_cwt_scalogram_vectorized(da, fs, **kwargs):
    debug_print(f"ENTERING: compute_cwt_scalogram_vectorized | Input: {da.shape}")
    freqs, ts, coefs = compute_vectorized_cwt(da, fs, **kwargs)
    power = cp.abs(coefs)**2
    avg_power = cp.nanmean(power, axis=1)
    output_np = cp.asnumpy(avg_power)
    debug_print(f"EXITING: compute_cwt_scalogram_vectorized | Output: {output_np.shape}")
    return (np.array(freqs), cp.asnumpy(ts)), output_np

def compute_cwt_coh_vectorized(da, fs, sigma_t=10, sigma_f=10, **kwargs):
    debug_print(f"ENTERING: compute_cwt_coh_vectorized | Input: {da.shape}")
    freqs, ts, coefs = compute_vectorized_cwt(da, fs, **kwargs)
    c_a = coefs[:, :, 0, :, :]
    c_b = coefs[:, :, 1, :, :]
    s_xy = cp.nanmean(c_a * cp.conj(c_b), axis=1)
    s_xx = cp.nanmean(cp.abs(c_a)**2, axis=1)
    s_yy = cp.nanmean(cp.abs(c_b)**2, axis=1)
    debug_print(f"MID: compute_cwt_coh_vectorized | Cross-spec averaged: {s_xy.shape}")
    
    coh_res = cp.zeros_like(s_xy, dtype=cp.float32)
    for s in range(s_xy.shape[0]):
        smooth_xy = (gaussian_filter(s_xy[s].real, (sigma_f, sigma_t)) + 
                     1j * gaussian_filter(s_xy[s].imag, (sigma_f, sigma_t)))
        smooth_xx = gaussian_filter(s_xx[s].real, (sigma_f, sigma_t))
        smooth_yy = gaussian_filter(s_yy[s].real, (sigma_f, sigma_t))
        coh = smooth_xy / (cp.sqrt(smooth_xx * smooth_yy) + 1e-6)
        coh_res[s] = cp.abs(coh)
        
    output_np = cp.asnumpy(coh_res[:, np.newaxis, :, :])
    debug_print(f"EXITING: compute_cwt_coh_vectorized | Output: {output_np.shape}")
    return (np.array(freqs), cp.asnumpy(ts)), output_np

def compute_cwt_icoh_vectorized(da, fs, sigma_t=10, sigma_f=10, **kwargs):
    debug_print(f"ENTERING: compute_cwt_icoh_vectorized | Input: {da.shape}")
    freqs, ts, coefs = compute_vectorized_cwt(da, fs, **kwargs)
    c_a = coefs[:, :, 0, :, :]
    c_b = coefs[:, :, 1, :, :]
    s_xy = cp.nanmean(c_a * cp.conj(c_b), axis=1)
    s_xx = cp.nanmean(cp.abs(c_a)**2, axis=1)
    s_yy = cp.nanmean(cp.abs(c_b)**2, axis=1)
    debug_print(f"MID: compute_cwt_icoh_vectorized | Cross-spec averaged: {s_xy.shape}")
    
    icoh_res = cp.zeros_like(s_xy, dtype=cp.float32)
    for s in range(s_xy.shape[0]):
        smooth_xy = (gaussian_filter(s_xy[s].real, (sigma_f, sigma_t)) + 
                     1j * gaussian_filter(s_xy[s].imag, (sigma_f, sigma_t)))
        smooth_xx = gaussian_filter(s_xx[s].real, (sigma_f, sigma_t))
        smooth_yy = gaussian_filter(s_yy[s].real, (sigma_f, sigma_t))
        coh = smooth_xy / (cp.sqrt(smooth_xx * smooth_yy) + 1e-6)
        icoh_res[s] = cp.imag(coh)
        
    output_np = cp.asnumpy(icoh_res[:, np.newaxis, :, :])
    debug_print(f"EXITING: compute_cwt_icoh_vectorized | Output: {output_np.shape}")
    return (np.array(freqs), cp.asnumpy(ts)), output_np