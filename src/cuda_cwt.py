import cupy as cp
import numpy as np
import pywt
from pywt._functions import integrate_wavelet
from scipy.fft import next_fast_len
from math import ceil, floor

from cupyx.scipy import ndimage


def compute_cwt_matrix_gpu(data_np, fs, f_min=1, f_max=100, n_scales=100, wavelet="cmor1.5-1.5", precision=12):
    """
    Returns a cupy array !!!
    """
    # 1. Setup metadata
    n_channels, n_samples = data_np.shape
    Ts = 1.0 / fs
    wav_obj = pywt.DiscreteContinuousWavelet(wavelet)
    
    # 2. Match scale logic
    cf = pywt.central_frequency(wav_obj)
    scales = np.geomspace(cf/(f_max*Ts), cf/(f_min*Ts), num=n_scales)
    frequencies = cf / (scales * Ts)

    # 3. Get Integrated Wavelet
    int_psi, x = pywt.integrate_wavelet(wav_obj, precision=precision)
    if wav_obj.complex_cwt:
        int_psi = np.conj(int_psi)
    
    data_gpu = cp.asarray(data_np, dtype=cp.float64)
    int_psi_gpu = cp.asarray(int_psi)
    
    dt_out = cp.complex128 if wav_obj.complex_cwt else cp.float64
    out = cp.empty((n_scales, n_channels, n_samples), dtype=dt_out)

    x_range = x[-1] - x[0]
    step = x[1] - x[0]

    for i, scale in enumerate(scales):
        # --- Resampling ---
        j = cp.arange(scale * x_range + 1) / (scale * step)
        j = j.astype(cp.int32)
        if j[-1] >= int_psi_gpu.size:
            j = j[j < int_psi_gpu.size]
        
        int_psi_scale = int_psi_gpu[j][::-1]
        
        # --- FFT Convolution ---
        conv_len = n_samples + int_psi_scale.size - 1
        size_scale = int(2**ceil(np.log2(conv_len))) 
        
        fft_data = cp.fft.fft(data_gpu, n=size_scale, axis=-1)
        fft_wav = cp.fft.fft(int_psi_scale, n=size_scale, axis=-1)
        
        conv = cp.fft.ifft(fft_wav * fft_data, axis=-1)
        conv = conv[..., :conv_len]
        
        # --- Derivative and Scale ---
        coef = - cp.sqrt(scale) * cp.diff(conv, axis=-1)
        
        # --- Centering / Slicing ---
        d = (coef.shape[-1] - n_samples) / 2.0
        if d > 0:
            start = floor(d)
            coef = coef[..., start : start + n_samples]
            
        out[i] = coef

    return frequencies, cp.arange(n_samples) * Ts, out.transpose(1, 0, 2)


import cupy as cp
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import gaussian_filter

def compute_complex_coherence_gpu(sig_a, sig_b, cwt_params, sigma_time=10, sigma_freq=5):
    if sig_a.ndim == 1: sig_a = sig_a[np.newaxis, :]
    if sig_b.ndim == 1: sig_b = sig_b[np.newaxis, :]

    freqs, ts, coefs_a = compute_cwt_matrix_gpu(sig_a, **cwt_params)
    freqs, ts, coefs_b = compute_cwt_matrix_gpu(sig_b, **cwt_params)

    c_a = coefs_a[0]
    c_b = coefs_b[0]

    raw_s_xy = c_a * cp.conj(c_b)
    raw_s_xx = cp.abs(c_a)**2
    raw_s_yy = cp.abs(c_b)**2

    s_xy = gaussian_filter(raw_s_xy.real, sigma=(sigma_freq, sigma_time)) + \
           1j * gaussian_filter(raw_s_xy.imag, sigma=(sigma_freq, sigma_time))
    
    s_xx = gaussian_filter(raw_s_xx, sigma=(sigma_freq, sigma_time))
    s_yy = gaussian_filter(raw_s_yy, sigma=(sigma_freq, sigma_time))

    complex_coh = s_xy / (cp.sqrt(s_xx * s_yy) + 1e-3)

    return freqs, ts, complex_coh

def compute_msc_from_complex(complex_coh):
    return cp.abs(complex_coh)**2

def compute_icoh_from_complex(complex_coh):
    return cp.imag(complex_coh)

# BELOW IS ALL LIKELY WORTHLESS JUNK

"""
def compute_session_average_cwt(trials_da, fs, **cwt_kwargs):

    n_trials, n_channels, n_samples = trials_da.shape
    n_scales = cwt_kwargs.get('n_scales', 100)
    
    gpu_power_sum = cp.zeros((n_channels, n_scales, n_samples), dtype=cp.float32)
    
    for i in range(n_trials):
        #print(f"Trial {i}")
        freqs, t_axis, cwt_gpu = compute_cwt_matrix_gpu(
            trials_da.values[i], 
            fs, 
            **cwt_kwargs
        )
        
        gpu_power_sum += cp.abs(cwt_gpu).astype(cp.float32)**2
        
        del cwt_gpu
        
    avg_power = (gpu_power_sum / n_trials).get()
    
    return freqs, t_axis.get(), avg_power


def compute_session_median_cwt(trials_da, fs, **cwt_kwargs):

    n_trials, n_channels, n_samples = trials_da.shape
    n_scales = cwt_kwargs.get('n_scales', 100)
    
    # Store all trials in VRAM to compute median: (trials, channels, scales, samples)
    gpu_power_stack = cp.empty((n_trials, n_channels, n_scales, n_samples), dtype=cp.float32)
    
    for i in range(n_trials):
        freqs, t_axis, cwt_gpu = compute_cwt_matrix_gpu(
            trials_da.values[i], 
            fs, 
            **cwt_kwargs
        )
        
        gpu_power_stack[i] = cp.abs(cwt_gpu).astype(cp.float32)**2
        
        del cwt_gpu
        
    median_power = cp.median(gpu_power_stack, axis=0).get()
    
    return freqs, t_axis.get(), median_power

def compute_all_trials_cwt(trials_da, fs=1000, **cwt_kwargs):
    #Loops through trials, computes CWT on GPU, and stores result in CPU RAM.
    #Handles both xarray DataArrays and raw NumPy arrays.
    # Check if it's xarray or numpy
    if hasattr(trials_da, 'values'):
        data_to_loop = trials_da.values
    else:
        data_to_loop = trials_da

    n_trials, n_channels, n_samples = data_to_loop.shape
    n_scales = cwt_kwargs.get('n_scales', 100)
    all_power = np.zeros((n_trials, n_channels, n_scales, n_samples), dtype=np.float32)
    
    for i in range(n_trials):
        trial_data = data_to_loop[i]
        
        freqs, t_axis, cwt_gpu = compute_cwt_matrix_gpu(
            trial_data, fs, **cwt_kwargs
        )
        
        all_power[i] = cp.abs(cwt_gpu).astype(cp.float32).get()
        
        del cwt_gpu
        cp.get_default_memory_pool().free_all_blocks() 

    return freqs, t_axis, all_power

def gpu_rank_sum_z(real_group, shuff_group):

    n1, n2 = real_group.shape[0], shuff_group.shape[0]
    N = n1 + n2
    combined = cp.concatenate([real_group, shuff_group], axis=0)
    
    # Ranking logic
    flat = combined.reshape(N, -1)
    ranks = cp.argsort(cp.argsort(flat, axis=0), axis=0).astype(cp.float32) + 1
    
    # U-Stat and Z-score
    rank_sum1 = cp.sum(ranks[:n1], axis=0)
    u1 = rank_sum1 - (n1 * (n1 + 1) / 2)
    mu_u = (n1 * n2) / 2.0
    sigma_u = cp.sqrt((n1 * n2 * (N + 1)) / 12.0)
    
    z = (u1 - mu_u) / sigma_u
    return z.reshape(real_group.shape[1:])

def generate_phase_shuffled_vram(trials_da):

    #Vectorized phase shuffling. Handles xarray or numpy/cupy inputs.

    # If it's an xarray, grab the values. If it's already an array, use it directly.
    data = trials_da.values if hasattr(trials_da, 'values') else trials_da
    
    n_trials, n_ch, n_samples = data.shape
    # Ensure we are working with numpy for the FFT to keep VRAM clear
    if hasattr(data, 'get'): # It's a cupy array
        data = data.get()
        
    fft_data = np.fft.rfft(data, axis=-1)
    phases = np.random.uniform(0, 2*np.pi, fft_data.shape)
    
    # Apply phase shift: exp(i * phases)
    # We use the magnitude of the original signal with the new random phases
    shuffled_fft = np.abs(fft_data) * np.exp(1j * phases)
    
    # Inverse FFT back to time domain
    shuffled_data = np.fft.irfft(shuffled_fft, n=n_samples, axis=-1)
    
    return shuffled_data.astype(np.float32)

def run_complete_cluster_analysis(trials_da, fs=1000, n_perms=1000, cluster_alpha=0.05, n_shuffs_per_trial=50, **cwt_kwargs):
    n_trials, n_ch, n_samples = trials_da.shape
    n_scales = cwt_kwargs.get('n_scales', 100)
    
    # 1. Compute Individual Trial Z-Maps (Pixel-wise Normalization)
    print(f"Computing Trial-by-Trial Pixel-wise Z-Maps ({n_shuffs_per_trial} shuffs/trial)...")
    all_z_maps = cp.zeros((n_trials, n_ch, n_scales, n_samples), dtype=cp.float32)
    power_real_stack = cp.zeros((n_trials, n_ch, n_scales, n_samples), dtype=cp.float32)
    
    for i in range(n_trials):
        trial = trials_da.values[i] if hasattr(trials_da, 'values') else trials_da[i]
        
        # Real Raw Power
        freqs, t_axis, cwt_real = compute_cwt_matrix_gpu(trial, fs, **cwt_kwargs)
        real_p = cp.abs(cwt_real).astype(cp.float32)
        power_real_stack[i] = real_p
        
        # Generate Pixel-wise Null Distribution for THIS trial
        trial_shuffs = cp.zeros((n_shuffs_per_trial, n_ch, n_scales, n_samples), dtype=cp.float32)
        for s in range(n_shuffs_per_trial):
            # Phase shuffle the raw signal
            shuff_raw = generate_phase_shuffled_vram(trial[None, ...])
            _, _, cwt_shuff = compute_cwt_matrix_gpu(shuff_raw[0], fs, **cwt_kwargs)
            trial_shuffs[s] = cp.abs(cwt_shuff).astype(cp.float32)
            
        m_null = cp.mean(trial_shuffs, axis=0)
        s_null = cp.std(trial_shuffs, axis=0) + 1e-9
        
        # Z-score this trial pixel-by-pixel
        all_z_maps[i] = (real_p - m_null) / s_null
        
        if i % 10 == 0:
            print(f" Finished Trial {i}/{n_trials}")

    # 2. Average Z-maps across trials (The Group Observed Statistic)
    obs_z_avg = cp.mean(all_z_maps, axis=0)
    
    # 3. Sign-Flipping Permutation Test
    print(f"Running Sign-Flip Permutation ({n_perms} iterations)...")
    z_threshold = 3.0  # Stiff threshold to maintain "Salience"
    final_results = []
    
    for ch in range(n_ch):
        ch_z_maps = all_z_maps[:, ch]
        max_densities = cp.zeros(n_perms)
        
        for p in range(n_perms):
            # Sign-flip entire trials
            signs = cp.random.choice(cp.array([-1, 1], dtype=cp.float32), size=(n_trials, 1, 1))
            perm_z = cp.mean(ch_z_maps * signs, axis=0)
            
            # Find the "Salience" (Cluster Mass/Density) of the biggest null cluster
            mask = perm_z > z_threshold
            labels, num = ndimage.label(mask)
            if num > 0:
                # Sum of Z-scores / Number of pixels = Density
                sums = ndimage.sum(perm_z, labels, cp.arange(num) + 1)
                sizes = ndimage.sum(mask.astype(float), labels, cp.arange(num) + 1)
                max_densities[p] = cp.max(sums / sizes)

        # Get the 95th (or 99th) percentile of Salience from the Null distribution
        thresh = np.percentile(max_densities.get(), 100 * (1 - cluster_alpha))
        sig_mask = np.zeros((n_scales, n_samples), dtype=bool)
        
        # Apply threshold to the real group average
        ch_obs_z = obs_z_avg[ch]
        obs_mask = ch_obs_z > z_threshold
        obs_labels, obs_num = ndimage.label(obs_mask)
        
        if obs_num > 0:
            obs_sums = ndimage.sum(ch_obs_z, obs_labels, cp.arange(obs_num) + 1)
            obs_sizes = ndimage.sum(obs_mask.astype(float), obs_labels, cp.arange(obs_num) + 1)
            obs_dens = obs_sums / obs_sizes
            for i, dens in enumerate(obs_dens):
                if dens > thresh:
                    sig_mask[obs_labels.get() == (i + 1)] = True
        
        final_results.append({'t_map': ch_obs_z.get(), 'sig_mask': sig_mask})
        cp.get_default_memory_pool().free_all_blocks()

    return freqs, t_axis, final_results, power_real_stack
"""