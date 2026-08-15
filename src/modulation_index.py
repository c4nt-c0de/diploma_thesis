import numpy as np
import cupy as cp
from scipy.signal import firwin, detrend
import matplotlib.pyplot as plt

def get_pactools_taps(fs, fc, bandwidth, n_cycles=None):
    if n_cycles is None:
        n_cycles = 1.65 * fc / bandwidth
    half_order = int(n_cycles * fs / fc / 2)
    order = 2 * half_order
    taps = firwin(order + 1, [fc - bandwidth/2, fc + bandwidth/2], 
                  pass_zero=False, fs=fs, window='hamming')
    return taps, order

def ozkurt_pac(signal, fs, f_low=(2, 14), f_high=(30,150), n_bins=50, low_fq_width=2.0):
    """
    signal : 1D array-like
        The input time-series data. Must be a single channel (1D). 
        Passing 2D arrays (e.g., [channels, time]) is currently not supported 
        and will require a loop or vectorized refactoring.
    fs : float
        The sampling frequency of the signal in Hz.
    f_low : tuple of (float, float), optional
        The range of frequencies (min, max) to be used for the phase-providing 
        modulator. Defaults to (2, 20).
    f_high : tuple of (float, float), optional
        The range of frequencies (min, max) to be used for the amplitude-providing 
        carrier. Defaults to (20, 150).
    n_bins : int, optional
        The number of frequency steps to compute for both the phase and amplitude 
        axes, resulting in an (n_bins, n_bins) comodulogram. Defaults to 50.
    low_fq_width : float, optional
        The bandwidth (Hz) of the FIR filters used for the phase range. 
        Higher values improve temporal resolution but reduce frequency 
        specificity. Defaults to 2.0.

    Returns
    -------
    mi_map : ndarray
        A 2D numpy array of shape (n_bins, n_bins) containing the Ozkurt 
        Modulation Index values.

    """
    n_pts = len(signal)
    epsilon = 1e-12
    
    signal = detrend(signal, type='linear')
    signal -= np.mean(signal)
    sig_gpu = cp.array(signal, dtype=cp.float64)

    low_fq_range = np.linspace(f_low[0], f_low[1], n_bins)
    high_fq_range = np.linspace(f_high[0], f_high[1], n_bins)

    h = cp.zeros(n_pts, dtype=cp.float64)
    if n_pts % 2 == 0:
        h[0] = h[n_pts // 2] = 1
        h[1:n_pts // 2] = 2
    else:
        h[0] = 1
        h[1:(n_pts + 1) // 2] = 2

    phases = cp.zeros((len(low_fq_range), n_pts), dtype=cp.complex128)
    for i, f in enumerate(low_fq_range):
        taps, order = get_pactools_taps(fs, f, low_fq_width)
        taps_gpu = cp.array(taps, dtype=cp.float64)
        filtered = cp.convolve(sig_gpu, taps_gpu, mode='same')
        f_sig = cp.fft.fft(filtered)
        analytic = cp.fft.ifft(f_sig * h)
        phases[i] = cp.roll(analytic / (cp.abs(analytic) + epsilon), - (order // 2))

    amplitudes = cp.zeros((len(high_fq_range), n_pts), dtype=cp.float64)
    high_width = 2.0 * low_fq_range.max() 
    for j, f in enumerate(high_fq_range):
        taps, order = get_pactools_taps(fs, f, high_width)
        taps_gpu = cp.array(taps, dtype=cp.float64)
        filtered = cp.convolve(sig_gpu, taps_gpu, mode='same')
        f_sig = cp.fft.fft(filtered)
        analytic = cp.fft.ifft(f_sig * h)
        amplitudes[j] = cp.abs(cp.roll(analytic, - (order // 2)))

    norm_a = cp.linalg.norm(amplitudes, axis=1)
    numerator = cp.abs(cp.matmul(phases, amplitudes.T))
    mi_map = numerator / (cp.sqrt(n_pts) * norm_a + epsilon)
    
    return cp.asnumpy(mi_map)


def plot_comodulogram(mi_map, f_low=(2, 14), f_high=(30, 100), title="Ozkurt PAC", cmap='turbo'):
    """
    Plots a PAC comodulogram with a forced square aspect ratio.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate extent based on the tuples
    extent = [f_low[0], f_low[1], f_high[0], f_high[1]]
    
    # Force 'auto' aspect so it fills the square regardless of bin counts
    im = ax.imshow(mi_map.T, origin='lower', extent=extent, aspect='auto', cmap=cmap)
    
    # Set axis labels
    ax.set_xlabel("Phase Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Amplitude Frequency (Hz)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Modulation Index')
    
    plt.tight_layout()
    return fig, ax


"""
Usage of the above. Seems reliable...


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from helpers import modulation_index

fs = 1000.
n_points = 10000 
n_trials = 50
f_low_target = 8.0
f_high_target = 80.0
f_low = (2, 20)
f_high = (40, 150)
n_bins = 100

accum_mi = np.zeros((n_bins, n_bins))

print(f"Running {n_trials} trials...")

for i in range(n_trials):
    t = np.arange(n_points) / fs
    phi = np.random.uniform(0, 2 * np.pi)
    
    phase_sig = np.sin(2 * np.pi * f_low_target * t + phi)
    modulator = (1 + phase_sig) / 2 
    carrier = np.sin(2 * np.pi * f_high_target * t)
    
    noise = np.random.normal(0, 0.8, n_points)
    test_signal = (carrier * modulator) + noise
    
    mi_map = modulation_index.ozkurt_pac(
        test_signal, 
        fs, 
        f_low=f_low, 
        f_high=f_high, 
        n_bins=n_bins
    )
    accum_mi += mi_map

avg_mi = accum_mi / n_trials

modulation_index.plot_comodulogram(
    avg_mi, 
    f_low=f_low, 
    f_high=f_high, 
    title=f"Averaged PAC ({n_trials} Trials): {f_low_target}Hz Phase / {f_high_target}Hz Amp"
)
plt.show()
"""


"""
###################################################################################################################################
The above reliably works.
Or at least I kinda trust it for now
Much cant be said about the below
"""


def ozkurt_pac_cross(sig_phase, sig_amp, fs, f_low=(2, 14), f_high=(30, 100), n_bins=100, low_fq_width=2.0):
    """
    Computes Cross-Channel PAC using the Ozkurt metric.
    
    sig_phase : 1D array-like
        The signal providing the phase (low frequency).
    sig_amp : 1D array-like
        The signal providing the amplitude envelope (high frequency).
    """
    n_pts = len(sig_phase)
    epsilon = 1e-12
    
    # Pre-process both signals
    sig_p = detrend(sig_phase, type='linear')
    sig_p -= np.mean(sig_p)
    sig_a = detrend(sig_amp, type='linear')
    sig_a -= np.mean(sig_a)
    
    # Push to GPU
    p_gpu = cp.array(sig_p, dtype=cp.float64)
    a_gpu = cp.array(sig_a, dtype=cp.float64)

    low_fq_range = np.linspace(f_low[0], f_low[1], n_bins)
    high_fq_range = np.linspace(f_high[0], f_high[1], n_bins)

    # Hilbert kernel
    h = cp.zeros(n_pts, dtype=cp.float64)
    if n_pts % 2 == 0:
        h[0] = h[n_pts // 2] = 1
        h[1:n_pts // 2] = 2
    else:
        h[0] = 1
        h[1:(n_pts + 1) // 2] = 2

    # --- Extract Phase from sig_phase ---
    phases = cp.zeros((len(low_fq_range), n_pts), dtype=cp.complex128)
    for i, f in enumerate(low_fq_range):
        taps, order = get_pactools_taps(fs, f, low_fq_width)
        taps_gpu = cp.array(taps, dtype=cp.float64)
        filtered = cp.convolve(p_gpu, taps_gpu, mode='same')
        f_sig = cp.fft.fft(filtered)
        analytic = cp.fft.ifft(f_sig * h)
        phases[i] = cp.roll(analytic / (cp.abs(analytic) + epsilon), - (order // 2))

    # --- Extract Amplitude from sig_amp ---
    amplitudes = cp.zeros((len(high_fq_range), n_pts), dtype=cp.float64)
    high_width = 2.0 * low_fq_range.max() 
    for j, f in enumerate(high_fq_range):
        taps, order = get_pactools_taps(fs, f, high_width)
        taps_gpu = cp.array(taps, dtype=cp.float64)
        filtered = cp.convolve(a_gpu, taps_gpu, mode='same')
        f_sig = cp.fft.fft(filtered)
        analytic = cp.fft.ifft(f_sig * h)
        amplitudes[j] = cp.abs(cp.roll(analytic, - (order // 2)))

    # Cross-calculate
    norm_a = cp.linalg.norm(amplitudes, axis=1)
    numerator = cp.abs(cp.matmul(phases, amplitudes.T))
    mi_map = numerator / (cp.sqrt(n_pts) * norm_a + epsilon)
    
    return cp.asnumpy(mi_map)



def ozkurt_pac_all_to_all(sig1, sig2, fs, f_low=(2, 14), f_high=(30, 100), n_bins=50, low_fq_width=2.0):
    n_pts = len(sig1)
    epsilon = 1e-12
    
    # Pre-process and Move to GPU
    def prep(s):
        s = detrend(s, type='linear')
        s -= np.mean(s)
        return cp.array(s, dtype=cp.float64)

    s1_gpu = prep(sig1)
    s2_gpu = prep(sig2)

    low_fq_range = np.linspace(f_low[0], f_low[1], n_bins)
    high_fq_range = np.linspace(f_high[0], f_high[1], n_bins)

    # Hilbert kernel
    h = cp.zeros(n_pts, dtype=cp.float64)
    if n_pts % 2 == 0:
        h[0] = h[n_pts // 2] = 1
        h[1:n_pts // 2] = 2
    else:
        h[0] = 1
        h[1:(n_pts + 1) // 2] = 2

    # --- Feature Extraction (Only do this once per channel) ---
    # Phase extracted from both
    ph1 = cp.zeros((n_bins, n_pts), dtype=cp.complex128)
    ph2 = cp.zeros((n_bins, n_pts), dtype=cp.complex128)
    # Amplitude extracted from both
    amp1 = cp.zeros((n_bins, n_pts), dtype=cp.float64)
    amp2 = cp.zeros((n_bins, n_pts), dtype=cp.float64)

    # Extract Phases
    for i, f in enumerate(low_fq_range):
        taps, order = get_pactools_taps(fs, f, low_fq_width)
        taps_gpu = cp.array(taps, dtype=cp.float64)
        
        for sig_gpu, ph_store in zip([s1_gpu, s2_gpu], [ph1, ph2]):
            filt = cp.convolve(sig_gpu, taps_gpu, mode='same')
            analytic = cp.fft.ifft(cp.fft.fft(filt) * h)
            ph_store[i] = cp.roll(analytic / (cp.abs(analytic) + epsilon), -(order // 2))

    # Extract Amplitudes
    high_width = 2.0 * low_fq_range.max() 
    for j, f in enumerate(high_fq_range):
        taps, order = get_pactools_taps(fs, f, high_width)
        taps_gpu = cp.array(taps, dtype=cp.float64)
        
        for sig_gpu, amp_store in zip([s1_gpu, s2_gpu], [amp1, amp2]):
            filt = cp.convolve(sig_gpu, taps_gpu, mode='same')
            analytic = cp.fft.ifft(cp.fft.fft(filt) * h)
            amp_store[j] = cp.abs(cp.roll(analytic, -(order // 2)))

    # --- Pairwise Modulation Index Calculation ---
    # We compute the 4 maps using matmul
    def get_mi(p, a):
        norm_a = cp.linalg.norm(a, axis=1)
        num = cp.abs(cp.matmul(p, a.T))
        return cp.asnumpy(num / (cp.sqrt(n_pts) * norm_a + epsilon))

    # Return as a dictionary or a 4D array [Phase_Ch, Amp_Ch, Freq, Freq]
    maps = np.zeros((2, 2, n_bins, n_bins))
    maps[0, 0] = get_mi(ph1, amp1) # Ch1 Self
    maps[0, 1] = get_mi(ph1, amp2) # Ch1 -> Ch2
    maps[1, 0] = get_mi(ph2, amp1) # Ch2 -> Ch1
    maps[1, 1] = get_mi(ph2, amp2) # Ch2 Self

    return maps


def plot_pairwise_pac(maps, f_low=(2,14), f_high=(30,100), ch_names=None):
    """
    Plots a 2x2 matrix of PAC comodulograms.
    
    Parameters:
    -----------
    maps : 4D array (2, 2, n_bins, n_bins)
        The result from ozkurt_pac_all_to_all.
    f_low, f_high : tuples
        Frequency ranges for labeling the axes.
    ch_names : list of str, optional
        Custom names for the two channels. Defaults to ["Ch1", "Ch2"].
    """
    # Handle default naming logic
    if ch_names is None:
        ch_names = ["Ch1", "Ch2"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    extent = [f_low[0], f_low[1], f_high[0], f_high[1]]
    
    # Determine global vmax for consistent scaling across all 4 plots
    vmax = maps.max()
    
    for i in range(2): # Phase-providing channel index
        for j in range(2): # Amplitude-providing channel index
            ax = axes[i, j]
            
            # maps[i, j] corresponds to Phase from ch_names[i] and Amp from ch_names[j]
            im = ax.imshow(maps[i, j].T, origin='lower', extent=extent, 
                           aspect='auto', cmap='jet', vmin=0, vmax=vmax)
            
            # Formatted Title
            title = f"{ch_names[i]} Phase to {ch_names[j]} Amp"
            if i == j:
                title += " (Self)"
            
            ax.set_title(title, fontweight='bold', pad=10)
            
            # Labels only on the outer edges for cleanliness
            if i == 1:
                ax.set_xlabel("Phase Frequency (Hz)")
            if j == 0:
                ax.set_ylabel("Amplitude Frequency (Hz)")

    # Adjust layout to make room for the colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Modulation Index')
    
    return fig, axes


