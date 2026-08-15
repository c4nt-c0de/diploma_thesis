import numpy as np
import pywt
import plotly.graph_objects as go
from scipy.signal import decimate

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.signal import welch
import xarray as xr

CMAP = 'jet'

def compute_cwt_matrix(data_np, fs, f_min=1, f_max=100, n_scales=100, downsample=1, wavelet="cmor3.0-1.0"):
    n_channels, n_samples = data_np.shape
    new_fs = fs / downsample
    Ts = 1.0 / new_fs
    #wavelet = "cmor3.0-1.0" #wavelet = "cmor6.0-1.0"

    cf = pywt.central_frequency(wavelet)
    
    # Define scales here
    scales = np.geomspace(cf/(f_max*Ts), cf/(f_min*Ts), num=n_scales)
    freqs = (cf / (scales * Ts)).astype(np.float32)
    
    test_sig = data_np[0] if downsample == 1 else decimate(data_np[0], q=downsample, zero_phase=True)
    new_len = len(test_sig)
    
    cwt_complex = np.zeros((n_channels, n_scales, new_len), dtype=np.complex64)
    for i in range(n_channels):
        sig = data_np[i].astype(np.float64)
        if downsample > 1:
            sig = decimate(sig, q=downsample, ftype='iir', zero_phase=True)
        coef, _ = pywt.cwt(sig, scales, wavelet, sampling_period=Ts, method="fft")
        cwt_complex[i] = coef
        
    return freqs, np.arange(new_len) * Ts, cwt_complex, scales # Added scales here


def plot_scalogram(freqs, time_axis, coef, title="Scalogram"):
    """Plotly Heatmap of a single channel's CWT coef (freqs x time)."""
    power_db = 10 * np.log10(np.abs(coef)**2 + 1e-10)
    zmin, zmax = np.percentile(power_db, [5, 95]) # Tighter bounds for better contrast
    
    # Decimate time axis for Plotly performance if data is huge
    step = max(1, len(time_axis) // 5000) 
    
    fig = go.Figure(data=go.Heatmap(
        x=time_axis[::step], y=freqs, z=power_db[:, ::step],
        colorscale=CMAP, zmin=zmin, zmax=zmax, zsmooth='best'
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="Time (s)", yaxis_title="Freq (Hz)",
        yaxis=dict(type="log", tickvals=[1, 2, 5, 10, 20, 50, 100])
    )
    fig.show()
from scipy.ndimage import gaussian_filter1d

def pairwise_cross_channel_coherence(cwt_complex, freqs, fs, n_cycles=3):
    n_ch, n_freq, _ = cwt_complex.shape
    # 1. Pre-smooth all Power (PSDs) and complex signals across the time axis
    # sigma scales per frequency: (n_cycles / f) * fs / 6
    sigmas = (n_cycles / freqs) * fs / 6
    s_psd = np.array([[gaussian_filter1d(np.abs(cwt_complex[c, f])**2, sigma=sigmas[f]) 
                       for f in range(n_freq)] for c in range(n_ch)])
    
    # 2. Iterate pairs and compute complex coherence
    pairs = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
    coh_list = []
    for i, j in pairs:
        # Smooth the Cross-Spectrum (CSD)
        csd = cwt_complex[i] * np.conj(cwt_complex[j])
        s_csd = np.array([gaussian_filter1d(csd[f].real, sigma=sigmas[f]) + 
                          1j*gaussian_filter1d(csd[f].imag, sigma=sigmas[f]) for f in range(n_freq)])
        
        # Coherence = Sxy / sqrt(Sxx * Syy)
        coh_list.append(s_csd / (np.sqrt(s_psd[i] * s_psd[j]) + 1e-10))

    coh_complex = np.array(coh_list)
    return coh_complex, np.abs(coh_complex)**2, np.imag(coh_complex), [f"Ch{i}-Ch{j}" for i, j in pairs]



import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter, NullFormatter
import numpy as np

def plot_nch_scalogram_with_psd(freqs, time_axis, cwt_matrix, roi_names=None, 
                               title="Multi-Channel Analysis", use_log=True, 
                               freq_range=None, save_path=None, mask=None, 
                               show_plot=True, clim=None, cbar_label=""):
    def to_numpy(x):
        if x is None: return None
        return x.get() if hasattr(x, 'get') else np.asarray(x)

    freqs, time_axis = to_numpy(freqs), to_numpy(time_axis)
    cwt_matrix, mask = to_numpy(cwt_matrix), to_numpy(mask)

    if freq_range is not None:
        f_min, f_max = freq_range[0] or freqs.min(), freq_range[1] or freqs.max()
        f_idx = (freqs >= f_min) & (freqs <= f_max)
        freqs, cwt_matrix = freqs[f_idx], cwt_matrix[:, f_idx, :]
        if mask is not None: mask = mask[:, f_idx, :]

    n_ch = cwt_matrix.shape[0]
    fig = plt.figure(figsize=(16, 3 * n_ch))
    gs = gridspec.GridSpec(n_ch, 2, width_ratios=[6, 1], wspace=0.08, hspace=0.3)
    
    display_data = 10 * np.log10(cwt_matrix + 1e-10) if use_log else cwt_matrix
    
    if clim is not None:
        vmin, vmax = clim
    else:
        vmin, vmax = np.percentile(display_data, [1, 99])
        
    y_ticks = [t for t in [4, 10, 20, 50, 100, 150] if freqs.min() <= t <= freqs.max()]
    x_ticks = np.arange(time_axis[0], time_axis[-1] + 0.1, 0.5)

    for i in range(n_ch):
        ax_scal = fig.add_subplot(gs[i, 0])
        mesh = ax_scal.pcolormesh(time_axis, freqs, display_data[i], 
                                  shading='auto', cmap='turbo', 
                                  vmin=vmin, vmax=vmax)
        
        if mask is not None:
            ax_scal.contour(time_axis, freqs, mask[i], levels=[0.5], colors='black', linewidths=2.0, alpha=0.8)
            ax_scal.contour(time_axis, freqs, mask[i], levels=[0.5], colors='white', linewidths=0.8, alpha=1.0)

        if use_log: ax_scal.set_yscale('log')
        ax_scal.set_ylim(freqs.min(), freqs.max())
        ax_scal.set_yticks(y_ticks)
        ax_scal.yaxis.set_major_formatter(ScalarFormatter())
        ax_scal.set_xticks(x_ticks)
        ax_scal.axvline(0, color='white', linestyle='--', alpha=0.7, lw=1.5)
        
        label = roi_names[i] if (roi_names is not None and i < len(roi_names)) else f"CH {i}"
        ax_scal.set_ylabel(f"{label}\n(Hz)", fontsize=11, fontweight='bold')
        
        if i == 0: ax_scal.set_title(title, fontsize=14)
        if i < n_ch - 1: ax_scal.set_xticklabels([])
        else: ax_scal.set_xlabel("Time (s)")

        ax_psd = fig.add_subplot(gs[i, 1], sharey=ax_scal)
        ax_psd.plot(np.mean(display_data[i], axis=1), freqs, color='black', linewidth=1.2)
        ax_psd.grid(True, which='both', alpha=0.3)
        ax_psd.set_xlim(vmin, vmax)
        plt.setp(ax_psd.get_yticklabels(), visible=False)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(mesh, cax=cbar_ax, label=cbar_label)
    
    if save_path: fig.savefig(save_path, bbox_inches='tight', dpi=150)
    if show_plot: plt.show()
    return fig
# --- Example Update to your execution block ---
# found_rois = [c for c in ['left_ACC', 'left_vmPFC', 'right_vmPFC', 'right_ACC'] if c in target_da.channel.values]
# plot_nch_scalogram_with_psd(f, t_centered, np.abs(coefs)**2, roi_names=found_rois, title=...)
# --- Example Usage ---
# plot_4ch_scalogram_with_psd(freqs, t_cwt, avg_scal, fs=1000)

# --- USAGE EXAMPLE ---
# freqs, t_axis, cwt_data = compute_cwt_matrix(data_np, fs=1000, downsample=2)
# mag_input = np.abs(cwt_data).mean(axis=0) # This is your NMF input


def compute_cwt_full_dataset(da, fs, f_min=1, f_max=100, n_scales=100):
    """
    Uses the geometric scale logic from cwt.py but scales to the whole xarray.
    """
    import pywt
    wavelet = "cmor3.0-1.0"
    Ts = 1.0 / fs
    cf = pywt.central_frequency(wavelet)
    
    # Matching your library's exact scale math
    scales = np.geomspace(cf/(f_max*Ts), cf/(f_min*Ts), num=n_scales)
    freqs = (cf / (scales * Ts)).astype(np.float32)

    # Broadcast across trials and channels
    coefs = xr.apply_ufunc(
        lambda x: pywt.cwt(x, scales, wavelet, sampling_period=Ts, method='fft')[0],
        da,
        input_core_dims=[['time']],
        output_core_dims=[['freq', 'time']],
        exclude_dims={'time'},
        vectorize=True
    )
    
    # Realign coordinates
    return coefs.assign_coords(freq=freqs).transpose('trial', 'channel', 'freq', 'time')