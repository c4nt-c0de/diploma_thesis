from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

# --- 1. STABILITY & PLOTTING UTILITIES ---

def check_stability(b, a=None, name="Filter", is_sos=True):
    if is_sos:
        z, p, k = signal.sos2zpk(b)
    else:
        z, p, k = signal.tf2zpk(b, a)
    
    max_p = np.max(np.abs(p))
    is_stable = max_p < 1.0
    
    if not is_stable:
        print(f"!!! UNSTABLE FILTER: [{name}] Max Pole: {max_p:.6f}")
    elif max_p > 0.999:
        print(f"! CRITICAL WARNING: [{name}] Marginal Stability (Max Pole: {max_p:.6f})")
        
    return is_stable, max_p

def plot_filter_diagnostic(b, a=None, fs=1000, name="Filter", is_sos=True):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    if is_sos:
        z, p, k = signal.sos2zpk(b)
        w, h = signal.sosfreqz(b, worN=8000, fs=fs)
        impulse_resp = signal.sosfilt(b, np.r_[0, 1, np.zeros(249)])
    else:
        z, p, k = signal.tf2zpk(b, a)
        w, h = signal.freqz(b, a, worN=8000, fs=fs)
        impulse_resp = signal.lfilter(b, a, np.r_[0, 1, np.zeros(249)])
    
    ax1.plot(w, 20 * np.log10(np.maximum(abs(h), 1e-5)))
    ax1.set_title("PSD")
    ax1.set_xlim([0, 300]); ax1.set_ylim([-80, 5])
    ax1.grid(True, alpha=0.3)

    ax2.plot(impulse_resp, color='black')
    ax2.set_title("Impulse Response")
    ax2.grid(True, alpha=0.3)

    uc = plt.Circle((0, 0), 1, fill=False, color='black', ls='--', alpha=0.5)
    ax3.add_patch(uc)
    ax3.scatter(np.real(z), np.imag(z), s=30, marker='o', edgecolors='blue', facecolors='none')
    ax3.scatter(np.real(p), np.imag(p), s=50, marker='x', color='red')
    ax3.set_title("Z-plane")
    ax3.set_xlim([-1.2, 1.2]); ax3.set_ylim([-1.2, 1.2]); ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.2)
    
    plt.suptitle(name, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# --- 2. FILTER FUNCTIONS ---

def apply_cheby2_highpass(data=None, cutoff=4, fs=1000, order=4, rs=40, plot=False):
    sos = signal.cheby2(order, rs, cutoff, btype='highpass', fs=fs, output='sos')
    stable, _ = check_stability(sos, name=f"Cheby2 HP {cutoff}Hz", is_sos=True)
    
    if (not stable) or plot:
        plot_filter_diagnostic(sos, fs=fs, name=f"Cheby2 HP {cutoff}Hz", is_sos=True)
        
    return signal.sosfiltfilt(sos, data, axis=-1) if data is not None else None

def apply_comb_filter(data=None, f0=50, fs=1000, q=10, plot=False):
    b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
    stable, _ = check_stability(b, a, name=f"Comb {f0}Hz", is_sos=False)
    
    if (not stable) or plot:
        plot_filter_diagnostic(b, a, fs=fs, name=f"Comb {f0}Hz", is_sos=False)
        
    return signal.filtfilt(b, a, data, axis=-1) if data is not None else None

def apply_cheby2_filterbank(data=None, cutoffs=None, fs=1000, q=0.5, rs=40, plot=False, include_raw=False):
    if cutoffs is None:
        cutoffs = np.arange(2, 210, 10).tolist()
        
    results = {}
    if data is not None and include_raw:
        results[0] = data

    for fc in cutoffs:
        if fc == 0: continue
            
        raw_order = q * (fc / (fs / 100)) * 10 
        order = int(np.clip(raw_order, 4, 10))
        if order % 2 != 0: order += 1
        
        sos = signal.cheby2(order, rs, fc, btype='highpass', fs=fs, output='sos')
        stable, _ = check_stability(sos, name=f"{fc}Hz", is_sos=True)
        
        if (not stable) or plot:
            plot_filter_diagnostic(sos, fs=fs, name=f"{fc}Hz Bank", is_sos=True)
            
        if data is not None:
            results[fc] = signal.sosfiltfilt(sos, data, axis=-1)
            
    return results if data is not None else None