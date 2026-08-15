import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as cp_ndimage # GPU labeling
from scipy import stats
import xarray as xr
import os
from cupyx.scipy.ndimage import label
from scipy import stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from cupyx import scatter_add
import matplotlib.colors as colors

from  helpers import cwt

def run_cluster_permtest_gpu(stack_a, stack_b, params):
    n_permutations = params.get("n_perms", 1000)
    alpha = params.get("alpha", 0.05)
    p_thresh = params.get("p_thresh", 0.05)

    a_gpu = cp.asarray(stack_a)
    b_gpu = cp.asarray(stack_b)
    n_a, n_b = a_gpu.shape[0], b_gpu.shape[0]
    combined = cp.concatenate([a_gpu, b_gpu], axis=0)
    
    df = n_a + n_b - 2
    t_threshold = float(stats.t.ppf(1 - p_thresh/2, df))
    
    struct = cp.ones((3, 3, 3)) 

    def get_max_cluster_mass_gpu(t_map):
        mask = cp.abs(t_map) > t_threshold
        labels, n_labels = cp_ndimage.label(mask, structure=struct)
        if n_labels == 0: return 0.0
        
        cluster_ids = cp.arange(1, n_labels + 1)
        masses = cp_ndimage.sum(cp.abs(t_map), labels, cluster_ids)
        return float(cp.max(masses))

    m_a, m_b = a_gpu.mean(axis=0), b_gpu.mean(axis=0)
    v_a, v_b = a_gpu.var(axis=0, ddof=1), b_gpu.var(axis=0, ddof=1)
    obs_t = (m_a - m_b) / cp.sqrt(v_a/n_a + v_b/n_b)
    obs_max_mass = get_max_cluster_mass_gpu(obs_t)

    null_masses = cp.zeros(n_permutations)
    print(f"Running {n_permutations} permutations on GPU...")
    
    for i in range(n_permutations):
        idx = cp.random.permutation(n_a + n_b)
        p_a = combined[idx[:n_a]]
        p_b = combined[idx[n_a:]]
        
        pm_a, pm_b = p_a.mean(axis=0), p_b.mean(axis=0)
        pv_a, pv_b = p_a.var(axis=0, ddof=1), p_b.var(axis=0, ddof=1)
        p_t = (pm_a - pm_b) / cp.sqrt(pv_a/n_a + pv_b/n_b)
        
        null_masses[i] = get_max_cluster_mass_gpu(p_t)
        
        if (i+1) % 500 == 0:
            print(f"Permutation {i+1}/{n_permutations}")

    cluster_mass_limit = cp.percentile(null_masses, 100 * (1 - alpha))
    
    labels, n_labels = cp_ndimage.label(cp.abs(obs_t) > t_threshold, structure=struct)
    sig_t_map = cp.zeros_like(obs_t)
    
    if n_labels > 0:
        cluster_ids = cp.arange(1, n_labels + 1)
        masses = cp_ndimage.sum(cp.abs(obs_t), labels, cluster_ids)
        valid_clusters = cluster_ids[masses > cluster_mass_limit]
        sig_mask = cp.isin(labels, valid_clusters)
        sig_t_map[sig_mask] = obs_t[sig_mask]

    print(f"Complete. Mass Threshold: {float(cluster_mass_limit):.2f}")
    return cp.asnumpy(obs_t), cp.asnumpy(sig_t_map), float(cluster_mass_limit)


def save_permtest_results(output_path, obs_t, sig_t, mass_thresh, ad_xr, wt_xr, params, save_plots=True):
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Directory {output_dir} does not exist. Creating it...")
        os.makedirs(output_dir, exist_ok=True)

    ad_mean = ad_xr.mean(dim='animal')
    wt_mean = wt_xr.mean(dim='animal')
    diff_mean = ad_mean - wt_mean

    results_ds = xr.Dataset(
        data_vars={
            "observed_t": (["channel", "frequency", "time"], obs_t),
            "significant_t": (["channel", "frequency", "time"], sig_t),
            "group_ad_mean": (["channel", "frequency", "time"], ad_mean.values),
            "group_wt_mean": (["channel", "frequency", "time"], wt_mean.values),
            "power_diff": (["channel", "frequency", "time"], diff_mean.values),
        },
        coords={
            "channel": ad_xr.channel.values,
            "frequency": ad_xr.frequency.values,
            "time": ad_xr.time.values,
        },
        attrs={
            "n_permutations": params.get("n_perms", 1000),
            "alpha": params.get("alpha", 0.05),
            "cluster_forming_p_thresh": params.get("p_thresh", 0.05),
            "cluster_mass_threshold": mass_thresh,
            "n_ad": len(ad_xr.animal),
            "n_wt": len(wt_xr.animal)
        }
    )

    results_ds.to_netcdf(output_path)
    print(f"Results successfully saved to: {output_path}")

    return results_ds


def plot_permtest_results(results_ds, output_dir, prefix="analysis", 
                          t_clim=(-5, 5), diff_clim=(-3, 3), 
                          freq_range=None, save_plots=True):
    """
    Plots T-statistics and Mean Difference maps from a permutation test dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Prepare coordinates and mask
    freqs = results_ds.frequency.values
    time_axis = results_ds.time.values
    roi_names = results_ds.channel.values
    sig_mask = (results_ds.significant_t.values != 0).astype(float)
    
    # --- 1. Plot T-Statistics ---
    t_path = os.path.join(output_dir, f"{prefix}_tstats.png") if save_plots else None
    
    cwt.plot_nch_scalogram_with_psd(
        freqs=freqs,
        time_axis=time_axis,
        cwt_matrix=results_ds.observed_t.values,
        roi_names=roi_names,
        title=f"T-Statistics: AD vs WT (Cluster Mass Thresh: {results_ds.attrs['cluster_mass_threshold']:.2f})",
        use_log=False, 
        mask=sig_mask,
        freq_range=freq_range,
        save_path=t_path,
        show_plot=False,
        clim=t_clim,
        cbar_label="T-Statistic"
    )

    # --- 2. Plot Mean Power Difference ---
    diff_path = os.path.join(output_dir, f"{prefix}_mean_diff.png") if save_plots else None
    
    cwt.plot_nch_scalogram_with_psd(
        freqs=freqs,
        time_axis=time_axis,
        cwt_matrix=results_ds.power_diff.values,
        roi_names=roi_names,
        title="Mean Power Difference (AD - WT) with Significance Mask",
        use_log=False,
        mask=sig_mask,
        freq_range=freq_range,
        save_path=diff_path,
        show_plot=False,
        clim=diff_clim,
        cbar_label=r"$\Delta$ Power (dB)"
    )

    if save_plots:
        print(f"Plots saved to {output_dir} with prefix '{prefix}'")


# Second run. 1D cluster based permtest
import cupy as cp
import numpy as np
from cupyx.scipy.ndimage import label

import cupy as cp
import numpy as np
from cupyx import scatter_add
from cupyx.scipy.ndimage import label

def run_1d_cluster_test_gpu(group1, group2, n_permutations=1000, p_thresh=0.05):
    """
    group1 and group2 shape (subjects * samples)
    """
    g1, g2 = cp.asarray(group1), cp.asarray(group2)
    n_samples = g1.shape[-1]
    n1, n2 = g1.shape[0], g2.shape[0]
    n_total = n1 + n2
    combined = cp.concatenate([g1, g2], axis=0)
    
    """
    def get_welch_t(a, b):
        m1, m2 = cp.mean(a, axis=0), cp.mean(b, axis=0)
        v1, v2 = cp.var(a, axis=0, ddof=1) + 1e-12, cp.var(b, axis=0, ddof=1) + 1e-12
        return (m1 - m2) / cp.sqrt(v1 / n1 + v2 / n2)
    """
    def get_welch_t(a, b):
        n1, n2 = a.shape[0], b.shape[0]
        trim = 0.1 
        
        k1 = int(n1 * trim)
        k2 = int(n2 * trim)
        
        a_sorted = cp.sort(a, axis=0)
        b_sorted = cp.sort(b, axis=0)
        
        a_trimmed = a_sorted[k1 : n1 - k1, :]
        b_trimmed = b_sorted[k2 : n2 - k2, :]
        
        h1, h2 = a_trimmed.shape[0], b_trimmed.shape[0]
        
        m1, m2 = cp.mean(a_trimmed, axis=0), cp.mean(b_trimmed, axis=0)
        
        v1 = cp.var(a_trimmed, axis=0, ddof=1) + 1e-12
        v2 = cp.var(b_trimmed, axis=0, ddof=1) + 1e-12
        
        robust_se = cp.sqrt(v1 / h1 + v2 / h2)
        
        return (m1 - m2) / robust_se
        
    saved_shuffle_tmaps = cp.zeros((n_samples, n_permutations))

    for p in range(n_permutations):
        idx = cp.random.permutation(n_total)
        saved_shuffle_tmaps[:, p] = get_welch_t(combined[idx[:n1]], combined[idx[n1:]])

    lt = cp.percentile(saved_shuffle_tmaps, (p_thresh/2)*100, axis=1)
    ut = cp.percentile(saved_shuffle_tmaps, (1 - p_thresh/2)*100, axis=1)
    
    del saved_shuffle_tmaps
    cp.get_default_memory_pool().free_all_blocks()

    obs_t = get_welch_t(g1, g2)

    def get_max_cluster(t_map, l_thr, u_thr):
        pos_labels, n_pos = label(t_map > u_thr)
        neg_labels, n_neg = label(t_map < l_thr)
        max_stat = 0.0
        for i in range(1, n_pos + 1):
            s = float(cp.sum(t_map[pos_labels == i]))
            if s > max_stat: max_stat = s
        for i in range(1, n_neg + 1):
            s = float(cp.abs(cp.sum(t_map[neg_labels == i])))
            if s > max_stat: max_stat = s
        return max_stat

    null_dist = cp.zeros(n_permutations)
    for p in range(n_permutations):
        idx = cp.random.permutation(n_total)
        p_tmap = get_welch_t(combined[idx[:n1]], combined[idx[n1:]])
        null_dist[p] = get_max_cluster(p_tmap, lt, ut)

    pos_l, n_p = label(obs_t > ut)
    neg_l, n_n = label(obs_t < lt)
    
    obs_cluster_masks = []
    obs_cluster_stats = []
    
    for i in range(1, n_p + 1):
        m = pos_l == i
        obs_cluster_masks.append(m)
        obs_cluster_stats.append(float(cp.sum(obs_t[m])))
        
    for i in range(1, n_n + 1):
        m = neg_l == i
        obs_cluster_masks.append(m)
        obs_cluster_stats.append(float(cp.abs(cp.sum(obs_t[m]))))

    sig_mask = cp.zeros(obs_t.shape, dtype=bool)
    cluster_p_values = []
    
    found_sig = False
    for i, cluster_stat in enumerate(obs_cluster_stats):
        p_val = float(cp.mean(null_dist >= cluster_stat))
        cluster_p_values.append(p_val)
        
        if p_val <= 0.05:
            found_sig = True
            print(f"Found Significant Cluster: Mass={cluster_stat:.2f}, p={p_val:.4f}")
            sig_mask = cp.logical_or(sig_mask, obs_cluster_masks[i])
            
    if not found_sig:
        print("No significant clusters found.")

    return (cp.asnumpy(obs_t), 
            cp.asnumpy(sig_mask), 
            cp.asnumpy(null_dist), 
            cluster_p_values,
            ut,
            lt)

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupyx.scipy.ndimage import label

def plot_1d_clusters(t_map, null_dist, ut, lt, x_values=None ,critical_mass_p=None, title=''):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    t_map_gpu = cp.asarray(t_map)

    if x_values is None:
        x = np.arange(len(t_map))
    else:
        x = x_values
    
    ax1.plot(x, t_map, color='black', lw=1, label='T-statistic')
    
    # Handle potentially vectorized thresholds
    ut_cpu = cp.asnumpy(ut) if isinstance(ut, cp.ndarray) else np.full(x.shape, ut)
    lt_cpu = cp.asnumpy(lt) if isinstance(lt, cp.ndarray) else np.full(x.shape, lt)

    mass_05 = np.percentile(null_dist, 95)
    sig_mask_05 = cp.zeros(t_map_gpu.shape, dtype=bool)
    
    do_custom = critical_mass_p is not None and not np.isclose(critical_mass_p, 0.05)
    if do_custom:
        mass_custom = np.percentile(null_dist, 100 * (1 - critical_mass_p))
        sig_mask_custom = cp.zeros(t_map_gpu.shape, dtype=bool)
    
    pos_l, n_p = label(t_map_gpu > cp.asarray(ut))
    neg_l, n_n = label(t_map_gpu < cp.asarray(lt))
    observed_masses = []

    for labels, count in [(pos_l, n_p), (neg_l, n_n)]:
        for i in range(1, int(count) + 1):
            m = labels == i
            mass = float(cp.abs(cp.sum(t_map_gpu[m])))
            observed_masses.append(mass)
            
            if mass >= mass_05:
                sig_mask_05 = cp.logical_or(sig_mask_05, m)
            if do_custom and mass >= mass_custom:
                sig_mask_custom = cp.logical_or(sig_mask_custom, m)

    if do_custom:
        ax1.fill_between(x, t_map, where=cp.asnumpy(sig_mask_custom), 
                         color='orange', alpha=0.2, label=f'p < {critical_mass_p}')
    
    ax1.fill_between(x, t_map, where=cp.asnumpy(sig_mask_05), 
                     color='red', alpha=0.4, label='p < 0.05')
    
    # Plot thresholds as lines (works for scalar or vector)
    ax1.plot(x, ut_cpu, color='blue', ls='--', alpha=0.6, label='T-Thresh (Upper)')
    ax1.plot(x, lt_cpu, color='blue', ls='--', alpha=0.6, label='T-Thresh (Lower)')
    
    ax1.axhline(0, color='gray', lw=0.5)
    ax1.set_title(f"{title} Observed T-map")
    ax1.legend(loc='upper right')
    
    ax2.hist(null_dist, bins=50, color='gray', alpha=0.7)
    ax2.axvline(mass_05, color='red', ls='-', label=f'95th Perc (0.05): {mass_05:.2f}')
    
    if do_custom:
        ax2.axvline(mass_custom, color='orange', ls='--', 
                    label=f'{(1-critical_mass_p)*100:.1f}th Perc: {mass_custom:.2f}')
    
    max_obs = max(observed_masses) if observed_masses else 0
    ax2.axvline(max_obs, color='green', ls='-', lw=2, label=f'Observed Max: {max_obs:.2f}')
    ax2.set_title('Null Distribution (Joint-Max)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def plot_1d_groups(group1, group2, x_values=None, labels=("AD", "WT"), colors=('red', 'blue'), log_x=False, log_y=False, title=''):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if x_values is None:
        x_values = np.arange(group1.shape[-1])
        
    for data, label, color in zip([group1, group2], labels, colors):
        # 1. Plot individual subject traces (the "faded" lines)
        # alpha=0.15 is usually the sweet spot for 9-10 subjects
        for i in range(data.shape[0]):
            ax.plot(x_values, data[i, :], color=color, alpha=0.5, lw=0.8, zorder=1)
            
        # 2. Plot the group mean (bold and on top)
        mean = np.mean(data, axis=0)
        #ax.plot(x_values, mean, color=color, label=label, lw=2.5, zorder=2)
        
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
        
    if title:
        ax.set_title(title)
        
    # Standard electrophys aesthetics
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Frequency (Hz)")
    ax.legend()
    
    plt.tight_layout()
    return fig


"""
def plot_1d_clusters(t_map, mask, null_dist, t_threshold):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    x = np.arange(len(t_map))
    ax1.plot(x, t_map, color='black', lw=1, label='T-statistic')
    
    ax1.fill_between(x, t_map, where=mask, color='red', alpha=0.3, label='p < 0.05')
    
    ax1.axhline(t_threshold, color='blue', ls='--', alpha=0.6, label=f'Thresh: {t_threshold:.2f}')
    ax1.axhline(-t_threshold, color='blue', ls='--', alpha=0.6)
    ax1.axhline(0, color='gray', lw=0.5)
    
    ax1.set_title('Observed T-map')
    ax1.legend(loc='upper right')
    
    ax2.hist(null_dist, bins=50, color='gray', alpha=0.7)
    null_95 = np.percentile(null_dist, 95)
    ax2.axvline(null_95, color='red', ls=':', label='95th Percentile')
    ax2.set_title('Null Distribution (Max Cluster Masses)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
"""


def run_2d_cluster_test_gpu(group1, group2, n_permutations=1000, p_thresh=0.05, n_batches=10):
    g1, g2 = cp.asarray(group1), cp.asarray(group2)
    n1, n2 = g1.shape[0], g2.shape[0]
    n_total = n1 + n2
    n_freqs, n_times = g1.shape[1], g1.shape[2]
    combined = cp.concatenate([g1, g2], axis=0)
    
    def get_welch_t(a, b, axis=1):
        trim = 0.1
        k1, k2 = int(a.shape[axis] * trim), int(b.shape[axis] * trim)
        a_s = cp.sort(a, axis=axis)
        b_s = cp.sort(b, axis=axis)
        
        if a.ndim == 3:
            a_t = a_s[k1 : n1 - k1]
            b_t = b_s[k2 : n2 - k2]
        else:
            a_t = a_s[:, k1 : n1 - k1]
            b_t = b_s[:, k2 : n2 - k2]
            
        h1, h2 = a_t.shape[axis], b_t.shape[axis]
        m1, v1 = cp.mean(a_t, axis=axis), cp.var(a_t, axis=axis, ddof=1) + 1e-12
        m2, v2 = cp.mean(b_t, axis=0 if a.ndim==3 else axis), cp.var(b_t, axis=0 if a.ndim==3 else axis, ddof=1) + 1e-12
        return (m1 - m2) / cp.sqrt(v1 / h1 + v2 / h2)

    obs_t = get_welch_t(g1, g2, axis=0)
    
    perm_tmaps = cp.zeros((n_permutations, n_freqs, n_times), dtype=cp.float32)
    perms_per_batch = n_permutations // n_batches
    
    for b in range(n_batches):
        start = b * perms_per_batch
        end = (b + 1) * perms_per_batch if b != n_batches - 1 else n_permutations
        curr_size = end - start
        shuffled_idx = cp.stack([cp.random.permutation(n_total) for _ in range(curr_size)])
        batch_data = combined[shuffled_idx]
        perm_tmaps[start:end] = get_welch_t(batch_data[:, :n1], batch_data[:, n1:], axis=1)
        del batch_data
        cp.get_default_memory_pool().free_all_blocks()

    ut = cp.percentile(perm_tmaps, (1 - p_thresh/2)*100, axis=0)
    lt = cp.percentile(perm_tmaps, (p_thresh/2)*100, axis=0)

    def get_max_mass_fast(t_map, thresh_map, is_pos=True):
        mask = (t_map > thresh_map) if is_pos else (t_map < thresh_map)
        labels, n_comp = label(mask)
        if n_comp == 0: return 0.0
        
        sums = cp.zeros(int(n_comp + 1), dtype=cp.float32)
        weights = cp.abs(t_map) if not is_pos else t_map
        # Fixed the scatter_add call
        scatter_add(sums, labels.ravel(), weights.ravel())
        return float(cp.max(sums[1:]))

    null_dist = cp.zeros(n_permutations)
    for p in range(n_permutations):
        t_map = perm_tmaps[p]
        null_dist[p] = max(get_max_mass_fast(t_map, ut, True), 
                           get_max_mass_fast(t_map, lt, False))

    def get_sig_clusters(t_map, thresh_map, is_pos=True):
        mask = (t_map > thresh_map) if is_pos else (t_map < thresh_map)
        labels, n_comp = label(mask)
        if n_comp == 0: return cp.zeros_like(mask), []
        
        sums = cp.zeros(int(n_comp + 1), dtype=cp.float32)
        weights = cp.abs(t_map) if not is_pos else t_map
        scatter_add(sums, labels.ravel(), weights.ravel())
        
        p_vals = [float(cp.mean(null_dist >= s)) for s in sums[1:]]
        sig_indices = cp.where(cp.array(p_vals) <= 0.05)[0] + 1
        
        full_mask = cp.isin(labels, sig_indices)
        return full_mask, p_vals

    pos_mask, pos_p = get_sig_clusters(obs_t, ut, True)
    neg_mask, neg_p = get_sig_clusters(obs_t, lt, False)
    
    sig_pos = [p for p in pos_p if p <= 0.05]
    sig_neg = [p for p in neg_p if p <= 0.05]

    if sig_pos or sig_neg:
        print(f"\n[!] SIGNIFICANT CLUSTERS FOUND")
        for p in sig_pos:
            print(f"    -> Positive Cluster: p = {p:.4f}")
        for p in sig_neg:
            print(f"    -> Negative Cluster: p = {p:.4f}")

    return (cp.asnumpy(obs_t), 
            cp.asnumpy(pos_mask | neg_mask), 
            cp.asnumpy(null_dist), 
            pos_p + neg_p, 
            cp.asnumpy(ut), 
            cp.asnumpy(lt))



def plot_2d_clusters(t_map, ut, lt, x_axis=None, y_axis=None, title='2D Cluster Analysis'):
    if x_axis is None:
        x_axis = np.arange(t_map.shape[1])
    if y_axis is None:
        y_axis = np.arange(t_map.shape[0])

    X, Y = np.meshgrid(x_axis, y_axis)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=t_map,
        colorscale='Spectral',
        reversescale=True,
        cmid=0,
        colorbar=dict(title='T-stat', thickness=20),
        name='Observed T-map'
    ))

    surface_opts = dict(
        showscale=False,
        opacity=0.3,
        hoverinfo='skip'
    )

    fig.add_trace(go.Surface(
        x=X, y=Y, z=ut,
        colorscale=[[0, 'blue'], [1, 'blue']],
        name='Upper Threshold',
        **surface_opts
    ))

    fig.add_trace(go.Surface(
        x=X, y=Y, z=lt,
        colorscale=[[0, 'cyan'], [1, 'cyan']],
        name='Lower Threshold',
        **surface_opts
    ))

    camera = dict(
            eye=dict(x=1.5, y=1.5, z=1.2), # Distance/Angle of the view
            center=dict(x=0, y=0, z=0),     # Point camera looks at
            up=dict(x=0, y=0, z=1)          # Which direction is 'up'
        )
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='Time',
            yaxis_title='Frequency',
            zaxis_title='T-value',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1),
            camera=camera
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    #fig.show()
    return fig

def plot_2d_clusters_clean(t_map, sig_mask, x_axis, y_axis, title='2D Cluster Analysis'):
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Matplotlib pcolormesh(X, Y, Z) 
    # If x_axis = Phase (Low) and y_axis = Amplitude (High)
    # t_map must be shaped (len(y_axis), len(x_axis))
    im = ax.pcolormesh(
        x_axis, 
        y_axis, 
        t_map, 
        cmap='Spectral_r',
        shading='auto',
        vmin=-5, 
        vmax=5
    )
    
    if sig_mask is not None and sig_mask.any():
        X, Y = np.meshgrid(x_axis, y_axis)
        ax.contour(
            X, Y,
            sig_mask, 
            levels=[0.5], 
            colors='black', 
            linewidths=2
        )
        
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    if "PAC" in title:
        ax.set_xlabel("Phase Frequency (Low) [Hz]")
        ax.set_ylabel("Amplitude Frequency (High) [Hz]")
    else:
        ax.set_xlabel("X-Axis")
        ax.set_ylabel("Y-Axis")

    plt.colorbar(im, label='T-statistic')
    plt.tight_layout()
    
    return fig

def plot_2d_comparison(g1, g2, x_axis, y_axis, title='Group Comparison', use_log=False):
    m1 = np.nanmean(cp.asnumpy(g1), axis=0)
    m2 = np.nanmean(cp.asnumpy(g2), axis=0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    
    vmin = min(np.nanmin(m1), np.nanmin(m2))
    vmax = max(np.nanmax(m1), np.nanmax(m2))
    
    norm = colors.LogNorm(vmin=max(vmin, 1e-9), vmax=vmax) if use_log else None
    cmap = 'viridis'

    im1 = ax1.pcolormesh(x_axis, y_axis, m1, cmap=cmap, shading='auto', 
                         vmin=vmin if not use_log else None, 
                         vmax=vmax if not use_log else None, norm=norm)
    ax1.set_title("Group: AD (Mean)")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [s]")
    
    im2 = ax2.pcolormesh(x_axis, y_axis, m2, cmap=cmap, shading='auto', 
                         vmin=vmin if not use_log else None, 
                         vmax=vmax if not use_log else None, norm=norm)
    ax2.set_title("Group: WT (Mean)")
    ax2.set_xlabel("Time [s]")
    
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im2, cax=cbar_ax, label='Magnitude')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    return fig



def plot_2d_clusters_clean_WORKS(t_map, sig_mask, freqs, title='2D Cluster Analysis'):
    """
    t_map: (101, 101) observed T-statistics
    sig_mask: (101, 101) boolean mask from cluster test
    freqs: frequency axis values
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 1. The Heatmap of T-values
    # We use 'RdBu_r' so Red = AD > WT and Blue = WT > AD
    extent = [freqs[0], freqs[-1], freqs[0], freqs[-1]]
    im = ax.imshow(
        t_map, 
        extent=extent,
        origin='lower',
        aspect='auto',
        cmap='Spectral_r',
        vmin=-5, vmax=5 # Standard T-scale
    )
    
    # 2. The Significance Outlines
    # This draws a clean black contour around your significant clusters
    if sig_mask is not None and sig_mask.any():
        ax.contour(
            sig_mask, 
            levels=[0.5], 
            colors='black', 
            linewidths=2,
            extent=extent
        )
        
    # 3. Aesthetics
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Frequency (f2) [Hz]")
    ax.set_ylabel("Frequency (f1) [Hz]")
    
    # Add a reference line for the diagonal (where ratio is 1:1)
    ax.plot([freqs[0], freqs[-1]], [freqs[0], freqs[-1]], color='gray', linestyle='--', alpha=0.5)

    plt.colorbar(im, label='T-statistic')
    plt.tight_layout()
    
    return fig