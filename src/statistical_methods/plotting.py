import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plotly.graph_objects as go
from cupyx.scipy import ndimage as cp_ndimage
from helpers import cwt

def plot_permtest_results(results_ds, output_dir, prefix="analysis", 
                          t_clim=(-5, 5), diff_clim=(-3, 3), 
                          freq_range=None, save_plots=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    freqs = results_ds.frequency.values
    time_axis = results_ds.time.values
    roi_names = results_ds.channel.values
    sig_mask = (results_ds.significant_t.values != 0).astype(float)
    
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


def plot_1d_clusters(t_map, null_dist, ut, lt, x_values=None, critical_mass_p=None, title=''):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    t_map_gpu = cp.asarray(t_map)

    x = np.arange(len(t_map)) if x_values is None else x_values
    ax1.plot(x, t_map, color='black', lw=1, label='T-statistic')
    
    ut_cpu = cp.asnumpy(ut) if isinstance(ut, cp.ndarray) else np.full(x.shape, ut)
    lt_cpu = cp.asnumpy(lt) if isinstance(lt, cp.ndarray) else np.full(x.shape, lt)

    mass_05 = np.percentile(null_dist, 95)
    sig_mask_05 = cp.zeros(t_map_gpu.shape, dtype=bool)
    
    do_custom = critical_mass_p is not None and not np.isclose(critical_mass_p, 0.05)
    if do_custom:
        mass_custom = np.percentile(null_dist, 100 * (1 - critical_mass_p))
        sig_mask_custom = cp.zeros(t_map_gpu.shape, dtype=bool)
    
    pos_l, n_p = cp_ndimage.label(t_map_gpu > cp.asarray(ut))
    neg_l, n_n = cp_ndimage.label(t_map_gpu < cp.asarray(lt))
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
        for i in range(data.shape[0]):
            ax.plot(x_values, data[i, :], color=color, alpha=0.5, lw=0.8, zorder=1)
            
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
        
    if title:
        ax.set_title(title)
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel("Frequency (Hz)")
    ax.legend()
    
    plt.tight_layout()
    return fig


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

    surface_opts = dict(showscale=False, opacity=0.3, hoverinfo='skip')

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
        eye=dict(x=1.5, y=1.5, z=1.2),
        center=dict(x=0, y=0, z=0),
        up=dict(x=0, y=0, z=1)
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
    return fig


def plot_2d_clusters_clean(t_map, sig_mask, x_axis, y_axis, title='2D Cluster Analysis'):
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.pcolormesh(
        x_axis, y_axis, t_map, 
        cmap='Spectral_r', shading='auto', vmin=-5, vmax=5
    )
    
    if sig_mask is not None and sig_mask.any():
        X, Y = np.meshgrid(x_axis, y_axis)
        ax.contour(X, Y, sig_mask, levels=[0.5], colors='black', linewidths=2)
        
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

    ax1.pcolormesh(x_axis, y_axis, m1, cmap=cmap, shading='auto', 
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