import cupy as cp
import numpy as np
from cupyx.scipy import ndimage as cp_ndimage
from cupyx import scatter_add
from scipy import stats
import xarray as xr
import os

from helpers import cwt

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


def run_1d_cluster_test_gpu(group1, group2, n_permutations=1000, p_thresh=0.05):
    g1, g2 = cp.asarray(group1), cp.asarray(group2)
    n_samples = g1.shape[-1]
    n1, n2 = g1.shape[0], g2.shape[0]
    n_total = n1 + n2
    combined = cp.concatenate([g1, g2], axis=0)
    
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
        pos_labels, n_pos = cp_ndimage.label(t_map > u_thr)
        neg_labels, n_neg = cp_ndimage.label(t_map < l_thr)
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

    pos_l, n_p = cp_ndimage.label(obs_t > ut)
    neg_l, n_n = cp_ndimage.label(obs_t < lt)
    
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
        labels, n_comp = cp_ndimage.label(mask)
        if n_comp == 0: return 0.0
        
        sums = cp.zeros(int(n_comp + 1), dtype=cp.float32)
        weights = cp.abs(t_map) if not is_pos else t_map
        scatter_add(sums, labels.ravel(), weights.ravel())
        return float(cp.max(sums[1:]))

    null_dist = cp.zeros(n_permutations)
    for p in range(n_permutations):
        t_map = perm_tmaps[p]
        null_dist[p] = max(get_max_mass_fast(t_map, ut, True), 
                           get_max_mass_fast(t_map, lt, False))

    def get_sig_clusters(t_map, thresh_map, is_pos=True):
        mask = (t_map > thresh_map) if is_pos else (t_map < thresh_map)
        labels, n_comp = cp_ndimage.label(mask)
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