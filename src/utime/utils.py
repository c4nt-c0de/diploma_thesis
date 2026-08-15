import matplotlib.pyplot as plt
import numpy as np
import os

def plot_segmentation(signal, true_mask, pred_mask=None, title=None, save_path=None):
    if pred_mask is not None:
        target_len = pred_mask.shape[1]
        input_len = signal.shape[1]
        if input_len != target_len:
            diff = (input_len - target_len) // 2
            signal = signal[:, diff : diff + target_len]
            true_mask = true_mask[:, diff : diff + target_len]

    n_channels = signal.shape[0]
    fig, axes = plt.subplots(n_channels + 2, 1, figsize=(10, 8), sharex=True)
    time = np.arange(signal.shape[1])
    
    for i in range(n_channels):
        axes[i].plot(time, signal[i], color='black', lw=0.5)
        axes[i].set_ylabel("", fontsize=8)
        axes[i].set_yticks([]) 

    axes[-2].step(time, true_mask[0], color='red', label='GT Chew', alpha=0.8)
    axes[-2].set_ylabel("GT", fontsize=15)
    axes[-2].set_ylim(-0.1, 1.1)
    axes[-2].set_yticks([0, 1])

    if pred_mask is not None:
        axes[-1].plot(time, pred_mask[0], color='red', label='Pred Chew')
    
    axes[-1].set_ylabel("Model Output", fontsize=15)
    axes[-1].set_ylim(-0.1, 1.1)
    axes[-1].set_yticks([0, 1])
    
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close(fig) 
    else:
        plt.show()

def load_data(chewing_path, pressing_path, baseline_path=None):
    import helpers
    baseline_path = baseline_path or helpers.paths.BASELINE_FOLDER
    
    def load_with_masks(path):
        files = helpers.xutils.LFPAccessor.list_files(path)
        lib = []
        for f in files:
            with helpers.xutils.LFPAccessor.load_file(f) as da:
                data_np, _ = da.lfp.channels_to_numpy()
                mask = None
                if 'chewing_mask' in da.coords:
                    mask = da.coords['chewing_mask'].values
                elif hasattr(da, 'data_vars') and 'chewing_mask' in da.data_vars:
                    mask = da['chewing_mask'].values
                elif getattr(da, 'name', None) == 'chewing_mask':
                    mask = da.values
                
                if mask is None:
                    mask = np.zeros(data_np.shape[1])
                
                if mask.ndim == 1:
                    mask = mask[np.newaxis, :]
                lib.append((data_np.astype(np.float32), mask.astype(np.float32)))
        return lib

    def load_snippets_only(path):
        files = helpers.xutils.LFPAccessor.list_files(path)
        snippets = []
        for f in files:
            with helpers.xutils.LFPAccessor.load_file(f) as da:
                data_np, _ = da.lfp.channels_to_numpy()
                if data_np.shape[0] > data_np.shape[1]:
                    data_np = data_np.T
                snippets.append(data_np.astype(np.float32))
        return snippets

    def load_canvas(path):
        files = helpers.xutils.LFPAccessor.list_files(path)
        segments = []
        for f in files:
            with helpers.xutils.LFPAccessor.load_file(f) as da:
                data_np, _ = da.lfp.channels_to_numpy()
                segments.append(data_np.astype(np.float32))
                print(data_np.shape)
        canvas = np.concatenate(segments, axis=0)
        return canvas.T

    chew_snippets = load_snippets_only(chewing_path)
    rec_libs = load_with_masks(chewing_path) + load_with_masks(pressing_path)
    canvas = load_canvas(baseline_path)

    return chew_snippets, rec_libs, canvas