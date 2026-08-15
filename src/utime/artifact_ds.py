import numpy as np
import random
import torch
from torch.utils.data import Dataset
import helpers

class ArtifactDataset(Dataset):
    def __init__(self, chewing_lib, baseline_canvas, recording_libs, samples, epoch_size, clean_prob=0.5, fs=1000):
        self.chew_lib = chewing_lib
        self.baseline_canvas = baseline_canvas
        self.recording_libs = recording_libs 
        self.samples = samples
        self.epoch_size = epoch_size
        self.clean_prob = clean_prob
        self.artifact_ratio = 1.0  
        
        self.fs = fs
        self.buffer = 30 * fs # 30s buffer

    def __len__(self):
        return self.epoch_size

    def _get_quiet_segment_from_lib(self):
        if not self.recording_libs: return None
        rec_idx = random.randint(0, len(self.recording_libs) - 1)
        data, mask = self.recording_libs[rec_idx]
        
        total_len = data.shape[1]
        needed = self.samples + (2 * self.buffer) # 60s + 30s + 30s
        
        if total_len < needed: return None 

        for _ in range(50):
            start = random.randint(0, total_len - needed)
            # Check for ANY events in the window + buffers
            if np.sum(mask[:, start : start + needed]) == 0:
                return data[:, start + self.buffer : start + self.buffer + self.samples].copy()
        return None

        for _ in range(50):
            start = random.randint(0, total_len - needed)
            # Strict check: No events in the window OR the 30s buffers
            if np.sum(mask[:, start : start + needed]) == 0:
                # Return only the center 60s
                return data[:, start + self.buffer : start + self.buffer + self.samples].copy()
        
        return None

    def __getitem__(self, idx):
        # 50/50 Source: Clean Baseline Canvas vs Quiet slices from PR/CR recordings
        x = None
        if random.random() > 0.5:
            x = self._get_quiet_segment_from_lib()

        if x is None:
            total_len = self.baseline_canvas.shape[1]
            start_idx = random.randint(0, total_len - self.samples)
            x = self.baseline_canvas[:, start_idx : start_idx + self.samples].copy()
        
        y = np.zeros((1, self.samples), dtype=np.float32) 
        
        # 50/50 Clean vs Injected
        if random.random() > self.clean_prob:
            x, mask = self._inject(x, self.chew_lib)
            y[0, :] = mask
        
        return torch.from_numpy(x), torch.from_numpy(y)

    def _inject(self, x, lib):
        mask = np.zeros(x.shape[1], dtype=np.float32)
        if not lib: return x, mask
            
        n_events = random.randint(2, 6)
        for _ in range(n_events):
            art = random.choice(lib).copy()
            art_len = art.shape[1]
            if art_len >= x.shape[1]: continue
            
            scale = random.uniform(0.5, 1.2) * self.artifact_ratio
            window = self._get_tukey_window(art_len, alpha=0.1)
            p = random.randint(0, x.shape[1] - art_len)
            
            x[:, p : p + art_len] += (art * scale * window)
            mask[p : p + art_len] = 1.0
            
        return x, mask

    def _get_tukey_window(self, n, alpha=0.1):
        win = np.ones(n, dtype=np.float32)
        m = int(alpha * (n - 1) / 2)
        if m <= 0: return win
        pos = np.arange(m)
        taper = 0.5 * (1 + np.cos(np.pi * (pos / m - 1)))
        win[:m] = taper
        win[-m:] = taper[::-1]
        return win
    
class ManualTestDataset(Dataset):
    def __init__(self, folder_path):
        self.files = list(helpers.xutils.LFPAccessor.list_files(folder_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with helpers.xutils.LFPAccessor.load_file(self.files[idx]) as da:
            x, _ = da.lfp.channels_to_numpy()
            x = x.astype(np.float32)
            
            # --- ADD THIS TRANSPOSE CHECK ---
            if x.shape[0] > x.shape[1]:
                x = x.T
            # --------------------------------
            
            mask = None
            # ... rest of your mask detection logic ...
            if 'chewing_mask' in da.coords:
                mask = da.coords['chewing_mask'].values
            elif hasattr(da, 'data_vars') and 'chewing_mask' in da.data_vars:
                mask = da['chewing_mask'].values
            elif getattr(da, 'name', None) == 'chewing_mask':
                mask = da.values
            
            if mask is None:
                mask = np.zeros(x.shape[1]) # Now x.shape[1] is time
            
            y = mask.astype(np.float32)
            if y.ndim == 1:
                y = y[np.newaxis, :]
            
            return torch.from_numpy(x), torch.from_numpy(y)