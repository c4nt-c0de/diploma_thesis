import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utime_model import UTime
from artifact_ds import ArtifactDataset
import os
import numpy as np
from utils import plot_segmentation
import pandas as pd


def train_model(chew_lib, recording_libs, baseline_data, val_loader, device, 
                batch_size, epochs, lr, dropout, samples, crop_margin, 
                run_name, epoch_size, plot_every=False, plot_train=False, plot_end=False, save_freq=10):
    
    run_dir = os.path.join(os.getcwd(), run_name)
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    train_ds = ArtifactDataset(chew_lib, baseline_data, recording_libs, samples, epoch_size=epoch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    model = UTime(in_channels=4, n_classes=1, dropout=dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            
            diff = (y.shape[2] - output.shape[2]) // 2
            y_match = y[:, :, diff : diff + output.shape[2]]
            
            loss = criterion(output[:, :, crop_margin:-crop_margin], 
                             y_match[:, :, crop_margin:-crop_margin])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_out = model(vx)
                diff = (vy.shape[2] - v_out.shape[2]) // 2
                vy_match = vy[:, :, diff : diff + v_out.shape[2]]
                v_loss = criterion(v_out[:, :, crop_margin:-crop_margin], 
                                 vy_match[:, :, crop_margin:-crop_margin])
                val_loss += v_loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if plot_every and (epoch + 1) % 10 == 0:
            vx, vy = next(iter(val_loader))
            vx, vy = vx.to(device), vy.to(device)
            v_out = model(vx)
            
            pred = torch.sigmoid(v_out).detach().cpu().numpy()[0]
            sig = vx.detach().cpu().numpy()[0]
            gt = vy.detach().cpu().numpy()[0]
            
            fname = os.path.join(plot_dir, f"epoch_{epoch+1}.png")
            #plot_segmentation(sig, gt, pred, title=f"Epoch {epoch+1}", save_path=fname)
            plot_segmentation(sig, gt, pred, save_path=fname)
        
        history.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})

        if (epoch + 1) % save_freq == 0 or (epoch + 1) == epochs:
            torch.save(model.state_dict(), os.path.join(run_dir, f"model_ep{epoch+1}.pth"))
            pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
            
    return model