import matplotlib
matplotlib.use('Agg')

import torch
import os
import random
import warnings
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader
from trainer import train_model
from utils import load_data, plot_segmentation 
from artifact_ds import ArtifactDataset, ManualTestDataset
from datetime import datetime
import helpers

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def create_visual_pages(pdf, model, loader, device, title, num_samples=5):
    model.eval()
    pop_size = len(loader.dataset)
    actual_samples = min(num_samples, pop_size)
    indices = random.sample(range(pop_size), actual_samples)
    
    with torch.no_grad():
        for idx in indices:
            x, y = loader.dataset[idx]
            x_in = x.unsqueeze(0).to(device)
            out = model(x_in)
            out_sig = torch.sigmoid(out).cpu()
            
            plot_segmentation(
                signal=x.numpy(), 
                true_mask=y.numpy() if y.dim() == 2 else y.unsqueeze(0).numpy(), 
                pred_mask=out_sig.squeeze(0).numpy(),
                title=f"{title} | Sample {idx}"
            )
            
            fig = plt.gcf()
            pdf.savefig(fig)
            plt.close(fig)

def run_sweep():
    params = {
        "lr": [1e-4],
        "dropout": [0.1, 0.2],
        "batch_size": [32]
    }
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_id = f"sweep_{datetime.now().strftime('%m%d_%H%M')}"
    base_dir = os.path.join(os.getcwd(), run_id)
    os.makedirs(base_dir, exist_ok=True)
    
    chew, _, base = load_data(
        chewing_path=os.path.join(helpers.paths.INTERMEDIATES, "manual_labels/chewing/pr"), 
        pressing_path=os.path.join(helpers.paths.INTERMEDIATES, "manual_labels/pressing/pr")
    )
    val_loader = DataLoader(ManualTestDataset(os.path.join(helpers.paths.INTERMEDIATES, "manual_labels/chewing_test_set")), batch_size=8)

    master_report_path = os.path.join(base_dir, "MASTER_SWEEP_REPORT.pdf")
    results_summary = []

    with PdfPages(master_report_path) as master_pdf:
        for lr in params["lr"]:
            for do in params["dropout"]:
                param_str = f"lr{lr}_do{do}"
                print(f"\n>>> STARTING: {param_str}")
                
                run_dir = os.path.join(base_dir, param_str)
                os.makedirs(run_dir, exist_ok=True)

                model, history = train_model(
                    chew, base, val_loader, device, 
                    batch_size=params["batch_size"][0], epochs=epochs, lr=lr, dropout=do,
                    samples=60000, crop_margin=2000, run_dir=run_dir, epoch_size=1000
                )

                # Identify Best Epoch (index + 1)
                best_val_loss = min(history['val'])
                best_epoch = history['val'].index(best_val_loss) + 1
                results_summary.append((param_str, best_val_loss, best_epoch))

                # --- Add Loss Page ---
                fig_loss, ax1 = plt.subplots(figsize=(10, 6))
                ax2 = ax1.twinx()
                ax1.plot(range(1, epochs+1), history["train"], 'g-', label="Train Loss")
                ax1.plot(range(1, epochs+1), history["val"], 'b-', label="Val Loss")
                ax2.plot(range(1, epochs+1), history["scaling"], 'r--', alpha=0.3, label="Art Ratio")
                
                # Mark best epoch on plot
                ax1.axvline(x=best_epoch, color='orange', linestyle=':', label=f'Best Ep: {best_epoch}')
                ax1.set_title(f"PARAMS: {param_str}\nBest Val Loss: {best_val_loss:.4f} at Epoch {best_epoch}")
                fig_loss.legend(loc="upper right")
                master_pdf.savefig(fig_loss)
                plt.close(fig_loss)

                # --- Reload Best Weights for Visuals ---
                best_model_path = os.path.join(run_dir, "best_model.pth")
                if os.path.exists(best_model_path):
                    model.load_state_dict(torch.load(best_model_path, map_location=device))
                    visual_title_suffix = f"BEST EPOCH {best_epoch}"
                else:
                    visual_title_suffix = "FINAL EPOCH (Best not found)"

                # --- Add Visual Samples ---
                train_vis_ds = ArtifactDataset(chew, base, 60000, epoch_size=5)
                train_vis_ds.artifact_ratio = history["scaling"][best_epoch-1]
                train_vis_loader = DataLoader(train_vis_ds, batch_size=1)
                
                create_visual_pages(master_pdf, model, train_vis_loader, device, f"{param_str} | {visual_title_suffix} (TRAIN)")
                create_visual_pages(master_pdf, model, val_loader, device, f"{param_str} | {visual_title_suffix} (TEST)")
                
                print(f">>> FINISHED: {param_str} (Best Loss: {best_val_loss:.4f} @ Ep {best_epoch})")

    # Write Top 5 Models to text file with Epoch info
    results_summary.sort(key=lambda x: x[1])
    ranking_path = os.path.join(base_dir, "top_models.txt")
    with open(ranking_path, "w") as f:
        f.write(f"Top 5 Models by Validation Loss (Run: {run_id})\n")
        f.write(f"{'Rank':<5} | {'Model Configuration':<20} | {'Loss':<10} | {'Epoch':<5}\n")
        f.write("-" * 55 + "\n")
        for i, (name, loss, ep) in enumerate(results_summary[:5]):
            f.write(f"{i+1:<5} | {name:<20} | {loss:.6f} | {ep:<5}\n")

    print(f"\n[DONE] Master report: {master_report_path}")
    print(f"[DONE] Top models list: {ranking_path}")

if __name__ == "__main__":
    run_sweep()