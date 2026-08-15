import torch
import argparse
import os
from trainer import train_model
from utils import load_data
from artifact_ds import ManualTestDataset
from torch.utils.data import DataLoader
import helpers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--epoch_size", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--crop", type=int, default=2000)
    parser.add_argument("--plot_end", action="store_true")
    parser.add_argument("--plot_every", action="store_true")
    parser.add_argument("--plot_train", action="store_true") # NEW
    parser.add_argument("--run_name", type=str, default="chew_reg")
    parser.add_argument("--test_folder", type=str, 
                        default=os.path.join(helpers.paths.INTERMEDIATES, "manual_labels/chewing_test_set"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    chew, rec_libs, base = load_data(
        chewing_path=os.path.join(helpers.paths.INTERMEDIATES, "manual_labels/chewing/pr"), 
        pressing_path=os.path.join(helpers.paths.INTERMEDIATES, "manual_labels/pressing/pr")
    )
    print("\n--- DATA LOADING CHECK ---")
    print(f"Chewing Snippets: {len(chew)} items")
    print(f"Recording Libs:   {len(rec_libs)} items")
    print(f"Baseline Canvas:  {base.shape} (Channels, Samples)") # Expected: (16, Millions)
    print("--------------------------\n")

    val_loader = DataLoader(ManualTestDataset(args.test_folder), batch_size=1, shuffle=False)
    
    train_model(
        chew_lib=chew, 
        recording_libs=rec_libs,
        baseline_data=base, 
        val_loader=val_loader, 
        device=device,
        batch_size=args.batch_size, 
        epochs=args.epochs, 
        lr=args.lr, 
        dropout=args.dropout, 
        samples=args.samples, 
        crop_margin=args.crop,
        run_name=args.run_name, 
        epoch_size=args.epoch_size,
        plot_every=args.plot_every,
        plot_train=args.plot_train, # NEW
        plot_end=args.plot_end
    )

if __name__ == "__main__":
    main()