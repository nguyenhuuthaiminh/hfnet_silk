import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from tqdm import tqdm

from training.config import SAVE_DIR
from training.utils import desc4silk

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

def train_model(
    model,
    train_loader,
    val_loader,
    config,
    lr=1e-3,
    patience=5,
    milestone=[7],
    epochs=20,
    log_file="training_log.csv"
):
    """
    Trains and validates the model, logging progress to a CSV file.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training set.
        val_loader (DataLoader): DataLoader for validation set.
        config (dict): Configuration dictionary for the model/loss.
        lr (float): Initial learning rate.
        patience (int): Number of epochs to wait for improvement before early stopping.
        epochs (int): Total number of epochs to train.
        log_file (str): File path for CSV logging.

    Returns:
        None. The function saves checkpoints to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set up TensorBoard writer
    writer = SummaryWriter(log_dir=SAVE_DIR+"runs")


    # Optimizer and learning rate scheduler
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones= milestone, gamma = 0.1)

    # Set up early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Prepare CSV logging
    columns = ['epoch', 'global_desc_l2', 'local_desc_l2', 'detector_crossentropy',
               'train_loss', 'val_loss', 'learning_rate']
    pd.DataFrame(columns=columns).to_csv(SAVE_DIR+log_file, index=False)
    
    # Main training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        metrics = {'global_desc_l2': 0.0, 'local_desc_l2': 0.0, 'detector_crossentropy': 0.0}
        
        # ----------------------------
        #       Training Phase
        # ----------------------------
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images = images.to(device)
            inputs = {k: v.to(device) for k, v in labels.items()}
    
            optimizer.zero_grad()
            
            # Forward pass
            ret = model(images)
            outputs = desc4silk(ret,inputs) # Example function call for post-processing
            # Compute total loss
            total_loss, loss_dict, logvar = model._compute_loss(inputs, outputs, config)
            
            # Log metrics to TensorBoard
            writer.add_scalar('Loss/Train', total_loss.item(), epoch)
            writer.add_scalar('Loss/Global', loss_dict['global_desc_l2'], epoch)
            writer.add_scalar('Loss/Local', loss_dict['local_desc_l2'], epoch)
            writer.add_scalar('Loss/Detector', loss_dict['detector_crossentropy'], epoch)
            writer.flush()  

            # Backprop and update
            total_loss.backward()
            optimizer.step()

            # Accumulate training metrics
            train_loss += total_loss.item()
            metrics['global_desc_l2'] += loss_dict['global_desc_l2']
            metrics['local_desc_l2'] += loss_dict['local_desc_l2']
            metrics['detector_crossentropy'] += loss_dict['detector_crossentropy']

        # Normalize training metrics by number of batches
        num_batches = len(train_loader)
        for key in metrics:
            metrics[key] /= num_batches
        train_loss /= num_batches

        # Clear unused GPU memory (optional)
        torch.cuda.empty_cache()
        
        # ----------------------------
        #       Validation Phase
        # ----------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]"):
                images = images.to(device)
                inputs = {k: v.to(device) for k, v in labels.items()}
                
                # Forward pass
                ret = model(images)
                outputs = desc4silk(ret,inputs)
                
                # Compute validation loss
                total_loss, loss_dict, _ = model._compute_loss(inputs, outputs, config)
                val_loss += total_loss.item()
        # visualize_heatmaps(outputs['logits'],inputs['keypoint_map'])
        val_loss /= len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ----------------------------
        #       Logging
        # ----------------------------
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Global L2: {metrics['global_desc_l2']:.4f} | "
            f"Local L2: {metrics['local_desc_l2']:.4f} | "
            f"Detector CE: {metrics['detector_crossentropy']:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # (Optional) Log any extra info from logvar
        if logvar:
            extra_info = " | ".join([f"{k}: {v:.4f}" for k, v in logvar.items()])
            print(extra_info)

        # Save metrics to CSV
        row_data = [
            epoch,
            metrics['global_desc_l2'],
            metrics['local_desc_l2'],
            metrics['detector_crossentropy'],
            train_loss,
            val_loss,
            current_lr
        ]
        df = pd.DataFrame([row_data], columns=columns)
        df.iloc[:, 1:] = df.iloc[:, 1:].astype(float).round(4)  # Format to 4 decimals
        df.to_csv(log_file, mode='a', header=False, index=False)

        # ----------------------------
        #   Early Stopping Check
        # ----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_DIR+"best_model.pth")
        else:
            patience_counter += 1
            torch.save(model.state_dict(), SAVE_DIR+"last_model.pth")
            if patience_counter >= patience:
                writer.close()
                print("Early stopping triggered.")
                break
