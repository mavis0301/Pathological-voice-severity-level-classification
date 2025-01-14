import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from utils.utils import mask_to_polygons, polygons_to_mask

train_dir_img = Path("CGMH/CGMH_img_mask/new/change/merge/train/img")
train_dir_mask = Path("CGMH/CGMH_img_mask/new/change/merge/train/mask")
val_dir_img = Path("CGMH/CGMH_img_mask/new/change/merge/val/img")
val_dir_mask = Path("CGMH/CGMH_img_mask/new/change/merge/val/mask")
# train_dir_img = Path("CGMH/unet_img_mask/img/train")
# train_dir_mask = Path("CGMH/unet_img_mask/mask/train")
# val_dir_img = Path("CGMH/unet_img_mask/img/val")
# val_dir_mask = Path("CGMH/unet_img_mask/mask/val")
dir_checkpoint = Path('./tmp/')

output_train_dir = Path("0113/0113_train_mask_batch_64_2")
output_val_dir = Path("0113/0113_val_mask_batch_64_2")
true_train_dir = Path("0113/0113_true_train_mask_batch_64_2")
true_val_dir = Path("0113/0113_true_val_mask_batch_64_2")

output_train_dir.mkdir(exist_ok=True)
output_val_dir.mkdir(exist_ok=True)
true_train_dir.mkdir(exist_ok=True)
true_val_dir.mkdir(exist_ok=True)


def generate_masks(model, dataloader, device, output_dir, true_output_dir):
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc='Generating masks'):
            images, masks_true, filenames = batch['image'], batch['mask'], batch['filename']
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)

            # Predict the masks
            predictions = model(images)
            predictions = F.sigmoid(predictions) > 0.5  # Apply threshold

            for i, (predicted_mask, true_mask) in enumerate(zip(predictions, masks_true)):
                predicted_mask = predicted_mask.squeeze().cpu().numpy().astype(np.uint8)
                true_mask = true_mask.cpu().numpy().astype(np.uint8)

                if np.any(predicted_mask):  # If any GA region exists, fill missing parts
                    polygons, _ = mask_to_polygons(predicted_mask)
                    predicted_mask = polygons_to_mask(polygons, predicted_mask.shape)

                # Scale mask values to 255 for visualization
                predicted_mask[predicted_mask == 1] = 255
                true_mask[true_mask == 1] = 255

                # Save the predicted and true masks as images
                predicted_output_path = output_dir / f"{filenames[i]}"
                true_output_path = true_output_dir / f"{filenames[i]}"
                Image.fromarray(predicted_mask).convert("L").save(predicted_output_path)
                Image.fromarray(true_mask).convert("L").save(true_output_path)


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.1,
        gradient_clipping: float = 1.0,
):
    train_set = BasicDataset(train_dir_img, train_dir_mask, img_scale, out_img="gray")
    val_set = BasicDataset(val_dir_img, val_dir_mask, img_scale, out_img="gray")

    n_val = len(val_set)
    n_train = len(train_set)

    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    optimizer = optim.AdamW(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    changed = False

    for epoch in range(1, epochs + 1):
        if epochs - epoch <= 20:
            logging.info("No augmentation for the last 20 epochs.")
            train_loader.dataset.augment = False

        if not changed and global_step > 1000:
            logging.info("Switching to SGD optimizer.")
            changed = True
            optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=momentum)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                logging.info(f"Epoch {epoch}/{epochs}, Step {global_step}, Loss: {loss.item()}")

            val_score = evaluate(model, val_loader, device, amp)
            scheduler.step(val_score)

            logging.info(f'Epoch {epoch} Validation Dice score: {val_score}')

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

    logging.info("Generating masks for train and validation datasets.")
    generate_masks(model, train_loader, device, output_train_dir, true_train_dir)
    generate_masks(model, val_loader, device, output_val_dir, true_val_dir)



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=80, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-2,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    if not os.path.isdir(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    logging.basicConfig(filename=f"{dir_checkpoint}/log", level=logging.DEBUG, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
