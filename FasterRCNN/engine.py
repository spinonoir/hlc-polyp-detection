from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from src.config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUTPUT_DIR
from src.config import VISUALIZE_AFTER_TRANSFORM, SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from src.model import *
from src.utils import Averager
from tqdm import tqdm
from src.PolypDataset import get_dataloaders
import torch
import matplotlib.pyplot as plt
import time
plt.style.use('ggplot')

# The training loop function
def train(train_data_loader, model):
    print('Training...')
    global train_iter
    global train_losses

    # THE LOOP w/Beautiful progress bar
    with tqdm(total=len(train_data_loader)) as pbar:
        for data in train_data_loader:
            # Get the images and targets from the data loader
            images, targets = data
            
            # Move the images and targets to the GPU
            images = list(image.to(DEVICE) for image in images)
            for target in targets:
                target['boxes'] = target['boxes'].to(DEVICE)
                target['labels'] = target['labels'].to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            loss_dict = model(images, targets)

            # Get the loss
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            losses.backward()

            # Update the weights
            optimizer.step()

            # Update the progress bar
            pbar.set_description(f"Loss: {losses.item():.4f}")
            # pbar.update(1)
            # Update the losses
            train_loss_hist.send(losses.item())
            train_iter += 1

    return train_losses 

# The validation loop function
def validate(val_data_loader, model):
    print('Validating...')
    global val_iter
    global val_losses

    # THE LOOP w/Beautiful progress bar
    with tqdm(total=len(val_data_loader)) as pbar:
        for data in pbar:
            # Get the images and targets from the data loader
            images, targets = data

            # Move the images and targets to the GPU
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            # Forward pass
            with torch.no_grad():
                loss_dict = model(images, targets)

            # Get the loss
            losses = sum(loss for loss in loss_dict.values())
            val_losses = losses.item()
    

            # Update the progress bar
            pbar.set_description(f"Loss: {losses.item():.4f}")
            # pbar.update(1)

            # Update the losses
            val_loss_hist.send(losses.item())
            val_iter += 1

    return val_losses

# The main function
if __name__ == '__main__':
    # Initialize the model and move to GPU (if available)
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    # Obtain model parameters to be optimized/updated in this run.
    params = filter(lambda p: p.requires_grad, model.parameters())

    # Define the optimizer
    # TODO: Try out alternatives to SGD --> Maybe use the ABC algorithm 
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)

    # Initialize training loss tracking variables for plotting
    train_loss_hist = Averager()
    train_iter = 1
    train_losses = []
    # Initialize validation loss tracking variables for plotting
    val_loss_hist = Averager()
    val_iter = 1
    val_losses = []

    # Give the model a name :-)
    MODEL_NAME = 'polyps_model_1'

    
    train_loader, valid_loader = get_dataloaders()

    # Show transformed images if VISUALIZE_AFTER_TRANSFORM is True
    # TODO: Don't use this until we have rewritten the show_transformed_images function
    # to work with pyplot instead of cv2
    if VISUALIZE_AFTER_TRANSFORM:
        from src.utils import show_transformed_image
        show_transformed_image(train_loader, model)


    # The MAIN Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch {epoch}/{NUM_EPOCHS}')

        # Reset training and validation loss histories
        train_loss_hist.reset()
        val_loss_hist.reset()

        # Prepare training and validation plots:
        figure_1, train_ax = plt.subplots()
        figure_2, val_ax = plt.subplots()

        # Start the timer and begin training and validation
        start = time.time()

        # The training loop
        train_losses = train(train_loader, model)

        # The validation loop
        val_losses = validate(valid_loader, model)

        # Print the training and validation loss
        print(f'Epoch {epoch} train loss: {train_loss_hist.value:.3f} val loss: {val_loss_hist.value:.3f}')
        end = time.time()
        print(f'Training time: {((end - start) / 60):.3f}min for {train_iter} iterations')


        if (epoch % SAVE_MODEL_EPOCH == 0) or (epoch == NUM_EPOCHS):
            # Save the model
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,f'model{epoch}.pth'))
            print(f'Saved model to {os.path.join(OUTPUT_DIR,f"model{epoch}.pth")}')
        
        if (epoch % SAVE_PLOTS_EPOCH == 0) or (epoch == NUM_EPOCHS):
            # Generate plots
            train_ax.plot(train_losses, color='blue')
            train_ax.set_xlabel('Iterations')
            train_ax.set_ylabel('Training Loss')
            val_ax.plot(val_losses, color='red')
            val_ax.set_xlabel('Iterations')
            val_ax.set_ylabel('Validation Loss')
            figure_1.savefig(os.path.join(OUTPUT_DIR,f'train_loss{epoch}.png'))
            figure_2.savefig(os.path.join(OUTPUT_DIR,f'val_loss{epoch}.png'))
            print(f'Saved plots to {os.path.join(OUTPUT_DIR,f"[train or val]_loss{epoch}.png")}')

        plt.close('all')
