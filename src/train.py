"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
from src.jax_extras import cross_loss_fn

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
INPUT_SHAPE = 40

# Setup directories
train_dir = "datasets/X_train.npy"
train_labels_dir = "datasets/y_train.npy"
test_dir = "datasets/X_test.npy"
test_labels_dir = "datasets/y_test.npy"



# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    train_labels_dir=train_labels_dir,
    test_dir=test_dir,
    test_labels_dir=test_labels_dir,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
nn_init, model = model_builder.MLP_stax(
    input_shape=INPUT_SHAPE,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names),
    batch_size = BATCH_SIZE
)

# Set loss and optimizer
loss_fn = cross_loss_fn
optimizer = opt_init, opt_update, get_params = optimizers.momentum(LEARNING_RATE, 0.9)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS
             )

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
