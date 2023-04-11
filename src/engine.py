"""
Contains functions for training and testing a JAX model.
"""

from tqdm.auto import tqdm
from typing import Dict, List, Tuple


def JAX_train_step(model: callable, # could be also "params: Tuple"
               loss_fn: callable, 
               optimizer: callable,
               dataloader,
               opt_state,
               lr
               ) -> Tuple:
  """Trains a JAX model for a single epoch.

    ...........

  Args:
    model: 
    dataloader: 
    loss_fn: 
    optimizer: 
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:

  """


  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  #opt_init, opt_update, get_params = optimizers.adam(step_size=learning_r)

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):

      loss_value, acc, opt_state = train_step(lr, opt_state, batch, loss_fn)

      train_loss += loss #.item()
      train_acc += acc


  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc


def JAX_test_step(model: callable, # could be also "params: Tuple"
              loss_fn: callable, 
              dataloader) -> Tuple:
  test_loss, test_acc = 0, 0
  for batch, (X, y) in enumerate(dataloader):

    test_loss_value, test_acc = test_step(opt_state, batch)

    test_loss += test_loss_value
    test_acc += test_acc

  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: callable, 
          optimizer: callable,
          loss_fn: callable,
          epochs: int,
          train_dataloader,
          test_dataloader,
          lr 
          ) -> Dict[str, List]:

    
  """T

  Args:

  Returns:
 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }

  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = JAX_train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          lr
                                          )
      test_loss, test_acc = JAX_test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn
          )

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results
