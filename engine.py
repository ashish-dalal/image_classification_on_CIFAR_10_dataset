import torch
from tqdm.auto import tqdm

from typing import Dict, List, Tuple

def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device) -> Tuple[float, float]:

    """
    Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy).
    """

    # model in train mode
    model.train()

    # setup train and accuracy
    train_loss, train_accuracy = 0, 0

    # Loop through dataloader batches
    for batch, (X,y) in enumerate(dataloader):

        # send data to target device
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X)

        # Calculate accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # optimizer zero
        optimizer.zero_grad()

        # backprop
        loss.backward()

        # optimizer step
        optimizer.step()

        # calculate accumulate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_accuracy += (y_pred_class == y).sum().item() / len(y_pred)

    # loss accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_accuracy = train_accuracy / len(dataloader)

    return train_loss, train_accuracy


def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device) -> Tuple[float, float]:

    """
    Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy).
    """

    # put model in eval mode
    model.eval()

    # setup loss and accuracy
    test_loss, test_accuracy = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():

        # Loop through dataloader batches
        for batch, (X,y) in enumerate(dataloader):

            # send data to device
            X, y = X.to(device), y.to(device)

            # Forward pass 
            test_pred_logits = model(X)

            # calculate accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # calculate accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_accuracy += (test_pred_labels == y).sum().item() / len(test_pred_labels)
        
        # loss accuracy per batch
        test_loss /= len(dataloader)
        test_accuracy /= len(dataloader)

        return test_loss, test_accuracy


def train(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device) -> Dict[str, List]:

    """
    Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    """

    # create result dict
    results = {"train_loss":[],
              "train_accuracy":[],
              "test_loss":[],
              "test_accuracy":[]}
    
    for epoch in tqdm(range(epochs)):
        
        # train step
        train_loss, train_accuracy = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                device=device)

        test_loss, test_accuracy = test_step(model=model,
                                             dataloader=test_dataloader,
                                             loss_fn=loss_fn,
                                             device=device)
        
        # Print out what's happening
        print(
            f"\nEpoch: {epoch+1} |\n-------------------------------------------------------------------------------\n"
            f"train_loss: {train_loss:.4f} | "
            f"train_accuracy: {train_accuracy:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_accuracy: {test_accuracy:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_accuracy)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_accuracy)

    return results