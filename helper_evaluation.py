import matplotlib.pyplot as plt
import itertools
import os
import random
import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras.models import Model

def plot_training_history(history: dict, figsize: tuple[int, int]) -> None:
    """
    Plots the loss and accuracy curves for training and validation in a vertical layout.

    This function accepts a model history dictionary and plots the training and validation loss,
    as well as the accuracy over the epochs. It creates two separate plots: one for the loss and
    another for the accuracy, allowing for a clear visualization of the model's performance over time.

    Args:
        history (dict): A model history dictionary containing the history of training/validation loss and accuracy,
                        recorded at the end of each epoch. Expected keys are 'loss', 'val_loss', 'accuracy', and 'val_accuracy'.
        figsize (tuple[int, int]): A tuple specifying the width and height in inches of the figure to be plotted.
                                   This allows customization of the plot size for better readability and fitting into different contexts.

    Returns:
        None: This function does not return any value. It generates and displays matplotlib plots, visualizing the
              training and validation loss and accuracy over epochs.

    Example usage:
        plot_training_history(history, figsize=(10, 10))
    """
    loss = history["loss"]
    val_loss = history["val_loss"]

    accuracy = history["accuracy"]
    val_accuracy = history["val_accuracy"]

    epochs = range(1, len(history["loss"]) + 1)  # Start epochs at 1

    # Plotting setup for a vertical layout
    plt.figure(figsize=figsize)  # Use provided figure size

    # Plot loss
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot = loss
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot = accuracy
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()  # Adjust layout to not overlap
    plt.show()


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray = None,
    figsize: tuple[int, int] = (10, 10),
    text_size: int = 15,
    cmap: str = "Blues",
    norm: bool = False,
    savefig: bool = False,
) -> None:
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels, with options to normalize
    and save the figure.

    Args:
      y_true (np.ndarray): Array of truth labels (must be same shape as y_pred).
      y_pred (np.ndarray): Array of predicted labels (must be same shape as y_true).
      classes (np.ndarray): Array of class labels (e.g., string form). If `None`, integer labels are used.
      figsize (tuple[int, int]): Size of output figure (default=(10, 10)).
      text_size (int): Size of output figure text (default=15).
      norm (bool): If True, normalize the values in the confusion matrix (default=False).
      savefig (bool): If True, save the confusion matrix plot to the current working directory (default=False).

    Returns:
        None: This function does not return a value but displays a Confusion Matrix. Optionally, it saves the plot.

    Example usage:
      make_confusion_matrix(y_true=test_labels, # ground truth test labels
                            y_pred=y_preds, # predicted labels
                            classes=class_names, # array of class label names
                            figsize=(15, 15),
                            text_size=10,
                            norm=True,
                            savefig=True)
    """
    # Create the confusion matrix
    cm = (
        confusion_matrix(y_true, y_pred, normalize="true")
        if norm
        else confusion_matrix(y_true, y_pred)
    )

    # Plot the figure
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)

    # Set class labels
    if classes is not None:
        labels = classes
    else:
        labels = np.arange(len(cm))

    # Set the labels and titles
    ax.set(
        title="Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
    )
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()
    plt.xticks(rotation=70, fontsize=text_size)
    plt.yticks(fontsize=text_size)

    # Annotate the cells with the appropriate values
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            f"{cm[i, j]:.2f}" if norm else f"{cm[i, j]}",
            horizontalalignment="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black",
            size=text_size,
        )

    plt.tight_layout()
    # Save the figure if requested
    if savefig:
        plt.savefig("confusion_matrix.png")
    plt.show()


def show_model_prediction_time(
    model: Model, samples: np.ndarray, figsize: tuple[int, int] = (10, 10)
) -> tuple[float, float]:
    """
    Times how long a model takes to make predictions on samples.

    Args:
        model: A trained model, capable of making predictions.
        samples: A batch of samples to predict on. Expected to be in the correct format for the model.
        figsize (tuple[int, int]): Size of output figure (default=(10, 10)).

    Returns:
        total_time (float): Total elapsed time for the model to make predictions on samples, in seconds.
        time_per_pred (float): Average time in seconds per single sample prediction.

    Example usage:
        total_time, time_per_pred = show_model_prediction_time(model, samples)
    """
    start_time = time.perf_counter()  # get start time
    model.predict(samples)  # make predictions
    end_time = time.perf_counter()  # get finish time
    total_time = end_time - start_time  # calculate how long predictions took to make
    time_per_pred = total_time / len(samples)  # find prediction time per sample

    plt.figure(figsize=figsize)
    plt.scatter(total_time, time_per_pred)
    plt.title("Time how long a model takes to make predictions on sample")
    plt.xlabel(f"Total time: {total_time:.5f} s")
    plt.ylabel(f"Time per prediction: {time_per_pred:.5f} s")
    plt.show()

    return total_time, time_per_pred