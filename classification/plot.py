import os
import matplotlib.pyplot as plt


def plot_losses(model_name, train_losses, validation_losses):
    """
    Plots the training and validation losses over epochs.
    """

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.title("Training and Validation Loss over Epochs")
    plt.grid(True)

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.legend()

    plt.ylim(bottom=0)
    
    plt.savefig(
        os.path.join(
            os.path.dirname(__file__), "loss_plot_" + model_name + ".png"
        )
    )

    plt.show()