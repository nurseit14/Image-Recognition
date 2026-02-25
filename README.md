# Image Recognition on CIFAR-10 🖼️

A complete pipeline for training a Custom Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch. This project features data augmentation, a flexible CNN architecture, and an automated hyperparameter grid search to identify the optimal model configuration.

## 📌 Important Notes

*   **Dataset Not Included**: To keep the repository lightweight, the CIFAR-10 dataset is not included in the uploaded files. However, the notebook uses `torchvision` to automatically download the dataset into a local `./data` directory when executed.
*   **No Screenshots**: Visuals and screenshots of the training plots have intentionally been omitted from this repository. You can generate and view these plots (loss and accuracy curves) by running the evaluation cells at the end of the notebook.

## 🚀 Features

*   **Custom CNN Architecture**: A straightforward but effective 3-block CNN utilizing Convolutional layers, Batch Normalization, MaxPooling, and Dropout.
*   **Hyperparameter Grid Search**: Automates the search across multiple configurations (optimizing Batch Size, Optimizer type [Adam vs SGD], Learning Rate, Weight Decay, and Dropout).
*   **Dynamic Checkpointing & Early Stopping**: Automatically tracks validation accuracy, saves the best model state, and halts training if no improvement is observed for a defined patience period.
*   **Data Augmentation**: Employs `RandomCrop` and `RandomHorizontalFlip` to improve model generalization.

## 🛠️ Requirements

Make sure you have the following libraries installed:
*   Python 3.x
*   [PyTorch](https://pytorch.org/) & Torchvision
*   NumPy
*   Matplotlib

## 🏃‍♂️ How to Run

1. Clone the repository to your local machine.
2. Launch Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook Image_Recognition.ipynb
   ```
3. Run all cells in the notebook. 
   *   The script will first download the CIFAR-10 dataset (~170MB) into the `./data` directory.
   *   It will then evaluate all combinations defined in the hyperparameter search space.
   *   Model checkpoints will be saved in the `./checkpoints` directory.
   *   Once training is complete, the notebook will output the best run's summary and plot the loss/accuracy curves.

## 📊 Model Details

The underlying neural network is defined as `SimpleCNN` and consists of:
*   **Feature Extractor**: Three progressive convolutional blocks (32 -> 64 -> 128 channels) using 3x3 kernels.
*   **Classifier**: Fully connected layers reducing dimensions down to the 10 target classes of the CIFAR-10 dataset.
