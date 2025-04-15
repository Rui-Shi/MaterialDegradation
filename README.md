# Identify Material Degradation by Deep Learning Approach

## Project Description

This project investigates the use of deep learning, specifically Convolutional Neural Networks (CNNs), to identify and classify material degradation, focusing on crack generation in polymer materials (Underfills - UF) used in microelectronic packaging. The degradation is often caused by long-term exposure to high temperatures. The study aims to establish a connection between the material's microstructure evolution (observed via optical microscope images) and its mechanical properties, a relationship difficult to quantify using traditional physical approaches.

The project utilizes image classification techniques on a dataset of microstructure images, comparing the performance of traditional Artificial Neural Networks (ANN) and CNNs for identifying the presence and potentially the number of cracks. Techniques like data augmentation and dropout were employed to mitigate overfitting issues common with small datasets.

## Folder Contents

* **`Final Project Report.pdf`**: Detailed report covering the project background, methodology (ANN and CNN models), data processing, results, analysis (including overfitting), and conclusions.
* **`Identify Material Degradation by Deep Learning Approach.pptx`**: Presentation slides summarizing the project's motivation, methods, results, and challenges.
* **`Project Proposal.pdf`**: Initial proposal outlining the project's goals and approach.
* **`Py_code/`**: Directory containing the Python scripts.
    * `code_Rui Shi.py`: Python script implementing and comparing ANN and CNN models using TensorFlow/Keras for crack classification based on image directories. Includes code for loading images, splitting data, building models, training, and evaluating accuracy.
    * `code_Yunli Zhang.py`: Python script using TensorFlow/Keras for image classification. Includes data loading (`image_dataset_from_directory`), preprocessing, model building (CNN), training, evaluation, data augmentation, and dropout techniques.
    * `test2.py`: Example script using the Fashion MNIST dataset with TensorFlow/Keras.
    * `test3.py`: Example script using a flower image dataset with TensorFlow/Keras for image classification.

## Code Overview

The primary code for the analysis resides in the `Py_code` directory.
* `code_Rui Shi.py` reads images from specified directories (categorized by the number of cracks), preprocesses them, builds both an ANN and a CNN model, trains them on the image data, and evaluates their performance.
* `code_Yunli Zhang.py` utilizes TensorFlow's image dataset loading utilities, applies data augmentation and dropout, builds a CNN model, and trains/evaluates it for classifying images into categories.

## Dependencies

Based on the Python scripts, the following libraries are required:
* TensorFlow (`tf`, `tensorflow.keras`)
* Matplotlib (`matplotlib.pyplot`)
* NumPy (`numpy`)
* OpenCV (`cv2`)
* Pillow (`PIL`)
* os
* pathlib
* random
* tqdm

## Usage

1.  **Environment Setup**: Ensure Python and the required dependencies (TensorFlow, Matplotlib, NumPy, OpenCV, Pillow) are installed. Setting up TensorFlow might require specific environment configurations.
2.  **Data Preparation**: The scripts expect image data to be organized into specific directories (e.g., "No crack", "One crack", etc. in `code_Rui Shi.py` or a main directory with subdirectories for classes in `code_Yunli Zhang.py`). Update the file paths within the scripts to point to the correct data locations.
3.  **Execution**: Run the Python scripts (`code_Rui Shi.py` or `code_Yunli Zhang.py`) using a Python interpreter. The scripts will load data, build the models, train them, and output accuracy/loss plots and final evaluation metrics.

## Authors

* Yunli Zhang
* Rui Shi

## References

* The Degradation Mechanisms of Underfills Subjected to Isothermal Long-Term Aging
* Python for Everyone, 2nd edition, Cay Horstmann Rance Necaise
