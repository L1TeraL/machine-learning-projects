# YOLO Object Detection

An object detection project using the Ultralytics YOLO framework.

## Project Overview

The project demonstrates how to prepare a dataset, train an object detection model, evaluate its performance, and make predictions on images.

## Features

- Dataset preparation and splitting
- Object detection using YOLO
- Model training
- Model evaluation
- Predictions on test images

## Technologies

- Python
- Ultralytics YOLO
- PyYAML
- Matplotlib
- Pillow

## Dataset Setup

Place the downloaded and extracted dataset in a directory named `dataset` inside this project directory.

The dataset should contain images and corresponding labels in a supported YOLO format, or a compatible dataset YAML configuration.

The dataset and model weights are not included in this repository.

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the script from this directory:

```bash
python main.py
```

The script may download pretrained model weights when needed.

## Notes

Training time depends on the dataset size and available hardware. Generated training files and model weights are not included in the repository.