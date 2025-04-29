# TinyML Light vs Dark Image Classifier on OpenMV

![TinyML](https://docs.openmv.io/_images/pinout-openmv-cam-h7-plus-ov5640.png) ![License](https://img.shields.io/badge/License-MIT-green.svg)

This project implements a tiny machine learning model to classify images as "light" or "dark" using an OpenMV Cam.

- Model trained using TensorFlow
- Optimized with post-training quantization (float16)
- Deployed on OpenMV Cam H7
- Real-time inference

## Project Structure
- `training/` : Train and quantize model
- `openmv/` : MicroPython script for OpenMV

## How to Run
1. Train the model: `python train_model.py`
2. Copy `tiny_model_quantized.tflite` and `main_openmv.py` to your OpenMV Cam.
3. See the classification result ("Dark Image" or "Bright Image") in OpenMV IDE.

## Requirements
- TensorFlow
- OpenMV IDE
