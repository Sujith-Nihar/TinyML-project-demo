# TinyML Light vs Dark Image Classifier on OpenMV

This project implements a tiny machine learning model to classify images as "light" or "dark" using an OpenMV Cam.

- Model trained using TensorFlow
- Optimized with post-training quantization (float16)
- Deployed on OpenMV Cam H7
- Real-time inference

## Project Structure
- `training/` : Train and quantize model
- `openmv/` : MicroPython script for OpenMV
- `sample_images/` : Example images

## How to Run
1. Train the model: `python train_model.py`
2. Copy `tiny_model_quantized.tflite` and `main_openmv.py` to your OpenMV Cam.
3. See the classification result ("Dark Image" or "Bright Image") in OpenMV IDE.

## Requirements
- TensorFlow
- OpenMV IDE
