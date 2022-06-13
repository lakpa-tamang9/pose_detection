#!/bin/sh
#singlepose thunder
mkdir models
cd models
wget -O thunder.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/3?lite-format=tflite 

# singlepose lightning
wget -O lightning.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3?lite-format=tflite