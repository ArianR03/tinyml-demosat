# Atmospheric Layer Classification

## Project Overview

## Hardware Setup

## Software & Model

## How to User
### 1. Run the payload
    - Install Arduino IDE
    - Select board: **Arduino Nano 33 BLE Sense Rev2**
    - Upload `layer_classification_tflite.ino`

### 2. Data Collection
    - Power the Arduino with a 9V Battery source
    - Data is logged in serial and saved into an openlog sd card

### 3. Post-Flight Analysis
    - Run `analyze.py` to plot **true vs. ML layer classification**
    - Example command: `py data/analyze.py`

## References
    - Arduino TensorFlow Lite
    `https://docs.arduino.cc/hardware/nano-33-ble-sense-rev2/`

    - University of Wyoming Atmospheric Science Radiosonde Archive
    `https://weather.uwyo.edu/wsgi/sounding?datetime=2025-09-03%200:00:00&id=72476&src=BUFR&type=TEXT:LIST`

    - Research Paper
