# Atmospheric Layer Classification (TinyML DemoSat 2025)

A **TinyML-based atmospheric sensing payload** designed to classify atmospheric layers in real time using an **Arduino Nano 33 BLE Sense Rev2** and **BME280 environmental sensor**. The system integrates embedded machine learning (TensorFlow Lite) with onboard data logging for post-flight analysis. 

---

## Project Overview

This project demonstrates how **machine learning can run on low-power MCUs** to interpret environmental data. During flight, the system continuously reads **pressure** and **temperature** data, classifies the current **atmospheric layer**, and logs both the raw and inferred data to an SD card for later validation.

**Key objectives**:
- Implement a deployable TinyML model on an embedded board.
- Classify real-time sensor data into atmospheric regions (e.g., Troposphere, Stratosphere).
- Validate classifications against reference radiosonde data from the **University of Wyoming** archive. 

---

## Hardware Setup
| Component | Description |
|-----------|-------------|
| **Arduino Nano 33 BLE Sense Rev2** | Main MCU for sensor data processing and ML inference |
| **BME280 Sensor** | Measures temperature and barometric pressure |
| **OpenLog Data Logger** | Logs serial output via UART to microSD for post-flight analysis |
| **9V Battery Supply** | Provides power during standalone operation |

---

## Software & Model

The onboard software (`layer_classification_tflite.ino`) handles:
- Sensor initialization and data acquisition
- Real-time TensorFlow Lite inference
- Serial and SD card data logging

The model was trained using historical radiosonde datasets and converted to a **TensorFlow Lite (.tflite)** format for embedded deployment.

**Model Workflow:**
1. Collect atmospheric sensor data
2. Train a neural network classifier in python
3. Convert to TensorFlow Lite model
4. Embed `.tflite` file into Arduino firmware

--- 

## How to Use

### 1. Run the payload
1. Install the **Arduino IDE**
2. Download the required libraries for the MCU.
3. Select the board: `Arduino Nano 33 BLE Sense Rev2`
4. Open and upload `layer_classification_tflite.ino`

### 2. Data Collection
- Power the Arduino using a **9V battery**
- Data will be printed via Serial and saved automatically to the SD card using **OpenLog**
