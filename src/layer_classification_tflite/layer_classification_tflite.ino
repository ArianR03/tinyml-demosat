/*
  Atmospheric Layer Classification (TinyML Integration)
  DemoSat 2025

  This programs intended purpose reads data from a BMP280 enviornmental sensor
  (pressure, temperature) and feeds it into a TensorFlow Lite model that will run on
  the Arduino Nano 33 BLE Sense Rev2. This model classifies the data in real-time into 
  one of several atmospheric "layers." These results are printed into serial and stored 
  for future data analysis.
*/

#include <TensorFlowLite.h> 
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include <Arduino_BMI270_BMM150.h>
#include "layer_model.h"

#include <cmath>

#define PROBLEM_LED 2

// Create sensor object for BME280
Adafruit_BME280 bme;
unsigned long timeStamp;

namespace{
  /*
    Global variables used by TensorFlow Lite:
      - error_reporter: handles debug/error messages
      - model: pointer of the TFLite model stored in directory.
      - interpreter: runs the model on inputs to produce outputs.
      - model_input/model_output: handle input/output tensors
      - tensor_arena: memory buffer where tensors for model input/output are stored.
  */

  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  
  // Atmospheric Constants used to calculate altitude
  constexpr float SEA_LEVEL_PRESSURE = 1013.25;
  constexpr float LAPSE_RATE = 0.0065;
  constexpr float EXPONENT = 0.190284;
}

void blink_led(){
  digitalWrite(PROBLEM_LED, LOW);
  delay(50);
  digitalWrite(PROBLEM_LED, HIGH);
  delay(50);
}

void setup() {
  /*
    Setup function runs once on startup:
      - Start Serial communication at 9600 baud
      - Load and verify TFLite model
  */

  // USB Connection
  Serial.begin(9600);
  
  // UART Connection for storing data in OpenLog
  Serial1.begin(9600);
  Wire.begin();
  delay(1000);

  pinMode(PROBLEM_LED, OUTPUT);
  digitalWrite(PROBLEM_LED, LOW);

  // Setup error reporter for error validation
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Get converted TFLite model
  model = tflite::GetModel(layer_model);
  
  // Ensure that the model version matches schema
  if (model->version() != TFLITE_SCHEMA_VERSION){
    error_reporter->Report("Model version does not match schema.");
    while(1);
  }

  // Pulls in all operation implementations
  static tflite::AllOpsResolver resolver;
  
  // Build an interpreter to run the model.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );
  interpreter = &static_interpreter; 

  // Allocate memory from the tensor arena for the models tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk){
    error_reporter->Report("AllocateTensors() failed.");
  }

  // Declare input/output pointer tensors for data processing.
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Start up BME280
  while (!bme.begin(0x77)){
    Serial.println("Failed to initalize BME280.");
    blink_led();
  }
  Serial.println("Initalized BME280.");
  Serial1.println("Initalized BME280.");
  
  while (!IMU.begin()){
    Serial.println("Failed to initalize IMU sensor.");
    blink_led();
  }
  Serial.println("Initalized IMU sensor.");
  Serial1.println("Initalized IMU sensor.");

  // CSV Columns
  Serial.println("\ntime,pressure,temp,alt_m,x_acc,y_acc,z_acc,predicted_layer");
  Serial1.println("\ntime,pressure,temp,alt_m,x_acc,y_acc,z_acc,predicted_layer");
  timeStamp = 0;

  digitalWrite(PROBLEM_LED, HIGH);
}

int argmax(const float* arr, int size){
  /*
    This methods purpose is to return an argmax value [0-5] from an array
    of probailities. Every inference, a new array of probabilities are entered
    in the parameter "arr." This will then go through and find the index of the 
    highest number and return it's index.

    Parameters:
      - arr (const float*): model_output
      - size (int): Set by default (6)

    Returns:
      index (int): Index of highest probability in array.
  */

  // Declare variables for argmax function
  float max = arr[0];
  int index = 0;

  // Iterate through each item to find highest probality and store
  // it's index and value.
  for (int i = 1; i < size; i++){
    if(arr[i] > max){
      max = arr[i];
      index = i;
    }
  }
  return index;
}

float alt_m(float pressure, float temp){
  /*
    This method calculates the altitude (in meters above sea level)
    from measured air pressure and temperature using the barometric formula:

      Altitude (m) = ((Temperature (C) + 273.15) / 0.0065) * (1 - (Pressure (hPa) / 1013.25)^0.190284)

    Parameters:
      pressure (float): pressure in hPa
      temp (float): temperature in C

    Returns:
      Altitude in m above sea level
  */

  return ((temp + 273.15) / LAPSE_RATE) * (1 - pow((pressure / SEA_LEVEL_PRESSURE), EXPONENT));
}

void loop() {
  /*
    This methods purpose is to read the data from the BMP280 sensor
    and load these values into the TFLite model to classify the data
    into an atmospheric layer in real-time.
  */
  
  timeStamp = millis();
  float x, y, z;

  // Declare variables and read barometric sensor outputs
  float pressure = bme.readPressure() / 100.0;
  float temp = bme.readTemperature();
  float altitude = alt_m(pressure, temp); 
  IMU.readAcceleration(x,y,z);

  /*
    Reliable Data Collection
      - If the model was trained off of [pressure, temp] (array format)
        the model will need to interpret the data the same way inside
        of an array tensor.
      
    Set respective input index tensors to their respective data columns.
  */
  model_input->data.f[0] = pressure;
  model_input->data.f[1] = temp;

  // Invoke interpreter to run model
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk){
    error_reporter->Report("Invoke failed on input.");
  }

  // Declare prob (probability) with empty array for tensor output
  float prob[6] = {};

  // Iterate through each index of the output (length: 6, refer to build.py)
  // and store inside of prob array.

  for (int i = 0; i < 6; i++){
    prob[i] = model_output->data.f[i];
  }

  // Declare and integer variable "layer_class" and run argmax function.
  int layer_class = argmax(prob, 6);

  // Print out values and iterate loop().
  Serial.print(timeStamp); Serial.print(",");
  Serial.print(pressure); Serial.print(",");
  Serial.print(temp); Serial.print(",");
  Serial.print(altitude); Serial.print(",");
  Serial.print(x); Serial.print(",");
  Serial.print(y); Serial.print(",");
  Serial.print(z); Serial.print(",");
  Serial.print(layer_class); Serial.println();

  // Store data into OpenLog sensor from UART connection.
  Serial1.print(timeStamp); Serial1.print(",");
  Serial1.print(pressure); Serial1.print(",");
  Serial1.print(temp); Serial1.print(",");
  Serial1.print(altitude); Serial1.print(",");
  Serial1.print(x); Serial1.print(",");
  Serial1.print(y); Serial1.print(",");
  Serial1.print(z); Serial1.print(",");
  Serial1.print(layer_class); Serial1.println();

  delay(500);
}