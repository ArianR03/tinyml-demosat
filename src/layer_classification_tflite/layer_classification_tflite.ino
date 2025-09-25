/*

*/

#include <TensorFlowLite.h> 

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <Adafruit_Sensor.h>
#include <Adafruit_BME280.h>
#include "layer_model.h"

Adafruit_BME280 bme;

#define DEBUG 0

namespace{
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize]; 
}

void setup() {
  
  Serial.begin(9600);

#if DEBUG
  while (!Serial);
#endif

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(layer_model);
  
  if (model->version() != TFLITE_SCHEMA_VERSION){
    error_reporter->Report("Model version does not match schema.");
    while(1);
  }

  static tflite::AllOpsResolver resolver;
  
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize
  );

  interpreter = &static_interpreter; 

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk){
    error_reporter->Report("AllocateTensors() failed.");
  }

  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  while(!bme.begin(0x77)){
    Serial.println("BME280 not found. Retrying...");
  }
  Serial.println("BME280 connected at 0x77");

}

int argmax(const float* arr, int size){

  int max = arr[0];
  int index;

  // [0.12, 0.45, 0.55, 0.67, 0.85, 0.76]

  for (int i = 1; i < size; i++){
    if(arr[i] > max){
      max = arr[i];
      index = i;
    }
  }
  return index;

}
void loop() {
  // put your main code here, to run repeatedly:
  float pressure = bme.readPressure() / 100.0F;
  float temp = bme.readTemperature();

  model_input->data.f[0] = pressure;
  model_input->data.f[1] = temp;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk){
    error_reporter->Report("Invoke failed on input.");
  }

  float prob[6] = {};

  for (int i = 0; i < 6; i++){
    prob[i] = model_output->data.f[i];
  }

  int layer_class = argmax(prob, 6);
  Serial.print(layer_class);
}

