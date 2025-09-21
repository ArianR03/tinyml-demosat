#include <TensorFlowLite.h> 

// #include "tensorflow/lite/micro/kernels/micro_ops.h"
// #include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
// #include "tensorflow/lite/micro/micro_interpreter.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "layer_model.h"

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
  // put your setup code here, to run once:
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
}

void loop() {
  // put your main code here, to run repeatedly:

}
