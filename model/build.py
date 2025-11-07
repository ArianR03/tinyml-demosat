import tensorflow as tf
from keras import layers, Sequential
import pandas as pd
import numpy as np
from pathlib import Path
import glob

class Model(object):

    @staticmethod
    def train_model(data: Path='data/historical/*.csv') -> object:
        """
        Layer Atmospheric Classification Model

        This method is used to define the parameters of what the model will be learning.
        This models intended purpose is to be trained on pressure and temperature to predict
        atmospheric layer classifiication. 

        TF Model
            Sequential
                - Constructed by adding layers which builds up a neural network. =
                - Can pass in single input tensors with outputting a single tensor

        Args
            data (Path): Direct path to historical data for training.
            activation (model): 'relu,' used for learning complex patterns in data.
            input_shape (size of input): Amount of features that will be given.

        Return
            model (object): Trained model
        """

        model = Sequential([
            layers.Dense(32, activation='relu', input_shape=(2,)),
            layers.Dense(6, activation='softmax')
        ])

        # Prepare model for training and evaluation by compiling
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Read file location and grab each csv file
        csv_files = glob.glob(data)

        # Read each csv from data folder
        all_dfs = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(all_dfs, ignore_index=True)

        # Pass the features that the model will be trained off of
        features = ['pressure_hPa', 'temperature_C']

        X = df[features]

        # What the model will be predicting
        y = df['layer']

        model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)

        return model
    
    @staticmethod
    def prediction_layer(input_data, model: object) -> int:
        """
        This method is primarily used to test the model itself. Passing in the data
        it was trained off of; pass it into the parameter as [pressure, temp] (allign with
        the way it was trained). This will run the model and predict an Atmospheric 
        Layer Classification between [0-5].

        Args:
            input_data: Test data -> float[pressure, temp] -> [600.6, 40.4]
            model (object): Model that was trained in train_model() method

        Returns: 
            Layer Classification (int):
                0.) Surface Level
                1.) Lower Troposphere
                2.) Upper Troposphere
                3.) Tropopause
                4.) Lower Stratosphere
                5.) Stratosphere
        """

        # Read data as a numpy array
        new_data = np.array(input_data)

        # Predict data that was tested (outputs probability stats)
        pred = model.predict(new_data)
        
        # Output as [0,5]
        predicted_layer = pred.argmax(axis=1)
        
        # Return the predicted layer classification
        return predicted_layer
    
    def convert_model_to_tflite(self, trained_model: object) -> bytes:
        """
        This method is used for converting a TensorFlow model into a TensorFlow Lite which 
        is used to deploy on Microcontrollers (Arduino Nano 33 BLE Sense Rev2). This TFLite 
        model becomes a much smaller and faster to run model to run.

        Args:
            None
        
        Returns:
            tflite_model (object): TensorFlow Model converted to TFLite.
        """

        # Declare model name for tflite file
        tflite_model_name = "layer_model"

        # Initialize variable to be used to convert model
        converter = tf.lite.TFLiteConverter.from_keras_model(trained_model)
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

        # Convert the model into TFLite
        tflite_model = converter.convert()

        # Open a new file and write the model into it.
        open(tflite_model_name + '.tflite', 'wb').write(tflite_model)

        return tflite_model
    
    def hex_to_c_arr(self, hex_data, var_name: str):
        """
        This method is used to pass in the tflite model into a C header
        array. This allowed the intrepreter within the Arduino to read the model
        and analyze the data correctly. 

        Args:
            hex_data (object): Trained model
            var_name (name of file): Defined in write_to_c file.
        
        Returns:
            C file: TFLite model converted to C Header array file for intrepretation in Arduino
        """
        # Initialize C string for file manipulation
        c_str = ''

        # Create C header guard
        c_str += '#ifndef ' + var_name.upper() + '_H\n'
        c_str += '#define ' + var_name.upper() + '_H\n\n'

        # Add array length 
        c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'
        
        c_str += 'unsigned char ' + var_name + '[] = {'
        hex_array = []

        for i, val in enumerate(hex_data):
            
            # Construct string from hex
            hex_str = format(val, '#04x')

            if (i + 1) < len(hex_data):
                hex_str += ','
            if (i+1) % 12 == 0:
                hex_str += '\n '
            hex_array.append(hex_str)

        # Add closing brace
        c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

        # Close out header guard
        c_str += '#endif //' + var_name.upper() + '_H'

        return c_str
    
    def write_to_c(self, trained_model: object):
        """
        This method is used to use the hex_to_c_arr() method to write into 
        the C file containing the array. This is used for the Arduino Nano to 
        intrepret the data and process the layer classification.

        Args:
            None

        Returns:
            None: Writes into C header array file.
        """
        c_model_name = "layer_model"

        with open(c_model_name + '.h', 'w') as file:
            file.write(self.hex_to_c_arr(self.convert_model_to_tflite(trained_model), c_model_name))

if __name__ ==  "__main__":

    # Call Model Object
    model = Model()

    # Train model
    trained_model = model.train_model()

    # Convert to TFLite model and write to C for Arduino
    model.convert_model_to_tflite(trained_model)
    model.write_to_c(trained_model)