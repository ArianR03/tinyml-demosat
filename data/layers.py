import pandas as pd
import glob

def layer(alt) -> int:
    """
    Data Insertion
    
    This method is used to add output layers to the existing that inside of
    data.csv. This data will be used to train the model off of pressure and temperature
    and trained off the altitude layer classification.

    Args:
        alt (int, in data csv): height passed in to label respective atmospheric classification

    Returns:
        Atmospheric Layer Classification
        [0]: Surface Level
        [1]: Lower Troposphere
        [2]: Upper Troposphere
        [3]: Tropopause
        [4]: Lower Stratosphere
        [5]: Stratosphere
    """
    if alt <= 2000:
        return 0
    elif alt <= 8000:
        return 1
    elif alt <= 12000:
        return 2
    elif alt <= 15000:
        return 3
    elif alt <= 25000:
        return 4
    else:
        return 5

def assign_layer() -> pd.DataFrame:
    """
    This method is to read the data csv that will be passed into
    the layer function to label the respective row data to a specific
    atmospheric layer classification.

    Args:
        None
    
    Returns:
        New Dataframe with atmospheric layer class
    """
    
    csv_files = glob.glob("data/historical/*.csv")

    for file in csv_files:
        # Read csv file with historical data | Reference: (University of Wyoming Atmospheric Science Radiosonde Archive)
        df = pd.read_csv(file)

        # Create a new column 'layer' and assign layer from respective height 
        df['layer'] = df['geopotential height_m'].apply(layer)

        # Write into csv and save
        df.to_csv(file, index=False)

    return df

if __name__ == "__main__":
    assign_layer()