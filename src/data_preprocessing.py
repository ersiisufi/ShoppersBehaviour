import pandas as pd


def load_data(file_path):
    """
    Load data from a CSV file.

    Parameters:
    file_path (str): The path to the CSV file.

    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    

# def build_preprocessor():
#     preprocessor = ColumnTransformer(
#         transformers = [
#             ('num', StandardScaler(), config.NUM_FEATURES),
#             ('cat',OneHotEncoder(handle_unknown = 'ignore'), config.CAT_FEATURES),
#             ('bin', 'passthrough', config.BIN_FEATURES)
#         ]
#     )
#     return preprocessor 
        
