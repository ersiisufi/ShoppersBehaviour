# config.py
PATH = '/Users/admin/Documents/ML projects/CustomerChurn/Data/raw/online_shoppers_intention.csv'
# Feature lists
NUM_FEATURES = [
    'Administrative', 'Administrative_Duration', 'Informational', 
    'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration', 
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 
    'Admin_Ratio', 'Info_Ratio', 'Product_Efficiency'
]

CAT_FEATURES = [
    'Month', 'VisitorType', 'OperatingSystems', 
    'Browser', 'Region', 'TrafficType'
]

BIN_FEATURES = ['Weekend', 'Is_High_Urgency']

TARGET = 'Revenue'

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42