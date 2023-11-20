import os
import sys
import math

PROJ_DIR = os.getcwd()
PAR_DIR = os.path.dirname(__file__)

sys.path.insert(0, PROJ_DIR)

from utils.files import DataFile, to_tensor

if __name__ == '__main__':
    train_file = DataFile(os.path.join(PAR_DIR, 'data/train.csv'))
        
    # TODO: Check if performing one-hot encoding on Pclass improved the result
            
    # Remove redundant columns
    train_file.remove_column('Name')
    train_file.remove_column('Ticket')
    train_file.remove_column('PassengerId')
    train_file.apply_hot_encoding('Sex', drop_first=True)
    train_file.apply_hot_encoding('Embarked')
    # Because there are too many missing fields (NaN) in this column so it will be removed
    train_file.remove_column('Cabin')
        
    # Replace missing values (NaN) with average of the column
    avg_age = train_file.get_column_avg('Age')
    pax_ages = train_file['Age']
    for i in range(0, len(pax_ages)):
        if(math.isnan(pax_ages[i])): pax_ages[i] = avg_age
    train_file['Age'] = pax_ages
    
    X, y = train_file.get_data_and_groundtruth('Survived', use_tensor=True)
    print(X.shape)
    print(y.shape)
    