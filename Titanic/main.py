import os
import sys

PROJ_DIR = os.getcwd()
PAR_DIR = os.path.dirname(__file__)

sys.path.insert(0, PROJ_DIR)

from utils.files import DataFile

if __name__ == '__main__':
    train_file = DataFile(os.path.join(PAR_DIR, 'data/train.csv'))   
        
    train_file.remove_column('Name')
    train_file.remove_column('Ticket')
    train_file.remove_column('Cabin')
    train_file.apply_hot_encoding('Sex', drop_first=True)
    train_file.apply_hot_encoding('Embarked')
    
    print(train_file.shape)
    print(train_file.to_string(truncate_H=False, truncate_V=False))
    