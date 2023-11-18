import os
import sys

PROJ_DIR = os.getcwd()
PAR_DIR = os.path.dirname(__file__)

sys.path.insert(0, PROJ_DIR)

from utils.files import DataFile

if __name__ == '__main__':
    train_file = DataFile(os.path.join(PAR_DIR, 'data/train.csv'))
