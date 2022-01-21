from get_data import *
from model_pytorch import *
import os


def main():
    pytorch = False
    os.chdir(os.path.dirname(os.path.abspath("SB_main.py")))
    if pytorch:
        practice()
    else:
        get_individual_data_files("data/")

if __name__ == '__main__':
    main()
