"""

Processes the mcv dataset into the mel spectrograms

"""

import csv
import pandas as pd

PATH = "./cv-corpus-23.0-2025-09-05/en/"

class DatasetProcessing():

    def __init__(self,
                 path) -> None:
        self.path = path


    def get_ages(self, 
                 file : str
                 ):
        
        if(file[-4:] != ".tsv"):
            file = file + ".tsv"

        number_of_age = {}

        with open(self.path + file, "r", encoding="utf-8") as f:
            tsv_reader = csv.reader(f, delimiter="\t", quotechar='"')
            tsv_reader.__next__()
            
            for row in tsv_reader:
                if(row[7] != ''):
                    if(not (row[7] in number_of_age.keys())):
                        number_of_age[row[7]] = 1
                    else:
                        number_of_age[row[7]] += 1

        print(number_of_age)


d = DatasetProcessing(PATH)

d.get_ages("train.tsv")