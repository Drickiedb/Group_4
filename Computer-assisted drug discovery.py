""""
Group assignment group 4
"""
# pip install rdkit
import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification


class DataPrep:

    def __init__(self):
        self.testset = 'tested_molecules_v2.csv'
        self.trainingset = 'tested_molecules-1.csv'
        self.untestedset = 'untested_molecules.csv'

    def openfiles(self):
        self.test_data = pd.read_csv(self.testset)
        self.training_data = pd.read_csv(self.trainingset)
        self.untested_data = pd.read_csv(self.untestedset)

