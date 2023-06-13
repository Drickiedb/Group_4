""""
Group assignment group 4
"""
# pip install rdkit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rdkit
import sklearn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
from rdkit import DataStructs
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification


class DataPrep:

    def __init__(self):
        self.testset = 'tested_molecules_v2.csv'
        self.trainingset = 'tested_molecules-1.csv'
        self.untestedset = 'untested_molecules.csv'
        self.test_data = pd.read_csv(self.testset)
        self.training_data = pd.read_csv(self.trainingset)
        self.untested_data = pd.read_csv(self.untestedset)
    def moldesc(self, data):
        PandasTools.AddMoleculeColumnToFrame(data, smilesCol='SMILES',includeFingerprints=True)
        #fpgen = AllChem.GetRDKitFPGenerator()
        data["n_Atoms"] = data['ROMol'].map(lambda x: x.GetNumAtoms())
        #data["pattern_fp"] = [fpgen.GetFingerprint(x) for x in data]
        #data["pair_fp"]
        #data["torsion_fp"]
        #data["morgan_fp"] =
        #data = data.drop(['ROMol'], axis=1)
        print(data.head())
DP = DataPrep()
DP.__init__()
DP.moldesc(data = DP.test_data)
