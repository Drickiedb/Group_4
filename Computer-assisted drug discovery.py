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
from rdkit.Chem import Descriptors, MACCSkeys, AllChem, ChemicalFeatures
from rdkit.Chem import PandasTools
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
        data["n_Atoms"] = data['ROMol'].map(lambda x: x.GetNumAtoms())
        data['MolecularWeight'] = data['SMILES'].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x)))
        data['LogP'] = data['SMILES'].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x)))
        data['TPSA'] = data['SMILES'].apply(lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x)))
        data['NumRotatableBonds'] = data['SMILES'].apply(lambda x: Descriptors.NumRotatableBonds(Chem.MolFromSmiles(x)))
        data['NumHDonors'] = data['SMILES'].apply(lambda x: Descriptors.NumHDonors(Chem.MolFromSmiles(x)))
        data['NumHAcceptors'] = data['SMILES'].apply(lambda x: Descriptors.NumHAcceptors(Chem.MolFromSmiles(x)))
        data['NumAromaticRings'] = data['SMILES'].apply(lambda x: Descriptors.NumAromaticRings(Chem.MolFromSmiles(x)))
        data['NumSaturatedRings'] = data['SMILES'].apply(lambda x: Descriptors.NumSaturatedRings(Chem.MolFromSmiles(x)))

        # Add molecular fingerprints
        data['MACCS_Fingerprint'] = data['SMILES'].apply(
            lambda x: MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x)).ToBitString())
        data['Morgan_Fingerprint'] = data['SMILES'].apply(
            lambda x: AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), 2).ToBitString())
        print(data.head())
DP = DataPrep()
DP.__init__()
DP.moldesc(data = DP.test_data)
