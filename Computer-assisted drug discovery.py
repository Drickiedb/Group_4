""""
Group assignment group 4
"""
# pip install rdkit
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
smiles = 'COC(=O)c1c[nH]c2cc(OC(C)C)c(OC(C)C)cc2c1=O'
mol = Chem.MolFromSmiles(smiles)
print(mol)
smi = Chem.MolToSmiles(mol)
print(smi)

class DataPrep:

    def __init__(self):
        self.firstvariable = 0

    def first_function(self):
        self.firstvariable
