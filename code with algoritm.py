""""
Group assignment group 4
"""
# pip install rdkit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rdkit
import sklearn
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, AllChem, ChemicalFeatures
from rdkit.Chem import PandasTools, Draw
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

class DataPrep:

    def __init__(self):
        self.testset = 'tested_molecules_v2.csv'
        self.trainingset = 'tested_molecules-1.csv'
        self.untestedset = 'untested_molecules.csv'

    def load_data(self, filename):
        return pd.read_csv(filename)
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
        return data


class DrugDiscoveryEDA:

    def __init__(self, data):
        self.data = data

    def explore_data(self):
        """
        Perform exploratory data analysis on the drug discovery data.
        """
        self.display_data_summary()
        self.plot_target_distribution()
        #self.visualize_molecular_structures()

    def display_data_summary(self):
        """
        Display a summary of the dataset.
        """
        print(self.data.head())  # Display the first few rows
        print(self.data.info())  # Summary of the dataset
        print(self.data.describe())  # Statistical summary
        print(self.data.shape)  # Number of rows and columns

    def plot_target_distribution(self):
        """
        Plot the distribution of the target variable.
        """
        target_counts = self.data["ALDH1_inhibition"].value_counts()
        plt.bar(target_counts.index, target_counts.values)
        plt.xlabel("ALDH1 Inhibition")
        plt.ylabel("Count")
        plt.show()

    def visualize_molecular_structures(self):
        """
        Visualize the molecular structures in the dataset.
        """
        PandasTools.AddMoleculeColumnToFrame(self.data, "SMILES", "Molecule")
        for i, mol in enumerate(self.data["Molecule"]):
            img = Draw.MolToImage(mol)
            plt.imshow(img)
            plt.axis("off")
            plt.show()

    def compute_descriptors(self):
        """
        Compute molecular descriptors for the molecules in the dataset.
        """
        self.data["MolecularWeight"] = self.data["Molecule"].map(Chem.Descriptors.MolWt)

    def analyze_descriptor_distribution(self, descriptor):
        """
        Analyze the distribution of a specific descriptor.

        Args:
            descriptor (str): Name of the descriptor column.
        """
        plt.hist(self.data[descriptor], bins=20)
        plt.xlabel(descriptor)
        plt.ylabel("Frequency")
        plt.show()

    def analyze_correlations(self):
        """
        Analyze the correlations between descriptors and the target variable.
        """
        self.corr_data = pd.DataFrame(self.data, columns=['ALDH1_inhibition','n_Atoms','MolecularWeight','LogP','TPSA','NumRotatableBonds','NumHDonors','NumHAcceptors','NumAromaticRings','NumSaturatedRings'])
        correlation_matrix = self.corr_data.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(self.corr_data.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.corr_data.columns)
        ax.set_yticklabels(self.corr_data.columns)
        plt.title("correlation matrix of dataset")
        plt.show()
        print(correlation_matrix)


    def select_features(self, selected_features):
        """
        Select a subset of features for further analysis.

        Args:
            selected_features (list): List of selected feature column names.
        """
        self.selected_features = selected_features

    def explore_feature_relationships(self):
        """
        Explore the relationships between selected features using scatter plots.
        """
        pd.plotting.scatter_matrix(self.data[self.selected_features], alpha=0.2)
        plt.show()

    def perform_dimensionality_reduction(self):
        """
        Perform dimensionality reduction using PCA and visualize the reduced features.
        """
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.data[self.selected_features])
        reduced_df = pd.DataFrame(reduced_features, columns=["PC1", "PC2"])
        reduced_df["ALDH1_inhibition"] = self.data["ALDH1_inhibition"]
        plt.scatter(reduced_df["PC1"], reduced_df["PC2"], c=reduced_df["ALDH1_inhibition"])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()


# Usage example:
data_prep = DataPrep()
test_data = data_prep.load_data(data_prep.testset)
c_test_data = data_prep.moldesc(test_data)

eda = DrugDiscoveryEDA(c_test_data)
eda.explore_data()
#eda.compute_descriptors()
#eda.analyze_descriptor_distribution("MolecularWeight")
eda.analyze_correlations()
eda.select_features(['ALDH1_inhibition','n_Atoms','MolecularWeight','LogP','TPSA','NumRotatableBonds','NumHDonors','NumHAcceptors','NumAromaticRings','NumSaturatedRings'])
eda.explore_feature_relationships()
eda.perform_dimensionality_reduction()
