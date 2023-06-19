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
from sklearn import preprocessing


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
    def data_scaling(self):

        standard_data = preprocessing.StandardScaler().fit(self.data[self.selected_features])
        self.scaled_standard_data = standard_data.transform(self.data[self.selected_features])
        plt.boxplot(self.scaled_standard_data)
        plt.show()
        return self.scaled_standard_data
    def perform_dimensionality_reduction(self):
        """
        Perform dimensionality reduction using PCA and visualize the reduced features.
        """
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(self.scaled_standard_data)
        reduced_df = pd.DataFrame(reduced_features, columns=["PC1", "PC2"])
        reduced_df["ALDH1_inhibition"] = self.data["ALDH1_inhibition"]
        plt.scatter(reduced_df["PC1"], reduced_df["PC2"], c=reduced_df["ALDH1_inhibition"])
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()
        return reduced_df

def algoritm_classifier(training_data,test_data):
    """
    Code to run random forest classifier
    """
    reduced_training=training_data.drop(columns="ALDH1_inhibition")
    tr_labels=training_data["ALDH1_inhibition"]
    
    if 'ALDH1_inhibition' in test_data.columns:
        reduced_test=test_data.drop(columns="ALDH1_inhibition")
        te_labels=test_data["ALDH1_inhibition"]
    
    rf=RandomForestClassifier(n_estimators=1000, random_state=42)
    fit=rf.fit(reduced_training,tr_labels)
    prediction=rf.predict(reduced_test)
    return prediction, te_labels

def algoritm_evaluation(prediction,te_labels):
    """
    Code to evaluate the algoritm used
    """
    
    prediction_df=pd.DataFrame(prediction)
    
    right_predictions=prediction_df[prediction_df[0]==te_labels]
    wrong_predictions=prediction_df[prediction_df[0]!=te_labels]
    
    t_pos=int(prediction_df[right_predictions==1].count())
    #print("True positives:"+str(t_pos))
    t_neg=int(prediction_df[right_predictions==0].count())
    #print("True negative:"+str(t_neg))
    f_pos=int(prediction_df[wrong_predictions==1].count())
    #print("False positives:"+str(f_pos))
    f_neg=int(prediction_df[wrong_predictions==0].count())
    #print("False negative:"+str(f_neg))
    
    Sn0=t_neg/te_labels.value_counts()[0]
    Pr0=t_neg/prediction_df[0].value_counts()[0]
    
    Sn1=t_pos/te_labels.value_counts()[1]
    Pr1=t_pos/prediction_df[0].value_counts()[1]

    BAcc=(Sn0+Sn1)/2
    sensitivity=t_pos/(t_pos+f_neg)
    specificity=t_neg/(t_neg+f_pos)
    precision=t_pos/(t_pos+f_pos)
    
    results=[BAcc, sensitivity, specificity, precision]
    return results
    
def data_conversion(c_training_data,c_test_data):
    training_eda=DrugDiscoveryEDA(c_training_data)
    test_eda=DrugDiscoveryEDA(c_test_data)
    
    training_eda.select_features(['ALDH1_inhibition','n_Atoms','MolecularWeight','LogP','TPSA','NumRotatableBonds','NumHDonors','NumHAcceptors','NumAromaticRings','NumSaturatedRings'])
    test_eda.select_features(['ALDH1_inhibition','n_Atoms','MolecularWeight','LogP','TPSA','NumRotatableBonds','NumHDonors','NumHAcceptors','NumAromaticRings','NumSaturatedRings'])

    training_eda.data_scaling()
    training=training_eda.perform_dimensionality_reduction()
    test_eda.data_scaling()
    test=test_eda.perform_dimensionality_reduction()
    return training, test
  
# Usage example:
data_prep = DataPrep()
raw_test_data = data_prep.load_data(data_prep.testset)
c_test_data = data_prep.moldesc(raw_test_data)

raw_training_data = data_prep.load_data(data_prep.trainingset)
c_training_data = data_prep.moldesc(raw_training_data)

raw_untested_data = data_prep.load_data(data_prep.untestedset)
c_untested_data = data_prep.moldesc(raw_untested_data)

eda = DrugDiscoveryEDA(c_test_data)
eda.explore_data()
#eda.compute_descriptors()
#eda.analyze_descriptor_distribution("MolecularWeight")
eda.analyze_correlations()
eda.select_features(['ALDH1_inhibition','n_Atoms','MolecularWeight','LogP','TPSA','NumRotatableBonds','NumHDonors','NumHAcceptors','NumAromaticRings','NumSaturatedRings'])
eda.explore_feature_relationships()
eda.data_scaling()
eda.perform_dimensionality_reduction()



training_data,test_data=data_conversion(c_training_data,c_test_data)
prediction,te_labels=algoritm_classifier(training_data, test_data)
x=algoritm_evaluation(prediction, te_labels)
print(x)
