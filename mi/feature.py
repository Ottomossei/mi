import numpy as np
import pandas as pd
from compname import ChemFormula

class CreateFeature:
    def __init__(self, path="./data/atom.csv", excluded_col=["Element"]):
        df_atom_features = pd.read_csv(path).drop(excluded_col, axis=1)
        df_atom_features = ((df_atom_features - df_atom_features.min()) / (df_atom_features.max() - df_atom_features.min()))
        self.atom_feature_values = df_atom_features.values.T
        self.atom_feature_colnames = df_atom_features.columns.values
    
    def get_ave_features(self, dict_molratio):
        dict_output = {}
        for label, molratioes in dict_molratio.items():
            matrix = np.zeros((molratioes.shape[0], self.atom_feature_colnames.shape[0]))
            for r, molratio in enumerate(molratioes):
                for f, atomfeature in enumerate(self.atom_feature_values):
                    matrix[r,f] = sum(molratio * atomfeature)
            dict_output[label+"(Ave)"] = matrix
        return dict_output

    def get_max_features(self, dict_molratio, exc=None):
        dict_output = {}
        for label, molratioes in dict_molratio.items():
            matrix = np.zeros((molratioes.shape[0], self.atom_feature_colnames.shape[0]))
            for r, molratio in enumerate(molratioes):
                for f, atomfeature in enumerate(self.atom_feature_values):
                    matrix[r,f] = max(molratio * atomfeature)
            if label not in exc:
                dict_output[label+"(Max)"] = matrix
        return dict_output

    def get_min_features(self, dict_molratio, exc=None):
        dict_output = {}
        for label, molratioes in dict_molratio.items():
            matrix = np.zeros((molratioes.shape[0], self.atom_feature_colnames.shape[0]))
            for r, molratio in enumerate(molratioes):
                for f, atomfeature in enumerate(self.atom_feature_values):
                    matrix[r,f] = min(molratio * atomfeature)
            if label not in exc:
                dict_output[label+"(Min)"] = matrix
        return dict_output

    def get_std_features(self, dict_molratio, exc=None):
        dict_output = {}
        for label, molratioes in dict_molratio.items():
            matrix = np.zeros((molratioes.shape[0], self.atom_feature_colnames.shape[0]))
            for r, molratio in enumerate(molratioes):
                for f, atomfeature in enumerate(self.atom_feature_values):
                    matrix[r,f] = np.std(molratio * atomfeature)
            if label not in exc:
                dict_output[label+"(Std)"] = matrix
        return dict_output