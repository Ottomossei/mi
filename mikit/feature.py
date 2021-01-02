import numpy as np
import pandas as pd
from .compname import ChemFormula
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm
import re


class CreateFeature:
    def __init__(self, path="./data/atom.csv", excluded_col=["Element"]):
        df_atom_features = pd.read_csv(path).drop(excluded_col, axis=1)
        df_atom_features = ((df_atom_features - df_atom_features.min()) / (df_atom_features.max() - df_atom_features.min()))
        self.atom_feature_values = df_atom_features.values.T
        self.atom_feature_colnames = df_atom_features.columns.values
    
    @staticmethod
    def reduct_matrix(molratio, atomfeature):
        idx = np.argwhere(molratio != 0.0)
        return (molratio[idx] * atomfeature[idx]).reshape([-1])

    def get_ave_features(self, dict_molratio):
        dict_output = {}
        for label, molratioes in dict_molratio.items():
            matrix = np.zeros((molratioes.shape[0], self.atom_feature_colnames.shape[0]))
            for r, molratio in enumerate(molratioes):
                for f, atomfeature in enumerate(self.atom_feature_values):
                    matrix[r,f] = sum(molratio * atomfeature)
            dict_output[label+"(Ave)"] = matrix
        return dict_output

    def get_max_features(self, dict_molratio, exc=[""]):
        dict_output = {}
        for label, molratioes in dict_molratio.items():
            matrix = np.zeros((molratioes.shape[0], self.atom_feature_colnames.shape[0]))
            for r, molratio in enumerate(molratioes):
                for f, atomfeature in enumerate(self.atom_feature_values):
                    matrix[r,f] = max(self.reduct_matrix(molratio, atomfeature))
            if label not in exc:
                dict_output[label+"(Max)"] = matrix
        return dict_output

    def get_min_features(self, dict_molratio, exc=[""]):
        dict_output = {}
        for label, molratioes in dict_molratio.items():
            matrix = np.zeros((molratioes.shape[0], self.atom_feature_colnames.shape[0]))
            for r, molratio in enumerate(molratioes):
                for f, atomfeature in enumerate(self.atom_feature_values):
                    matrix[r,f] = min(self.reduct_matrix(molratio, atomfeature))
            if label not in exc:
                dict_output[label+"(Min)"] = matrix
        return dict_output

    def get_std_features(self, dict_molratio, exc=[""]):
        dict_output = {}
        for label, molratioes in dict_molratio.items():
            matrix = np.zeros((molratioes.shape[0], self.atom_feature_colnames.shape[0]))
            for r, molratio in enumerate(molratioes):
                for f, atomfeature in enumerate(self.atom_feature_values):
                    matrix[r,f] = np.std(self.reduct_matrix(molratio, atomfeature))
            if label not in exc:
                dict_output[label+"(Std)"] = matrix
        return dict_output
    
    def get_df_learning(self, dict_feature, comp_names):
        output = pd.DataFrame(index=comp_names, columns=[])
        for label, features in dict_feature.items():
            labels = [label + "@" + colname for colname in self.atom_feature_colnames]
            output[labels] = features
        output = output.replace([np.inf, -np.inf], np.nan)
        return output.dropna(axis = 1, how = 'any')

class FilterMethod:
    def __init__(self, df):
        self.df = df
    
    @staticmethod
    def _feature_columns(df):
        feature_columns = df.columns.values
        p = "\@(.*)"
        main_feature_columns = list({"@" + re.findall(p, f)[0] for f in feature_columns})
        return feature_columns, main_feature_columns
    
    @staticmethod
    def _corr(feature_column, df, tol):
        feature_columns = df.columns.values
        target_columns = [f for f in feature_columns if re.findall(feature_column+'$', f)]
        df_corr = df.loc[:, target_columns].corr().abs()
        s_del, s_app = set(), set()
        for target_column in target_columns:
            overtol = list(df_corr[df_corr[target_column] > tol].index)
            overtol.remove(target_column)
            if (target_column not in s_del) and (target_column not in s_app):
                s_app.add(target_column)
            for o in overtol:
                if (o not in s_app) and (o not in s_del):
                    s_del.add(o)
        new_feature_columns = list(s_app)
        return df[new_feature_columns]
    
    def get_each_filter(self, tol):
        feature_columns, main_feature_columns = self._feature_columns(self.df)
        df_output = self._corr(main_feature_columns[0], self.df, tol)
        for feature_column in main_feature_columns[1:]:
            df_add = self._corr(feature_column, self.df, tol)
            df_output = pd.concat([df_output, df_add], axis=1)
        return df_output
