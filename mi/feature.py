import numpy as np
import pandas as pd
from compname import ChemFormula
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm

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

class WrapperMethod():
    def __init__(self, X, y, feature_names, train_rate=0.9):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=0, train_size=train_rate)
        self.feature_names = feature_names
    
    def forward_search(self, clf, best_score, best_params_idx):
        for f in tqdm(range(len(self.feature_names))):
            if f not in best_params_idx:
                clf.fit(self.X_train[:, list(np.array([f]+best_params_idx))], self.y_train)
                if clf.best_score_ > best_score:
                    best_score = clf.best_score_
                    best_param_idx = [f]
                    # print(self.feature_names[f])
                    # print(self.feature_names[f])
                    # print("  |  " + self.feature_names[f] + "  |  " + str("{:.3f}".format(best_score)))
        try:
            best_params_idx = best_params_idx + best_param_idx
            print()
            print("All Feature : " + str(self.feature_names[best_params_idx]))
            print("Add Feature : " + str(self.feature_names[best_param_idx]))
            print("Best Score : " + str("{:.3f}".format(best_score)))
            cv_score = best_score
            clf.fit(self.X_train[:, list(np.array(best_params_idx))], self.y_train)
            y_train_predict = clf.predict(self.X_train[:, list(np.array(best_params_idx))])
            y_test_predict = clf.predict(self.X_test[:, list(np.array(best_params_idx))])
            train_score = r2_score(self.y_train, y_train_predict)
            test_score = r2_score(self.y_test, y_test_predict)
        except:
            print("End")
        return best_params_idx, cv_score, train_score, test_score
    
    def calc_forward(self, model, hyper_params):
        cv_score, new_cv_score = 0.01, 0
        best_params_idx, cv_scores, train_scores, test_scores = list(), list(), list(), list()
        clf = GridSearchCV(model, hyper_params)
        while cv_score >= new_cv_score:
            new_cv_score = cv_score
            try:
                best_params_idx, cv_score, train_score, test_score = self.forward_search(clf, cv_score, best_params_idx)
                cv_scores.append(cv_score)
                train_scores.append(train_score)
                test_scores.append(test_score)
            except:
                return best_params_idx, cv_scores, train_scores, test_scores
        


        