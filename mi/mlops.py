from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tqdm import tqdm


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
        try:
            best_params_idx = best_params_idx + best_param_idx
            cv_score = best_score
            clf.fit(self.X_train[:, list(np.array(best_params_idx))], self.y_train)
            y_train_predict = clf.predict(self.X_train[:, list(np.array(best_params_idx))])
            y_test_predict = clf.predict(self.X_test[:, list(np.array(best_params_idx))])
            train_score = r2_score(self.y_train, y_train_predict)
            test_score = r2_score(self.y_test, y_test_predict)
        except:
            pass
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



