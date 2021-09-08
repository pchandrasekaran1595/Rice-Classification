import os
import re
import sys
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import utils as u

SC_X = StandardScaler()

#####################################################################################################

class Model(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
    
        if re.match(r"lgr", self.model_name, re.IGNORECASE):
            self.model = LogisticRegression(random_state=u.SEED)
        elif re.match(r"gnb", self.model_name, re.IGNORECASE):
            self.model = GaussianNB()
        elif re.match(r"knc", self.model_name, re.IGNORECASE):
            self.model = KNeighborsClassifier()
        elif re.match(r"dtc", self.model_name, re.IGNORECASE):
            self.model = DecisionTreeClassifier(random_state=u.SEED)
        elif re.match(r"rfc", self.model_name, re.IGNORECASE):
            self.model = RandomForestClassifier(random_state=u.SEED)
        elif re.match(r"xgc", self.model_name, re.IGNORECASE):
            self.model = XGBClassifier(random_state=u.SEED)
        else:
            raise ValueError("Incorrect model value. Supported values are :\n1. lgr\n2. gnb\n3. knc\n4. dtc\n5. rfc\n6. xgc")
        
    def fit(self, X=None, y=None):
        self.model.fit(X, y)
    
    def predict(self, X=None):
        return self.model.predict(X)
    
    def save(self):
        filename = "{}_Model.pkl".format(self.model_name)
        pickle.dump(self.model, open(os.path.join(u.MODEL_PATH, filename), "wb"))

#####################################################################################################

def ml_analysis(features=None, targets=None):
    args_1 = "--model-name"
    args_2 = "--test"

    model_name = None
    train_mode = True
    names = {
        "lgr" : "Logistic Regression Model",
        "gnb" : "Naive Bayes Model",
        "knc" : "K-Neighbours Classifier Model",
        "dtc" : "Decision Tree Classifier Model",
        "rfc" : "Random Forest Classifier Model",
        "xgc" : "XGB Classifier Model"
    }

    if args_1 in sys.argv: model_name = sys.argv[sys.argv.index(args_1) + 1]
    if args_2 in sys.argv: train_mode = False

    assert(isinstance(model_name, str))
    
    if train_mode:
        tr_feats, va_feats, tr_trgts, va_trgts = train_test_split(features, targets, test_size=0.25,
                                                                random_state=u.SEED, shuffle=True)

        tr_feats = SC_X.fit_transform(tr_feats)
        va_feats = SC_X.transform(va_feats)

        model = Model(model_name=model_name)
        model.fit(tr_feats, tr_trgts)

        y_pred = model.predict(va_feats)

        accuracy = accuracy_score(y_pred, va_trgts)
        precision, recall, f_score, _ = precision_recall_fscore_support(y_pred, va_trgts)

        u.breaker()
        u.myprint(names[model_name] + "\n", "green")
        u.myprint("Accuracy  : {:.5f}".format(accuracy), "green")
        u.myprint("Precision : {:.5f}, {:.5f}".format(precision[0], precision[1]), "green")
        u.myprint("Recall    : {:.5f}, {:.5f}".format(recall[0], recall[1]), "green")
        u.myprint("F1 Score  : {:.5f}, {:.5f}".format(f_score[0], f_score[1]), "green")
        u.breaker()

        model.save()
    else:
        raise NotImplementedError("Test Mode is not Implemented")

#####################################################################################################
