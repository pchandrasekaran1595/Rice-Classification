import os
import re
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from time import time
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.nn.utils import weight_norm as WN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import utils as u

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SC_X = StandardScaler()

#####################################################################################################

class DS(Dataset):
    def __init__(self, X=None, y=None, mode="train"):
        self.mode = mode

        assert(re.match(r"train", self.mode, re.IGNORECASE) or re.match(r"valid", self.mode, re.IGNORECASE) or re.match(r"test", self.mode, re.IGNORECASE))

        self.X = X
        if re.match(r"train", self.mode, re.IGNORECASE) or re.match(r"valid", self.mode, re.IGNORECASE):
            self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if re.match(r"train", self.mode, re.IGNORECASE) or re.match(r"valid", self.mode, re.IGNORECASE):
            return torch.FloatTensor(self.X[idx]), torch.FloatTensor(self.y[idx])
        else:
            return torch.FloatTensor(self.X[idx])

#####################################################################################################

class Model(nn.Module):
    def __init__(self, IL=None, HL=None, DP=None):
        super(Model, self).__init__()

        self.model = nn.Sequential()

        if len(HL) == 1:
            self.model.add_module("BN1", nn.BatchNorm1d(num_features=IL, eps=1e-5))
            self.model.add_module("FC1", WN(nn.Linear(in_features=IL, out_features=HL[0])))
            if isinstance(DP, float):
                self.model.add_module("DP1", nn.Dropout(p=DP))
            self.model.add_module("AN1", nn.ReLU())
            self.model.add_module("BN2", nn.BatchNorm1d(num_features=HL[0], eps=1e-5))
            self.model.add_module("FC2", WN(nn.Linear(in_features=HL[0], out_features=1)))
        
        elif len(HL) == 2:
            self.model.add_module("BN1", nn.BatchNorm1d(num_features=IL, eps=1e-5))
            self.model.add_module("FC1", WN(nn.Linear(in_features=IL, out_features=HL[0])))
            if isinstance(DP, float):
                self.model.add_module("DP1", nn.Dropout(p=DP))
            self.model.add_module("AN1", nn.ReLU())
            self.model.add_module("BN2", nn.BatchNorm1d(num_features=HL[0], eps=1e-5))
            self.model.add_module("FC2", WN(nn.Linear(in_features=HL[0], out_features=HL[1])))
            if isinstance(DP, float):
                self.model.add_module("DP2", nn.Dropout(p=DP))
            self.model.add_module("AN2", nn.ReLU())
            self.model.add_module("BN3", nn.BatchNorm1d(num_features=HL[1], eps=1e-5))
            self.model.add_module("FC3", WN(nn.Linear(in_features=HL[1], out_features=1)))
        
        elif len(HL) == 3:
            self.model.add_module("BN1", nn.BatchNorm1d(num_features=IL, eps=1e-5))
            self.model.add_module("FC1", WN(nn.Linear(in_features=IL, out_features=HL[0])))
            if isinstance(DP, float):
                self.model.add_module("DP1", nn.Dropout(p=DP))
            self.model.add_module("AN1", nn.ReLU())
            self.model.add_module("BN2", nn.BatchNorm1d(num_features=HL[0], eps=1e-5))
            self.model.add_module("FC2", WN(nn.Linear(in_features=HL[0], out_features=HL[1])))
            if isinstance(DP, float):
                self.model.add_module("DP2", nn.Dropout(p=DP))
            self.model.add_module("AN2", nn.ReLU())
            self.model.add_module("BN3", nn.BatchNorm1d(num_features=HL[1], eps=1e-5))
            self.model.add_module("FC3", WN(nn.Linear(in_features=HL[1], out_features=HL[2])))
            if isinstance(DP, float):
                self.model.add_module("DP3", nn.Dropout(p=DP))
            self.model.add_module("AN3", nn.ReLU())
            self.model.add_module("BN4", nn.BatchNorm1d(num_features=HL[2], eps=1e-5))
            self.model.add_module("FC4", nn.Linear(in_features=HL[2], out_features=1))
        
        else:
            raise ValueError("Incorrect Value supplied to 'HL' Argument")
    
    def get_optimizer(self, lr=1e-3, wd=0.0):
        return optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def get_plateau_scheduler(self, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=patience, eps=eps)
    
    def forward(self, x):
        return self.model(x)

#####################################################################################################

def fit(model=None, optimizer=None, scheduler=None, epochs=None, 
        early_stopping_patience=None,
        dataloaders=None, verbose=False):
    def getAccuracy(y_pred, y_true):
        y_pred, y_true = torch.sigmoid(y_pred).detach(), y_true.detach()

        y_pred[y_pred > 0.5]  = 1
        y_pred[y_pred <= 0.5] = 0

        return torch.count_nonzero(y_true == y_pred).item() / len(y_true)
    
    u.breaker()
    u.myprint("Training ...", "cyan")
    u.breaker()
    
    bestLoss = {"train" : np.inf, "valid" : np.inf}
    bestAccs = {"train" : 0.0, "valid" : 0.0}
    Losses, Accuracies = [], []

    model.to(DEVICE)
    start_time = time()
    for e in range(epochs):
        e_st = time()

        epochLoss = {"train" : 0.0, "valid" : 0.0}
        epochAccs = {"train" : 0.0, "valid" : 0.0}

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            lossPerPass, accsPerPass = [], []

            for X, y in dataloaders[phase]:
                X, y = X.to(DEVICE), y.to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(X)
                    loss = torch.nn.BCEWithLogitsLoss()(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                lossPerPass.append(loss.item())
                accsPerPass.append(getAccuracy(output, y))
            epochLoss[phase] = np.mean(np.array(lossPerPass))
            epochAccs[phase] = np.mean(np.array(accsPerPass))
        Losses.append(epochLoss)
        Accuracies.append(epochAccs)

        if early_stopping_patience:
            if epochLoss["valid"] < bestLoss["valid"]:
                bestLoss = epochLoss
                BLE = e + 1
                torch.save({"model_state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict()},
                           os.path.join(u.MODEL_PATH, "state.pt"))
                early_stopping_step = 0
            else:
                early_stopping_step += 1
                if early_stopping_step > early_stopping_patience:
                    print("\nEarly Stopping at Epoch {}".format(e))
                    break
        
        if epochLoss["valid"] < bestLoss["valid"]:
            bestLoss = epochLoss
            BLE = e + 1
            torch.save({"model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict()},
                        os.path.join(u.MODEL_PATH, "state.pt"))
        
        if epochAccs["valid"] > bestAccs["valid"]:
            bestAccs = epochAccs
            BAE = e + 1
        
        if scheduler:
            scheduler.step(epochLoss["valid"])
        
        if verbose:
            u.myprint("Epoch: {} | Train Loss: {:.5f} | Valid Loss: {:.5f} |\
Train Accs: {:.5f} | Valid Accs: {:.5f} | Time: {:.2f} seconds".format(e+1, epochLoss["train"], epochLoss["valid"],
                                                                       epochAccs["train"], epochAccs["valid"],
                                                                       time()-e_st), "cyan")
        
    u.breaker()
    u.myprint("Best Validation Loss at Epoch {}".format(BLE), "cyan")
    u.breaker()
    u.myprint("Best Validation Accs at Epoch {}".format(BAE), "cyan")
    u.breaker()
    u.myprint("Time Taken [{} Epochs] : {:.2f} minutes".format(len(Losses), (time()-start_time)/60), "cyan")
    u.breaker()
    u.myprint("Training Complete", "cyan")
    u.breaker()

    return Losses, Accuracies, BLE, BAE

#####################################################################################################

def predict_batch(model=None, dataloader=None, mode="test"):
    model.load_state_dict(torch.load(os.path.join(u.MODEL_PATH, "./state.pt"))["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    y_pred = torch.zeros(1, 1).to(DEVICE)
    if re.match(r"valid", mode, re.IGNORECASE):
        for X, _ in dataloader:
            with torch.no_grad():
                output = torch.sigmoid(model(X))
            y_pred = torch.cat((y_pred, output.view(-1, 1)), dim=0)
    elif re.match(r"test", mode, re.IGNORECASE):
        for X in dataloader:
            with torch.no_grad():
                output = torch.sigmoid(model(X))
            y_pred = torch.cat((y_pred, output.view(-1, 1)), dim=0)
    
    y_pred = y_pred[1:].detach().cpu().numpy()

    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0

    return y_pred

#####################################################################################################

def predict(model=None, data=None):
    raise NotImplementedError("Single sample/observation prediction is unavailable")

#####################################################################################################

def save_graphs(L: list, A: list) -> None:
    TL, VL, TA, VA = [], [], [], []
    for i in range(len(L)):
        TL.append(L[i]["train"])
        VL.append(L[i]["valid"])
        TA.append(A[i]["train"])
        VA.append(A[i]["valid"])
    x_Axis = np.arange(1, len(TL) + 1)
    plt.figure("Plots")
    plt.subplot(1, 2, 1)
    plt.plot(x_Axis, TL, "r", label="Train")
    plt.plot(x_Axis, VL, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Loss Graphs")
    plt.subplot(1, 2, 2)
    plt.plot(x_Axis, TA, "r", label="Train")
    plt.plot(x_Axis, VA, "b", label="Valid")
    plt.legend()
    plt.grid()
    plt.title("Accuracy Graphs")
    plt.savefig("./Graphs.jpg")
    plt.close("Plots")

#####################################################################################################

def dl_analysis(features=None, targets=None):
    args_1 = "--bs"
    args_2 = "--lr"
    args_3 = "--wd"
    args_4 = "--dp"
    args_5 = "--scheduler"
    args_6 = "--epochs"
    args_7 = "--early"
    args_8 = "--hl"
    args_9 = "--test"

    batch_size = 256
    lr, wd = 1e-3, 0
    dp = None
    scheduler = None
    do_scheduler = None
    epochs = 10
    early_stopping = 5
    train_mode = True
    
    if args_1 in sys.argv: batch_size = int(sys.argv[sys.argv.index(args_1) + 1])
    if args_2 in sys.argv: lr = float(sys.argv[sys.argv.index(args_2) + 1])
    if args_3 in sys.argv: wd = float(sys.argv[sys.argv.index(args_3) + 1])
    if args_4 in sys.argv: dp = float(sys.argv[sys.argv.index(args_4) + 1])
    if args_5 in sys.argv:
        do_scheduler = True
        patience = int(sys.argv[sys.argv.index(args_5) + 1])
        eps = float(sys.argv[sys.argv.index(args_5) + 2])
    if args_6 in sys.argv: epochs = int(sys.argv[sys.argv.index(args_6) + 1])
    if args_7 in sys.argv: early_stopping = int(sys.argv[sys.argv.index(args_7) + 1])
    if args_8 in sys.argv:
        if sys.argv[sys.argv.index(args_8) + 1] == "1":
            HL = [int(sys.argv[sys.argv.index(args_8) + 2])]
        elif sys.argv[sys.argv.index(args_8) + 1] == "2":
            HL = [int(sys.argv[sys.argv.index(args_8) + 2]), 
                  int(sys.argv[sys.argv.index(args_8) + 3])]
        elif sys.argv[sys.argv.index(args_8) + 1] == "3":
            HL = [int(sys.argv[sys.argv.index(args_8) + 2]), 
                  int(sys.argv[sys.argv.index(args_8) + 3]),
                  int(sys.argv[sys.argv.index(args_8) + 4])]
    if args_9 in sys.argv: train_mode = False

    if train_mode:
        tr_feats, va_feats, tr_trgts, va_trgts = train_test_split(features, targets, test_size=0.25,
                                                                random_state=u.SEED, shuffle=True)

        tr_feats = SC_X.fit_transform(tr_feats)
        va_feats = SC_X.transform(va_feats)

        tr_data_setup = DS(X=tr_feats, y=tr_trgts.reshape(-1, 1), mode="train")
        va_data_setup = DS(X=va_feats, y=va_trgts.reshape(-1, 1), mode="valid")

        dataloaders = {
            "train" : DL(tr_data_setup, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(u.SEED)),
            "valid" : DL(va_data_setup, batch_size=batch_size, shuffle=False)
        }

        torch.manual_seed(u.SEED)
        model = Model(IL=tr_feats.shape[1], HL=HL, DP=dp)
        optimizer = model.get_optimizer(lr=lr, wd=wd)
        if do_scheduler:
            scheduler = model.get_plateau_scheduler(optimizer=optimizer, patience=patience, eps=eps, verbose=True)
        
        L, A, _, _ = fit(model=model, optimizer=optimizer, scheduler=scheduler, epochs=epochs,
                         early_stopping_patience=early_stopping,
                         dataloaders=dataloaders, verbose=True)
        save_graphs(L, A)

        y_pred = predict_batch(model=model, dataloader=dataloaders["valid"], mode="valid")

        accuracy = accuracy_score(y_pred, va_trgts)
        precision, recall, f_score, _ = precision_recall_fscore_support(y_pred, va_trgts)

        u.myprint("Accuracy  : {:.5f}".format(accuracy), "green")
        u.myprint("Precision : {:.5f}, {:.5f}".format(precision[0], precision[1]), "green")
        u.myprint("Recall    : {:.5f}, {:.5f}".format(recall[0], recall[1]), "green")
        u.myprint("F1 Score  : {:.5f}, {:.5f}".format(f_score[0], f_score[1]), "green")
        u.breaker()
    else:
        raise NotImplementedError("Test Mode is not Implemented")

#####################################################################################################
