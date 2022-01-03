from pandas.core.base import DataError
import optuna
from optuna.trial import TrialState

import os

import pandas as pd
from dataloaders import make_generators

import torch
from torch import optim

import keras
from residual_attn.res_attn import AttentionResNet56
from caps_net.covid_caps import capsule_net, margin_loss

from utils import f1_m, precision_m, recall_m


DEVICE = torch.device("cpu")

#GET THESE INTO A JSON FILE FOR IMPORT ACROSS FILES
PATH = "/MULTIX/DATA/HOME/covid-19-benchmarking/"
BATCHSIZE = 12
CLASSES = 2
DIR = os.getcwd()
EPOCHS = 10
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10

MODEL = AttentionResNet56
# add these obj attributes and embed in code for functionality
#MODEL_TYPE = AttentionResNet56.type
#LOSS = AttentionResNet56.loss
#OPTIMIZER = AttentionResNet56.loss

class HyperOpt(torch.nn.Module):
    def __init__(self, model, data, dataloader) -> None:
        super().__init__()
        self.model = model
        self.data = data # dict of train and val
        self.dataloader = dataloader # dict of train and val

    def checks(self, model_type):
        assert model_type in ['keras', 'pytorch', 'fastai']

    def model_builder(self, trial):

        m = self.model

        if m.model_type == 'keras':
            learning_rate =  trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            opt = m.get_optimizer()
            loss_fn = m.get_loss()
            m.compile(optimizer=opt(lr=learning_rate), loss=loss_fn, metrics=['accuracy', f1_m, precision_m, recall_m])
            return m
        else:
            return m

    def obj(self, trial, optimizer_name):

        # Generate the model
        model = self.model_builder(trial).to(DEVICE)

        x_train, y_train = self.data['train'][:N_TRAIN_EXAMPLES]
        x_val, y_val = self.data['val'][:N_VALID_EXAMPLES]

        if self.model_type == 'keras':
            model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                shuffle=True,
                batch_size=BATCHSIZE,
                epochs=EPOCHS,
                verbose=False,
            )

            # Evaluate the model accuracy on the validation set.
            score = model.evaluate(x_val, y_val, verbose=0)
            return score[1]
        
        else:
            # Generate the optimizers
            optimizer_name = model.get_optimizer()
            lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
            optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

            # Training of the model.
            for epoch in range(EPOCHS):
                model.train()
                for batch_idx, (data, target) in enumerate(self.dataloader['train']):
                    # Limiting training data for faster epochs.
                    if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                        break

                    data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)

                    optimizer.zero_grad()
                    output = model(data)
                    loss_fn = model.get_loss()
                    loss = loss_fn(output, target)
                    
                    loss.backward()
                    optimizer.step()

                # Validation of the model.
                model.eval()
                correct = 0
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(self.dataloader['val']):
                        # Limiting validation data.
                        if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                            break
                        data, target = data.view(data.size(0), -1).to(DEVICE), target.to(DEVICE)
                        output = model(data)
                        # Get the index of the max log-probability.
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()

                accuracy = correct / min(len(self.dataloader['val'].dataset), N_VALID_EXAMPLES)

                trial.report(accuracy, epoch)

                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return accuracy



if __name__ == "__main__":

    data = pd.read_csv(PATH)
    train_data = data[data['kfold_1'] == 'train']
    val_data = data[data['kfold_1'] == 'val']

    train_data = train_data[:N_TRAIN_EXAMPLES]
    val_data = val_data[:N_VALID_EXAMPLES]

    data_dict = {'train':train_data, 'val': val_data}

    make_generators()
   # dataloader_dict = 

    model = AttentionResNet56()
    
    h_optimization = HyperOpt(model, data_dict, dataloader_dict)

    study = optuna.create_study(direction="maximize") # maximize for accuracy

    study.optimize(h_optimization, n_trials=100, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(" Saving best params...")
    hp_df = study.trials_dataframe()
    hp_df.to_csv(PATH + str(MODEL))