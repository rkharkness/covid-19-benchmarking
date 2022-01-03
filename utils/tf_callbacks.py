from keras import backend as K
import keras as keras
import matplotlib.pyplot as plt


# metrics (training and val)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# plot loss and accuracy (training and val)
class PlotLosses(keras.callbacks.Callback):

    def __init__(self, filename, k_fold):
        super().__init__()
        self.filename = filename
        self.k_fold = k_fold

    def on_train_begin(self, logs={}):
        self.i = 0
        self.j = 0
        self.x = []
        self.y = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_accuracy'))
        self.i += 1
        self.j += 1
        self.y.append(self.j)

        plt.plot(self.x, self.losses, label="Training loss")
        plt.plot(self.x, self.val_losses, label="Validation loss")
        plt.legend(['Training Loss','Validation Loss'])
        plt.grid()
        plt.savefig(self.filename + 'loss.png')
        plt.close()
        plt.plot(self.y, self.acc, label="Training accuracy")
        plt.plot(self.y, self.val_acc, label="Validation accuracy")
        plt.legend(['Training Accuracy','Validation Accuracy'])
        plt.grid()
        plt.savefig(self.filename + f'accuracy-{self.k_fold}.png')
        plt.close()
