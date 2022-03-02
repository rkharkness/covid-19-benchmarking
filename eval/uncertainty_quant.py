import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score

def mode_robustness(x_test, y_test, model, std_coef, number_prediction=200):
     result = []
     for std in std_coef:
        print("std: {}".format(std))
        x_class = x_test
        mean = 0
        sigma = np.std(x_class) * std
        gaussian = np.random.normal(mean, sigma, x_class.shape)
        x_class = x_class + gaussian

        mc_predictions = []

        for _ in range(number_prediction):
            y_p_class = model.predict(x_class)
            mc_predictions.append(y_p_class)

        mc_ensemble_pred = np.array(mc_predictions).mean(axis=0).argmax(axis=1)
        ensemble_acc = accuracy_score(y_test.argmax(axis=1), mc_ensemble_pred)
        ensemble_precision = precision_score(y_test.argmax(axis=1), mc_ensemble_pred, average='weighted')
        ensemble_recall = recall_score(y_test.argmax(axis=1), mc_ensemble_pred, average='weighted')
        ensemble_F1 = (2 * ensemble_precision * ensemble_recall) / (ensemble_precision + ensemble_recall)

        result = [ensemble_acc * 100, ensemble_precision * 100, ensemble_recall * 100, ensemble_F1 * 100]
        print("[{:0.5f}, {:0.5f}, {:0.5f}, {:0.5f}]\n".format(*result))

def uncertainty_plot(model, image, label=None, save=True, name=None, mc_iter=200, ylim=20):
    """uncertainty fuse net"""
    image = image[np.newaxis, :, :, :]
    if label is not None:
        label_idx = np.argmax(label)

    class_text = ['Negative', 'Positive']

    model['model']

    all_preds = []
    for _ in range(mc_iter):
        preds = model['model'](image)
        all_preds.append(preds)

    preds_mean = np.mean(all_preds, axis=0)
    preds_mean = preds_mean[0]
    preds_std = np.std(all_preds, axis=0)
    preds_std = preds_std[0]
    class_idx_mean = np.argmax(preds_mean)

    preds = np.transpose(np.stack(all_preds), (1, 0, 2))

    image = (image + 1) / 2
    image1 = (image * 255).astype("uint8")
    image1 = np.squeeze(image1, axis=0)
    image1 = np.squeeze(image1, axis=2)

    fig = plt.figure(figsize=[8, 6.5])

    plt.subplot(2, 1, 1)
    plt.imshow(image1, cmap='gray')
    if label is not None:
        title = """
    True: {}
    Prediction: {}
    Value Order: [{}, {}, {}]
    Mean: [{:0.2f}, {:0.2f}, {:0.2f}]
    STD: [{:0.2f}, {:0.2f}, {:0.2f}]
    """.format(class_text[label_idx], class_text[class_idx_mean], *class_text,
               *preds_mean, *preds_std)
    else:
        title = """
    Prediction: {}
    Value Order: [{}, {}, {}]
    Mean: [{:0.2f}, {:0.2f}, {:0.2f}]
    STD: [{:0.2f}, {:0.2f}, {:0.2f}]
    """.format(class_text[class_idx_mean], *class_text,
               *preds_mean, *preds_std)

    plt.title(title, fontsize=12)

    plt.axis('off')

    for i in range(3):
        plt.subplot(2, 1, 2)
        plt.hist(preds[0][:, i], bins=50, alpha=0.3, label=class_text[i])

        plt.axvline(np.quantile(preds[0][:, i], 0.5), color='red', linestyle='--', alpha=0.4)
        plt.axvline(0.5, color='green', linestyle='--')

        plt.xlabel('Uncertainty', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.ylim([0, ylim])
        plt.legend()
    plt.tight_layout()
    if save:
        fig.savefig('{}.pdf'.format(name), dpi=300, bbox_inches='tight')
    plt.show()