import logging
import torch
import numpy as np

logger = logging.Logger(__name__)


def get_prf(res):
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def gen_micro_macro_result(res):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_prf(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_prf(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    return {
        "micro_precision": round(micro_precision, 3),
        "micro_recall": round(micro_recall, 3),
        "micro_f1": round(micro_f1, 3),
        "macro_precision": round(macro_precision, 3),
        "macro_recall": round(macro_recall, 3),
        "macro_f1": round(macro_f1, 3)
    }


def null_accuracy_function(outputs, label, config, result=None):
    return None


def single_label_top1_accuracy(outputs, label, config, result=None):
    if result is None:
        result = []
    id1 = torch.max(outputs, dim=1)[1]
    id2 = label
    nr_classes = outputs.size(1)
    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
    for a in range(0, len(id1)):

        it_is = int(id1[a])
        should_be = int(id2[a])
        if it_is == should_be:
            result[it_is]["TP"] += 1
        else:
            result[it_is]["FP"] += 1
            result[should_be]["FN"] += 1

    return result


def multi_label_accuracy(outputs, label, config, result=None):
    if len(label[0]) != len(outputs[0]):
        raise ValueError('Input dimensions of labels and outputs must match.')

    outputs = outputs.data
    labels = label.data

    if result is None:
        result = []

    total = 0
    nr_classes = outputs.size(1)

    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})

    for i in range(nr_classes):
        outputs1 = (outputs[:, i] >= 0.5).long()
        labels1 = (labels[:, i].float() >= 0.5).long()
        total += int((labels1 * outputs1).sum())
        total += int(((1 - labels1) * (1 - outputs1)).sum())

        if result is None:
            continue

        result[i]["TP"] += int((labels1 * outputs1).sum())
        result[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        result[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        result[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())

    return result


def single_label_top2_accuracy(outputs, label, config, result=None):
    raise NotImplementedError

    if result is None:
        result = []

    id1 = torch.max(outputs, dim=1)[1]
    id2 = label
    nr_classes = outputs.size(1)
    while len(result) < nr_classes:
        result.append({"TP": 0, "FN": 0, "FP": 0, "TN": 0})
    for a in range(0, len(id1)):

        it_is = int(id1[a])
        should_be = int(id2[a])
        if it_is == should_be:
            result[it_is]["TP"] += 1
        else:
            result[it_is]["FP"] += 1
            result[should_be]["FN"] += 1

    _, prediction = torch.topk(outputs, 2, 1, largest=True)
    prediction1 = prediction[:, 0:1]
    prediction2 = prediction[:, 1:]

    prediction1 = prediction1.view(-1)
    prediction2 = prediction2.view(-1)

    return result



def IoU_softmax(predictions, targets, result = None, output_class = None):
    assert output_class != None
    if result == None:
        result = {"IoU": [0] * output_class,
                  "Dice": [0] * output_class,
                  "PA": [0] * output_class,
                  "mIoU": 0,
                  "cnt": [0.000001] * output_class}
    eps = 0.0001
    predictions = predictions.detach()
    targets = targets.detach()
    reverse_targets, reverse_predictions = [], []
    for idx in range(output_class):
        reverse_targets.append((targets == idx))
        reverse_predictions.append((predictions == idx))
    reverse_targets = torch.stack(reverse_targets, dim=0).float()
    reverse_predictions = torch.stack(reverse_predictions, dim=0).float()
    for idx in range(output_class):
        if torch.sum(reverse_targets[idx]) == 0:
            continue
        inter = torch.dot(reverse_predictions[idx].view(-1), reverse_targets[idx].view(-1))
        separate_sum = torch.sum(reverse_predictions[idx]) + torch.sum(reverse_targets[idx]) + eps
        union = separate_sum - inter
        total = torch.sum(targets == idx) + eps
        assert inter < union
        t1 = (2 * inter.float() + eps) / separate_sum.float()
        t2 = (inter.float() + eps) / union.float()
        t3 = (inter.float() + eps) / total.float()
        result["Dice"][idx] += float(t1)
        result["IoU"][idx] += float(t2)
        result["PA"][idx] += float(t3)
        result["cnt"][idx] += 1
    return result


def IoU_sigmoid(predictions, targets, result = None, output_class = None):
    if result == None:
        result = {"IoU": [0] * predictions.shape[1],
                  "Dice": [0] * predictions.shape[1],
                  "PA": [0] * predictions.shape[1],
                  "cnt": [0.0001] * predictions.shape[1]}
    eps = 0.0001
    predictions = predictions.permute(1, 0, 2, 3).float().contiguous()
    targets = targets.permute(1, 0, 2, 3).float().contiguous()
    for idx in range(predictions.shape[0]):
        if torch.sum(targets[idx]) == 0:
            continue
        inter = torch.dot(predictions[idx].view(-1), targets[idx].view(-1))
        separate_sum = torch.sum(predictions[idx]) + torch.sum(targets[idx]) + eps
        union = separate_sum - inter
        total = torch.sum(targets[idx]) + eps
        assert inter < union
        t1 = (2 * inter.float() + eps) / separate_sum.float()
        t2 = (inter.float() + eps) / union.float()
        t3 = (inter.float() + eps) / total.float()
        result["Dice"][idx] += t1
        result["IoU"][idx] += t2
        result["PA"][idx] += t3
        result["cnt"][idx] += 1
    return result