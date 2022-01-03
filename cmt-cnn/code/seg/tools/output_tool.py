import json

from .accuracy_tool import gen_micro_macro_result


def null_output_function(data, config, *args, **params):
    return ""


def basic_output_function(data, config, *args, **params):
    which = config.get("output", "output_value").replace(" ", "").split(",")
    temp = gen_micro_macro_result(data)
    result = {}
    for name in which:
        result[name] = temp[name]

    return json.dumps(result, sort_keys=True, ensure_ascii=False)

def IoU_output_function(data, config, *args, **params):
    lesion = True
    if len(data["IoU"]) == 2:
        lesion = False
    result = {}
    for name in ["IoU", "Dice", "PA"]:
        result[name] = ""
        for idx in range(len(data[name])):
            if idx != len(data[name]) - 1:
                result[name] += "%s:%.2lf," %(str(idx), data[name][idx] / data["cnt"][idx])
            else:
                result[name] += "%s:%.2lf" % (str(idx), data[name][idx] / data["cnt"][idx])
    result["mIoU"], result["mDC"], result["mPA"] = 0, 0, 0
    if lesion:
        for idx in range(5):
            result["mIoU"] += data["IoU"][idx] / data["cnt"][idx]
            result["mDC"] += data["Dice"][idx] / data["cnt"][idx]
            result["mPA"] += data["PA"][idx] / data["cnt"][idx]
        result["mDC"] = "%.3lf" % (result["mDC"] / 5)
        result["mIoU"] = "%.3lf" % (result["mIoU"] / 5)
        result["mPA"] = "%.3lf" % (result["mPA"] / 5)
    else:
        for idx in range(1, 2):
            result["mIoU"] += data["IoU"][idx] / data["cnt"][idx]
            result["mDC"] += data["Dice"][idx] / data["cnt"][idx]
            result["mPA"] += data["PA"][idx] / data["cnt"][idx]
        result["mDC"] = "%.3lf" % (result["mDC"])
        result["mIoU"] = "%.3lf" % (result["mIoU"])
        result["mPA"] = "%.3lf" % (result["mPA"])
    out_result = {}
    out_result["mDC"] = result["mDC"]
    out_result["mIoU"] = result["mIoU"]
    out_result["mPA"] = result["mPA"]
    return json.dumps(out_result, sort_keys=True, ensure_ascii=False)


def print_IoU(data):
    result = {}
    for name in ["IoU", "Dice", "PA"]:
        result[name] = ""
        for idx in range(len(data[name])):
            if idx != len(data[name]) - 1:
                result[name] += "%s:%.3lf," % (str(idx), data[name][idx] / data["cnt"][idx])
            else:
                result[name] += "%s:%.3lf" % (str(idx), data[name][idx] / data["cnt"][idx])
    return result