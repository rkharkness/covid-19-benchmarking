import logging
import os
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
import shutil
from timeit import default_timer as timer
import json
from tools.output_tool import print_IoU

from tools.eval_tool import valid, gen_time_str, output_value

logger = logging.getLogger(__name__)


def checkpoint(filename, model, optimizer, trained_epoch, config, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    save_params = {
        "model": model_to_save.state_dict(),
        "optimizer_name": config.get("train", "optimizer"),
        "optimizer": optimizer.state_dict(),
        "trained_epoch": trained_epoch,
        "global_step": global_step
    }

    try:
        torch.save(save_params, filename)
    except Exception as e:
        logger.warning("Cannot save models with error %s, continue anyway" % str(e))


def train(parameters, config, gpu_list):
    epoch = config.getint("train", "epoch")
    batch_size = config.getint("train", "batch_size")

    output_file = config.get("output", "output_file") if config.get("output", "output_file") != "None" else None
    if output_file is not None:
        if os.path.exists(output_file):
            os.remove(output_file)

    output_time = config.getint("output", "output_time")
    test_time = config.getint("output", "test_time")

    output_path = os.path.join(config.get("output", "model_path"), config.get("output", "model_name"))
    if os.path.exists(output_path):
        logger.warning("Output path exists, check whether need to change a name of model")
    os.makedirs(output_path, exist_ok=True)

    trained_epoch = parameters["trained_epoch"]
    model = parameters["model"]
    optimizer = parameters["optimizer"]
    dataset = parameters["train_dataset"]
    global_step = parameters["global_step"]
    output_function = parameters["output_function"]

    step_size = config.getint("train", "step_size")
    gamma = config.getfloat("train", "lr_multiplier")
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    exp_lr_scheduler.step(trained_epoch)

    logger.info("Training start....")

    if config.get("model", "model_name") in ["Pipeline"]:
        print("Epoch  Stage  Iterations  Time Usage    Loss   \tOutput Information")
    else:
        print("Epoch  Stage  Iterations  Time Usage    Loss    Output Information")

    total_len = len(dataset)
    more = ""
    if total_len < 10000:
        more = "\t"
    for epoch_num in range(trained_epoch, epoch):
        start_time = timer()
        current_epoch = epoch_num

        exp_lr_scheduler.step(current_epoch)

        acc_result = [None, None]
        total_loss = None

        output_info = ""
        step = -1
        for step, data in enumerate(dataset):
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    if len(gpu_list) > 0:
                        data[key] = Variable(data[key].cuda())
                    else:
                        data[key] = Variable(data[key])

            optimizer.zero_grad()

            results = model(data, config, gpu_list, acc_result, "train")

            loss, acc_result = results["loss"], results["acc_result"]

            if type(loss) == list:
                loss[0].backward()
                loss[1].backward()
                if total_loss == None:
                    total_loss = [0, 0]
                total_loss[0] += float(loss[0])
                total_loss[1] += float(loss[1])
            else:
                loss.backward()
                if total_loss == None:
                    total_loss = 0
                total_loss += float(loss)
            optimizer.step()

            if step % output_time == 0:
                output_stage_1 = output_function(acc_result[0], config)
                output_stage_2 = output_function(acc_result[1], config)

                delta_t = timer() - start_time

                if type(loss) == list:
                    output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                             "%.3lf, %.3lf\t" %(total_loss[0] / (step + 1), total_loss[1] / (step + 1)), output_stage_1 + output_stage_2, '\r', config)
                else:
                    output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                        gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                                 "%.3lf" % (total_loss / (step + 1)), output_stage_1 + output_stage_2, '\r', config)

            global_step += 1

        if type(loss) == list:
            output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf, %.3lf\t" % (total_loss[0] / (step + 1), total_loss[1] / (step + 1)), output_stage_1 + output_stage_2, None,
                         config)
        else:
            output_value(current_epoch, "train", "%d/%d" % (step + 1, total_len), "%s/%s" % (
                gen_time_str(delta_t), gen_time_str(delta_t * (total_len - step - 1) / (step + 1))),
                         "%.3lf" % (total_loss / (step + 1)), output_info, None, config)

        if output_file is not None:
            f = open(output_file, "a")
            f.write("epoch = %d" %epoch_num + json.dumps(print_IoU(acc_result[1]))+ "\n")
            f.close()

        if step == -1:
            logger.error("There is no data given to the model in this epoch, check your data.")
            raise NotImplementedError

        checkpoint(os.path.join(output_path, "%d.pkl" % current_epoch), model, optimizer, current_epoch, config,
                   global_step)

        if current_epoch % test_time == 0:
            with torch.no_grad():
                valid(model, parameters["valid_dataset"], current_epoch, None, config, gpu_list, output_function)
