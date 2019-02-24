import os
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from config import Config
import utils
from optim.adam import Adam
import shutil
import copy
from models.scse_pyramid_unet import UNet
from matplotlib import pyplot as plt

def start_run():
    config = Config()

    if os.path.exists(config.path):
        while True:
            cont_str = input("Name has been used. Continue and delete other log files? (y/n)")
            if cont_str.lower() == 'n':
                exit()
            elif cont_str.lower() == 'y':
                shutil.rmtree(config.path)
                break
            else:
                print("Invalid input.")

    device = torch.device("cuda")

    # tensorboard
    writer = SummaryWriter(log_dir=os.path.join(config.path, "tb"))
    writer.add_text('config', config.as_markdown(), 0)
    
    logger = utils.get_logger(os.path.join(config.path, "{}.log".format(config.name)))
    config.print_params(logger.info)

    logger.info("Logger is set - training start")

    # set gpu device id
    logger.info("Set GPU device {}".format(config.gpu))
    torch.cuda.set_device(config.gpu)

    # set seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.benchmark = True

    #TODO: fix folds/cv
    data_params, train_data, valid_data = utils.get_data()

    model = UNet(config.total_channels_to_add, data_params['num_classes'], data_params['input_channels'],
            config.shake_drop, not config.no_scse, config.num_downsamples, config.num_blocks_per_downsample)

    model = model.to(device)

    logger.info("Model Size (MB): {}".format(utils.param_size(model)))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=config.workers,
                                               pin_memory=True)

    nb_iters_train = config.epochs * len(train_loader)

    if config.lr_finder:
        w_sched_lr = utils.ExpFinderSchedule(config.w_lr_start, config.w_lr_end, nb_iters_train)
    else:
        w_sched_lr = utils.PiecewiseLinearOrCos([0.0, config.first_prop * nb_iters_train, nb_iters_train],
                np.array([config.w_lr_start, config.w_lr_middle, 
                    config.w_lr_end]),
                [False, True])

    if config.wd_finder:
        weight_decay = utils.ExpFinderSchedule(config.w_weight_decay, config.w_weight_decay_end, nb_iters_train) 
    else:
        weight_decay = config.w_weight_decay


    w_optim = Adam(model.parameters(),
                   lr=w_sched_lr,
                   weight_decay=weight_decay)

    cur_step = 0
    best_iou = 0.

    # training loop
    for epoch in range(config.epochs):

        cur_step = train(train_loader, model, w_optim, epoch, writer, device, config, logger, cur_step)

        if (epoch + 1) % config.val_freq == 0:
            # validation
            total_iou = validate(valid_loader, model, epoch, cur_step, writer, device, config, logger)

        saves = ['checkpoint']
        is_best = best_iou < total_iou
        # save
        if is_best:
            best_iou = iou
            saves.append('best')
        utils.save_item(model, config.path, saves)
        print("")

    logger.info("Final best iou = {:.4%}".format(best_iou))

def train(train_loader, model, w_optim, epoch, writer, device, config, logger, cur_step):
    iou = utils.AverageMeter()
    losses = utils.AverageMeter()

    initial_step = copy.deepcopy(cur_step)

    for step, trn_d in enumerate(train_loader):
        model.train()

        trn_d = trn_d.to(device, non_blocking=True)
        trn_X = trn_d[:,0]
        trn_y = trn_d[:,1] * 255.
        reshape = lambda x : x.reshape(x.size(0), 1, x.size(1), x.size(2))
        trn_X = reshape(trn_X)
        trn_y = reshape(trn_y)
        N = trn_X.size(0)

        logits_w = model(trn_X)
        loss = model.loss(logits_w, trn_y)
        w_grads = torch.autograd.grad(loss, w_optim.params())
        w_optim.step(w_grads)

        batch_iou = model.iou(logits_w, trn_y)
        losses.update(loss.item(), N)
        iou.update(batch_iou, N)

        if step % config.print_freq == 0 or step == len(train_loader)-1:
            logger.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "iou {iou.avg:.1%}".format(
                    epoch+1, config.epochs, step, len(train_loader)-1, losses=losses,
                    iou=iou))

        #assumes one lr and wd value
        for group in w_optim.param_groups:
            lr = w_optim.get(group['lr'])
            wd = w_optim.get(group['weight_decay'])
            break


        writer.add_scalar('train/lr', lr, cur_step)
        writer.add_scalar('train/wd', wd, cur_step)
        writer.add_scalar('train/loss', loss.item(), cur_step)
        writer.add_scalar('train/iou', batch_iou, cur_step)

        cur_step += 1

    logger.info("Train: [{:2d}/{}] Final iou {:.4%}".format(epoch+1, config.epochs, iou.avg))

    return cur_step

def validate(valid_loader, model, epoch, cur_step, writer, device, config, logger):
    iou = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with torch.no_grad():
        for step, trn_d in enumerate(valid_loader):
            trn_d = trn_d.to(device, non_blocking=True)
            X = trn_d[:,0]
            y = trn_d[:,1] * 255.
            reshape = lambda x : x.reshape(x.size(0), 1, x.size(1), x.size(2))
            X = reshape(X)
            y = reshape(y)
            N = X.size(0)

            logits = model(X)

            loss = model.loss(logits, y)

            batch_iou = model.iou(logits, y)
            losses.update(loss.item(), N)
            iou.update(batch_iou, N)

            if step % config.print_freq == 0 or step == len(valid_loader)-1:
                logger.info(
                    "Valid: [{:2d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "iou {iou.avg:.1%}".format(
                        epoch+1, config.epochs, step, len(valid_loader)-1, losses=losses,
                        iou=iou))

            break

    writer.add_scalar('val/loss', losses.avg, cur_step)
    writer.add_scalar('val/iou', iou.avg, cur_step)

    logger.info("Valid: [{:2d}/{}] Final iou {:.4%}".format(epoch+1, config.epochs, iou.avg))

    return iou.avg

if __name__ == '__main__':
    start_run()

