"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import ray
from ray import tune
from ray.air import session
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ignite.metrics import PSNR, SSIM
import numpy as np

config = {
    "lr": tune.loguniform(1e-4, 1e-1),
    "n_layers_D": tune.randint(3, 6),
    "lambda_A": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "lambda_B": tune.sample_from(lambda _: 2**np.random.randint(2, 9)),
    "batch_size": tune.choice([8, 16, 32])
}

def create_val(opt):
    opt.phase = "val"
    val = create_dataset(opt)
    return val

def train_model(config):
    opt = TrainOptions().parse()
    opt.lr = config["lr"]
    opt.lambda_A = config["lambda_A"]
    opt.lambda_A = config["lambda_B"]
    opt.n_layers_D = config["n_layers_D"]
    opt.batch_size = config["batch_size"]


    validation_dataset = create_dataset(opt)
    opt.phase = "train"

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options    

    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images for A to B = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations
    
    psnr_metric = PSNR()
    ssim_metric = SSIM()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        psnr_metric.reset()
        ssim_metric.reset()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.eval_freq == 0:
                val_ssim = 0.0
                val_psnr = 0.0
                for i, val_data in enumerate(validation_dataset):
                    with torch.no_grad():
                        model.set_input(val_data)
                        model.forward()
                        val_psnr += psnr_metric.compute(model.real_B, model.fake_B)
                        val_ssim += ssim_metric.compute(model.real_B, model.fake_B)
                val_psnr /= len(validation_dataset)
                val_ssim /= len(validation_dataset)
                session.report({"psnr":val_psnr, "ssim":val_ssim, "sum": val_psnr+val_ssim})

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    ray.init(num_cpus=2)

    algo = HyperOptSearch(
        metric="psnr",
        mode="max"
    )
    
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1
    )

    tuner = tune.Tuner(
        train_model,
        tune_config=tune.TuneConfig(
            num_samples=10,
            metric=opt.eval_metric,
            mode="max",
            search_alg=algo,
            scheduler=scheduler
        ),
        param_space=config
    )
    results = tuner.fit()
    
    best_result = results.get_best_result(metric=opt.eval_metric, mode="max")
    # logdir = best_result.log_dir
    print("Best Config:", best_result.config)
    ray.shutdown()
