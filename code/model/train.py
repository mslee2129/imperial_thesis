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
from ray.tune.suggest.hyperopt import HyperOptSearch
from ignite.metrics import PSNR

config = {
    "lr": tune.choice([0.0001, 0.001, 0.01]),
    "n_layers_D": tune.choice([3, 4, 5]),
    "lambda_A": tune.choice([1, 10, 50]),
    "lambda_B": tune.choice([1, 10, 50]),
    "batch_size": tune.choie([8, 16])
}

def train_model(config):
    opt = TrainOptions().parse()   # get training options
    opt.lr = config["lr"]
    opt.lambda_A = config["lambda_A"]
    opt.lambda_A = config["lambda_B"]
    opt.n_layers_D = config["n_layers_D"]
    opt.batch_size = config["batch_size"]

    opt.phase = "train"

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    opt.phase = "val"
    validation_dataset = create_dataset(opt)

    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images for A to B = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    psnr_metric = PSNR()

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights


            if total_iters % opt.eval_freq == 0:
                val_psnr = 0.0
                for i, val_data in enumerate(validation_dataset):
                    with torch.no_grad():
                        model.set_input(val_data)
                        model.forward()
                        val_psnr += psnr_metric.compute(model.real_B, model.fake_B)
                    val_psnr /= len(validation_dataset)
                    tune.report(psnr=val_psnr)

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
    ray.init()
    analysis = tune.run(train_model, 
                        config=config, 
                        search_alg=HyperOptSearch(space=config, metric="psnr", mode="max"), 
                        num_samples=10,
                        metric="psnr",
                        mode="max",
                        resources_per_trial={"gpu": 1})
    
    best_config = analysis.get_best_config(metric="psnr", mode="max")
    print("Best Config:", best_config)
    ray.shutdown()