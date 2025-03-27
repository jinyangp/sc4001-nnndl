import os
import argparse
import torch
from datetime import timedelta
from omegaconf import OmegaConf
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.util import create_model, load_state_dict, instantiate_from_config
from src.data.dataloader import custom_collate_fn

def main(args):

    model_config = args.config
    resume_path = args.resume_path
    gpus = args.gpus
    batch_size = args.batch_size
    num_workers = len(gpus) * batch_size
    proj_name = args.name
    max_epochs = args.max_epochs

    logdir = os.path.join('./logs/', proj_name)    
    # logger_freq = 10000
    learning_rate = 5e-5
    
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_config).cpu()
    if args.resume_path is not None:
        model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
    model.learning_rate = learning_rate
    config = OmegaConf.load(model_config)

    # data
    test_dataset = instantiate_from_config(config.dataset.test)
    test_dataloader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=custom_collate_fn, shuffle=False, pin_memory=True)
    trainer = pl.Trainer(
                        strategy="ddp",
                        accelerator="gpu", devices=gpus, 
                        precision=32,
                        accumulate_grad_batches=4,
                        default_root_dir=logdir,
                        check_val_every_n_epoch=1,
                        num_sanity_val_steps=1,
                        max_epochs=max_epochs)

    # Train!
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')

    # Adding arguments
    parser.add_argument('--name', type=str)
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--gpus', nargs='+', type=int, default=[1])
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)    
    # Parsing arguments
    args = parser.parse_args()

    # Calling the main function with parsed arguments
    main(args)