import datetime
import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TBLogger:
    def __init__(self, args, configs):
        self.output_name = f"{args.exp_name}_{args.env_name}_{args.seed}_{args.now}"

        try:
            log_dir = args.results_log_dir
        except AttributeError:
            log_dir = args.results_log_dir

        if log_dir is None:
            dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
            dir_path = os.path.join(dir_path, 'logs')
        else:
            dir_path = log_dir

        if not os.path.exists(dir_path):
            try:
                os.mkdir(dir_path)
            except:
                dir_path_head, dir_path_tail = os.path.split(dir_path)
                if len(dir_path_tail) == 0:
                    dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                os.mkdir(dir_path_head)
                os.mkdir(dir_path)

        try:
            self.full_output_folder = os.path.join(os.path.join(dir_path, 'logs_{}'.format(args.env_name)),
                                                   self.output_name)
        except:
            self.full_output_folder = os.path.join(os.path.join(dir_path, 'logs_{}'.format(args["env_name"])),
                                                   self.output_name)

        self.writer = SummaryWriter(log_dir=self.full_output_folder)

        print('logging under', self.full_output_folder)

        if not os.path.exists(self.full_output_folder):
            os.makedirs(self.full_output_folder)
        self.writer.add_text(
        "Experiments/exp_info",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{str(value)}|" for key, value in vars(args).items()])),
        )
        self.writer.add_text(
        "Experiments/hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{str(value)}|" for key, value in configs.items()])),
        )


    def add(self, name, value, x_pos):
        self.writer.add_scalar(name, value, x_pos)
