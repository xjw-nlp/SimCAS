import os
from os.path import exists, join
import json
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


class Recorder():
    def __init__(self, args, log=True, desc=None):
        self.log = log
        now = datetime.now()
        date = now.strftime("%y-%m-%d")
        cur_dir = f"./cache/{args.config}"
        if not os.path.isdir(cur_dir):
            os.mkdir(cur_dir)
        id = len(os.listdir(cur_dir))
        self.dir = os.path.join(cur_dir, f"{date}-{id}")
        if self.log:
            try:
                os.mkdir(self.dir)
            except:
                self.dir += '-more'
                os.mkdir(self.dir)
            if desc:
                with open(os.path.join(self.dir, "readme.txt"), "w") as f:
                    f.write(desc)
                self.f = open(os.path.join(self.dir, "log.txt"), "w")
            self.f = open(os.path.join(self.dir, "log.txt"), "w")
            self.writer = SummaryWriter(os.path.join(self.dir, "log"), flush_secs=60)
        
    def write_config(self, args, models, name):
        if self.log:
            with open(os.path.join(self.dir, "config.txt"), "w") as f:
                print(name, file=f)
                print(args, file=f)
                print(file=f)
                for (i, x) in enumerate(models):
                    print(x, file=f)
                    print(file=f)
        print(args)
        print()
        for (i, x) in enumerate(models):
            print(x)
            print()

    def print(self, x=None):
        if x is not None:
            print(x, flush=True)
        else:
            print(flush=True)
        if self.log:
            if x is not None:
                print(x, file=self.f, flush=True)
            else:
                print(file=self.f, flush=True)

    def __del__(self):
        if self.log:
            self.f.close()
            self.writer.close()

    def save(self, model, name):
        if self.log:
            torch.save(model.state_dict(), os.path.join(self.dir, name))


