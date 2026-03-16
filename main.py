


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import utils
import numpy as np
import modules
import torch
import torch.nn as nn
import datasets
import yaml
import json
import faulthandler

faulthandler.enable()

from seq_scripts import seq_train, seq_eval
import slr_network


class SLRProcessor(object):
    def __init__(self, arg):
        super().__init__()

        self.arg = arg
        self.save_arg()

        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)

        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(
            self.arg.work_dir, self.arg.print_log, self.arg.log_interval
        )

        self.dataset = {}
        self.data_loader = {}

        self.load_dataset_info()
        with open(self.arg.dataset_info["dict_path"], "r") as f:
            self.gloss_dict = json.load(f)

        self.model, self.optimizer = self.loading()
        self.best_dev_wer = 1000
        self.tasks = self.arg.dataset[-2:]

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open(f"{self.arg.work_dir}/config.yaml", "w") as f:
            yaml.dump(arg_dict, f)

    def model_to_device(self, model):
        # Respect utils.GpuDataParallel output_device
        model = model.to(self.device.output_device)
        return model

    def _move_optimizer_state_to_device(self, torch_optim, device):
        # Move Adam/AdamW momentum buffers, etc. to GPU
        for state in torch_optim.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    def load_model_weights(self, model, weight_path):
        ckpt = torch.load(weight_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print(f"Successfully Remove Weights: {w}.")
                else:
                    print(f"Can Not Remove Weights: {w}.")

        model.load_state_dict(state_dict, strict=False)

    def load_checkpoint_weights(self, model, optimizer):
        ckpt = torch.load(self.arg.load_checkpoints, map_location="cpu")

        # model
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print(f"Successfully Remove Weights: {w}.")
                else:
                    print(f"Can Not Remove Weights: {w}.")

        model.load_state_dict(state_dict, strict=False)

        # optimizer
        if "optimizer_state_dict" in ckpt:
            # Your checkpoints are saved via self.optimizer.state_dict() (wrapper),
            # so try wrapper load first; fallback to raw torch optimizer load.
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                optimizer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # scheduler
        if (
            "scheduler_state_dict" in ckpt
            and hasattr(optimizer, "scheduler")
            and optimizer.scheduler is not None
        ):
            try:
                optimizer.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                print("Warning: could not load scheduler_state_dict:", e)

        # move optimizer states to same device as model (IMPORTANT for AdamW)
        dev = next(model.parameters()).device
        self._move_optimizer_state_to_device(optimizer.optimizer, dev)

        # resume epoch
        if "epoch" in ckpt:
            self.arg.optimizer_args["start_epoch"] = int(ckpt["epoch"]) + 1
            print(
                "Resume from epoch:",
                ckpt["epoch"],
                "=> start_epoch set to",
                self.arg.optimizer_args["start_epoch"],
            )

        # rng
        if hasattr(self, "rng") and "rng_state" in ckpt:
            try:
                self.rng.load_rng_state(ckpt["rng_state"])
            except Exception as e:
                print("Warning: could not load rng_state:", e)

    def build_module(self, args):
        model_class = getattr(slr_network, self.arg.model)
        model = model_class(**args, gloss_dict=self.gloss_dict)
        return model

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=False,
            num_workers=self.arg.num_worker,
            collate_fn=self.feeder.collate_fn,
        )

    def load_data(self):
        print("Loading data")
        self.feeder = getattr(datasets, self.arg.feeder)

        dataset_list = zip(["train", "dev", "test"], [True, False, False])

        g2i_dict = {k: v["index"] for k, v in self.gloss_dict["gloss2id"].items()}

        for mode, train_flag in dataset_list:
            arg = self.arg.feeder_args
            arg["mode"] = mode
            arg["transform_mode"] = train_flag
            arg["dataset"] = self.arg.dataset
            self.dataset[mode] = self.feeder(gloss_dict=g2i_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)

        print("Loading data finished.")

    def load_dataset_info(self):
        with open(f"./configs/dataset_configs/{self.arg.dataset}.yaml", "r") as f:
            self.arg.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    def loading(self):
        # Set GPU device selection inside their wrapper
        self.device.set_device(self.arg.device)
        if "temporal_encoder" not in self.arg.model_args:
            self.arg.model_args["temporal_encoder"] = "bilstm"
        if self.arg.temporal_encoder is not None:
            self.arg.model_args["temporal_encoder"] = self.arg.temporal_encoder
        
        print("Loading model")
        model = self.build_module(self.arg.model_args)
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        # ✅ Move model to GPU FIRST so we can safely move optimizer state tensors too.
        model = self.model_to_device(model)

        # Now load weights/checkpoint
        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def judge_save_eval(self, epoch):
        save_model = (epoch % self.arg.save_interval == 0) and (epoch >= 0)
        eval_model = (epoch % self.arg.eval_interval == 0) and (epoch >= 0)
        return save_model, eval_model

    def save_model(self, epoch, save_path):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.optimizer.scheduler.state_dict()
                if hasattr(self.optimizer, "scheduler") and self.optimizer.scheduler is not None
                else {},
                "rng_state": self.rng.save_rng_state() if hasattr(self, "rng") else None,
            },
            save_path,
        )

    def custom_save_model(self, dev_wer, epoch, save_dir):
        dirs = os.listdir(save_dir)
        dirs = list(filter(lambda x: x.endswith(".pt"), dirs))
        assert len(dirs) <= 2

        best_path, cur_path = None, None
        for item in dirs:
            if "best" in item:
                best_path = os.path.join(save_dir, item)
            if "cur" in item:
                cur_path = os.path.join(save_dir, item)

        if cur_path is not None:
            os.system(f"rm {cur_path}")

        model_path = f"{save_dir}cur_dev_{dev_wer:05.2f}_epoch{epoch}_model.pt"
        self.save_model(epoch, model_path)

        if best_path is not None:
            if dev_wer <= self.best_dev_wer:
                os.system(f"rm {best_path}")
                model_path = f"{save_dir}best_dev_{dev_wer:05.2f}_epoch{epoch}_model.pt"
                self.save_model(epoch, model_path)
                self.best_dev_wer = dev_wer
        else:
            model_path = f"{save_dir}best_dev_{dev_wer:05.2f}_epoch{epoch}_model.pt"
            self.save_model(epoch, model_path)
            self.best_dev_wer = dev_wer

    def train(self):
        self.recoder.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))
        dev_error = 1000.0  # safe init

        for epoch in range(self.arg.optimizer_args["start_epoch"], self.arg.num_epoch):
            save_model, eval_model = self.judge_save_eval(epoch)

            seq_train(
                self.data_loader["train"],
                self.model,
                self.optimizer,
                self.device,
                epoch,
                self.recoder,
                **self.arg.train_args,
            )

            if eval_model:
                dev_error = self.test("dev", epoch)
                self.recoder.print_log("Dev WER: {:05.2f}%".format(dev_error))

            if save_model:
                self.custom_save_model(dev_error, epoch, self.arg.work_dir)

    def test(self, mode, epoch):
        wer = seq_eval(
            self.arg,
            self.data_loader[mode],
            self.model,
            self.device,
            mode,
            epoch,
            self.arg.work_dir,
            self.recoder,
            self.tasks,
            self.arg.evaluate_tool,
        )
        return wer

    def start(self):
        if self.arg.phase == "train":
            self.train()
        elif self.arg.phase == "test":
            self.recoder.print_log("Model:   {}.".format(self.arg.model))
            self.recoder.print_log("Weights: {}.".format(self.arg.load_weights))
            self.test("dev", 6667)
            self.test("test", 6667)
            self.recoder.print_log("Evaluation Done.\n")


if __name__ == "__main__":
    sparser = utils.get_parser()
    p = sparser.parse_args()

    if p.config is not None:
        with open(p.config, "r") as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)

        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        sparser.set_defaults(**default_arg)

    args = sparser.parse_args()
    main_processor = SLRProcessor(args)
    main_processor.start()
