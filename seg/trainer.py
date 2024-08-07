import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter

from .metrics import MultiAP, MultiMIOU

def create_lr_scheduler(optimizer, params: dict)->torch.optim.lr_scheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer: model optimizer
        params: scheduler parameters
    
    Return:
        scheduler: a learning rate scheduler
    """
    scheduler = None
    scheduler_params = params.get("parameters", None)

    if params["name"] == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif params["name"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)

    return scheduler

class ModelTrainer:
    """A model trainer class used in PyTorch model training.
    """

    def __init__(self, params):
        self.model_name     = params.get("model_name", "MyModel")
        self.save_path      = params.get("save_path", "./")
        self.loss_fn        = params.get("loss_fn", nn.MSELoss())
        self.metric_fn      = params.get("metric_fn", None)
        self.epochs         = params.get("epochs", 100)
        self.eval_per_epoch = params.get("eval_per_epoch", 10)
        self.save_per_epoch = params.get("save_per_epoch", 10)
        self.device         = params.get("device", "cpu")

        ## whether to use tensorboard to record training process
        self.tensorboard = params.get("tensorboard", False)
        if self.tensorboard:
            self.writer = SummaryWriter(os.path.join(self.save_path, f"{self.model_name}"))

    def fit(self, model, train_loader, eval_loader, optimizer, scheduler=None):
        device = self.device
        best_eval_loss = float('inf')

        ## move model to device
        model.to(device)

        ## add model architecture flag
        self.model_graph = False

        for epoch in range(self.epochs):
            train_loss = self._train(model, train_loader, optimizer, epoch+1)
            if scheduler:
                scheduler.step()

            ## evaluate the model when needed
            if (epoch + 1) % self.eval_per_epoch == 0:
                eval_loss, eval_metric = self._evaluate(model, eval_loader, epoch+1)

                # save best model based on evaluation loss
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    model_path = os.path.join(self.save_path, f"{self.model_name}_best.pth")
                    self._save_model(model, model_path)
                    print(f"Best model is saved in {model_path}.")

            ## save the model when needed
            if (epoch + 1) % self.save_per_epoch == 0:
                model_path = os.path.join(self.save_path, f"{self.model_name}_epoch_{epoch+1:04d}.pth")
                self._save_model(model, model_path)
                print(f"Model is saved in {model_path}")

        ## flush to disk
        self._flush_to_disk()

    def _step(self, model, X, y, optimizer):
        """Take one step training.
        """

        ## feed data to model
        pred = model(X)

        ## back-propagation
        optimizer.zero_grad()
        loss = self.loss_fn(pred, y)
        loss.backward()

        ## one step optimization
        optimizer.step()

        return loss.item()

    def _train(self, model, train_loader, optimizer, epoch):
        """Model train.
        """

        ## change to train mode
        model.train()
        running_loss = 0.
        device = self.device

        ## loop over training set
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch}/{self.epochs}, Training", leave=False):
            X, y = X.to(device), y.to(device)
            running_loss += self._step(model, X, y, optimizer) * X.size(0)

            if not self.model_graph:
                tqdm.write("Model architecture is saved!")
                self._write_model_graph(model, X)
                self.model_graph = True

        train_loss = running_loss / len(train_loader.dataset)

        ## write to tensorboard
        self._write_tensorboard_scalar("Loss/train", train_loss, epoch)

        tqdm.write(f'Epoch {epoch}/{self.epochs}, Training Loss: {train_loss:.4e}')

        return train_loss

    def _evaluate(self, model, eval_loader, epoch):
        """Model evaluation.
        """

        running_loss = 0.0
        eval_metric  = 0.0

        y_true = []
        y_pred = []
        device = self.device

        ## change to evaluation mode
        model.eval()

        ## no need to compute gradients in evaluation mode
        with torch.no_grad():
            ## loop over evaluation set
            for X, y in eval_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = self.loss_fn(pred, y)
                running_loss += loss.item() * X.size(0)

                y_pred.append(pred.detach().cpu())
                y_true.append(y.detach().cpu())

        eval_loss = running_loss / len(eval_loader.dataset)

        ## evaluation with metric
        if self.metric_fn:
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)

            eval_metric = self.metric_fn(y_pred, y_true)

        ## write to tensorboard
        self._write_tensorboard_scalar("Loss/eval", eval_loss, epoch)
        self._write_tensorboard_scalar("Metric/eval", eval_metric, epoch)

        tqdm.write(f'Epoch {epoch}/{self.epochs}, Evaluation Loss: {eval_loss:.4e}, Evaluation Metric: {eval_metric:.4f}')

        return eval_loss, eval_metric

    def _save_model(self, model, filepath):
        """Save current model."""
        torch.save(model.state_dict(), filepath)

    def _write_tensorboard_scalar(self, label, value, epoch):
        """A helper function to write scalars with tensorboard.
        """
        if self.tensorboard:
            self.writer.add_scalar(label, value, epoch)

    def _write_model_graph(self, model, data):
        """A helper function to add model architecture with tensorboard.
        """
        if self.tensorboard:
            self.writer.add_graph(model, data, verbose=False)

    def _flush_to_disk(self):
        """A helper function to flush the event file to disk.
        """
        if self.tensorboard:
            self.writer.flush()

class SegModelTrainer(ModelTrainer):
    """Model trainer for segmentation tasks. This trainer updates the evaluation methods.
    """

    def __init__(self, params):
        super().__init__(params)

        ## update the parameters for segmentation task
        self.num_classes = params.get("num_classes", 10)
        self.metric_fn = {
            # "AP": MultiAP(self.num_classes, average=None),
            "MIOU": MultiMIOU(self.num_classes)
        }
    
    def _evaluate(self, model, eval_loader, epoch):
        """Modified evaluation step for segmentation task.
        """

        ## evaluation loss is the same
        running_loss = 0.0
        eval_metric  = 0.0

        device = self.device
        scores = {metric_name: [] for metric_name in  self.metric_fn}

        ## change to evaluation mode
        model.eval()

        ## no need to compute gradients in evaluation mode
        with torch.no_grad():
            ## loop over evaluation set
            for X, y in eval_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)

                loss = self.loss_fn(pred, y)
                running_loss += loss.item() * X.size(0)

                pred = torch.where(pred > 0, 1.0, 0.0)
                y_pred = pred.detach().squeeze().unsqueeze(0).int().cpu()
                y_true = y.detach().squeeze().unsqueeze(0).int().cpu()

                # ## check pred
                # print(y_pred.permute(0,2,3,1)[0,:,:,0])
                # print(y_pred.permute(0,2,3,1)[0,:,:,1])
                # raise NotImplementedError

                ## evaluation with metric_fns
                for metric_name, fn in self.metric_fn.items():
                    # print(y_pred.shape, y_true.shape)
                    score = fn(y_pred, y_true)
                    # print(score)
                    scores[metric_name].append(score.unsqueeze(0))
            
            ## take average over multiple classes
            eval_metric_scores = ""
            for metric_name, metric_scores in scores.items():
                metric_scores = torch.cat(metric_scores, dim=0)
                # print(f"{metric_name} on individual sample : {metric_scores}")
                metric_scores = metric_scores.mean(dim=0)
                # print(f"{metric_name} after average : {metric_scores}")

                for label, score in enumerate(metric_scores):
                    tag = f"{metric_name}/class: {label}"
                    self._write_tensorboard_scalar(tag, score, epoch)
                    print(f"{tag}: {score:.4f}")
            
                ## averged metric
                tag = f"{metric_name}/average"
                avg_score = metric_scores.mean()
                self._write_tensorboard_scalar(tag, avg_score, epoch)

                eval_metric_scores += f"Evaluation Average {metric_name}: {avg_score:.4} "
        
        eval_loss = running_loss / len(eval_loader.dataset)
        self._write_tensorboard_scalar("Loss/eval", eval_loss, epoch)
        tqdm.write(f'Epoch {epoch}/{self.epochs}, Evaluation Loss: {eval_loss:.4e}, {eval_metric_scores}')

        return eval_loss, eval_metric