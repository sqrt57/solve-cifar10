import math
import collections
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from solve.dataset import DataSet, DataLoader


Hyper = collections.namedtuple('Hyper', 'name seed model optimizer optimizer_kwargs nepochs lr lr_scheduler preprocess', defaults=(False,))
Result = collections.namedtuple('Result', 'hyper model epochs lrs train_metrics validation_metrics')


class Metrics:
    def name(self) -> str:
        raise NotImplementedError()

    def batch(self, model, features_batch, labels_batch, prediction, loss) -> tuple:
        raise NotImplementedError()

    def summarize(self, chunks) -> Any:
        raise NotImplementedError()


class LossMetrics(Metrics):
    def name(self):
        return "Loss"

    def batch(self, model, features_batch, labels_batch, prediction, loss):
        return (loss, features_batch.shape[0])
    
    def summarize(self, chunks):
        total_loss = sum(chunk[0] for chunk in chunks)
        total_count = sum(chunk[1] for chunk in chunks)
        return total_loss / total_count

   
class AccuracyMetrics(Metrics):
    def name(self):
        return "Accuracy"

    def batch(self, model, features_batch, labels_batch, prediction, loss):
        correct = (prediction.argmax(1) == labels_batch).sum().item()
        return (correct, features_batch.shape[0])
    
    def summarize(self, chunks):
        total_correct = sum(chunk[0] for chunk in chunks)
        total_count = sum(chunk[1] for chunk in chunks)
        return total_correct * 100 / total_count


class Trainer:
    def __init__(self, training: DataLoader, validation: DataSet, device = None, metrics: list[Metrics] | None = None):
        self.device = device
        self.training = training
        self.validation = validation
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.metrics = metrics or [LossMetrics(), AccuracyMetrics()]

    def run_scenario(self, hyper: Hyper):
        print(f"Running scenario {hyper.name}: model={hyper.model.__name__} seed={hyper.seed}")
        torch.random.manual_seed(hyper.seed)
        model = hyper.model()
        if self.device:
            model = model.to(device=self.device)
        optimizer = hyper.optimizer(model.parameters(), lr=hyper.lr, **hyper.optimizer_kwargs)
        lr_scheduler = hyper.lr_scheduler(optimizer)

        epochs = []
        lrs = []
        train_metrics = { metric.name(): [] for metric in self.metrics }
        validation_metrics = { metric.name(): [] for metric in self.metrics }

        def preprocess(features):
            if hyper.preprocess:
                return features - features.mean(dim=(2,3), keepdim=True)
            return features

        def batch(features_batch, labels_batch):
            model.train()
            optimizer.zero_grad()
            pred = model.forward(preprocess(features_batch))
            loss = self.loss_fn(pred, labels_batch)
            (loss / features_batch.shape[0]).backward()
            optimizer.step()
            return pred, loss

        def testbatch(model, features_batch, labels_batch):
            with torch.no_grad():
                model.eval()
                pred = model.forward(preprocess(features_batch))
                loss = self.loss_fn(pred, labels_batch)
            return pred, loss

        def run_epoch(epoch, train):
            epochs.append(epoch)
            lrs.append(lr_scheduler.get_last_lr()[0])

            train_metric_chunks = { metric.name(): [] for metric in self.metrics }
            validation_metric_chunk = { }

            for chunk in self.training.batches():
                pred, loss = batch(chunk.features, chunk.labels) if train else testbatch(model, chunk.features, chunk.labels)
                for metric in self.metrics:
                    train_metric_chunks[metric.name()].append(metric.batch(model, chunk.features, chunk.labels, pred, loss.item()))

            pred, loss = testbatch(model, self.validation.features, self.validation.labels)
            for metric in self.metrics:
                validation_metric_chunk[metric.name()] = metric.batch(model, self.validation.features, self.validation.labels, pred, loss.item())

            for metric in self.metrics:
                train_metrics[metric.name()].append(metric.summarize(train_metric_chunks[metric.name()]))

            for metric in self.metrics:
                validation_metrics[metric.name()].append(metric.summarize([validation_metric_chunk[metric.name()]]))
            
            if (train):
                lr_scheduler.step()
        
        run_epoch(0, train=False)


        for epoch in tqdm(range(hyper.nepochs), desc="Epochs"):
            run_epoch(epoch + 1, train=True)

        return Result(hyper=hyper, model=model.to(device="cpu"), epochs=epochs, lrs=lrs, train_metrics=train_metrics, validation_metrics=validation_metrics)
