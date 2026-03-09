import math
import collections
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from solve.dataset import DataSet, DataLoader


Hyper = collections.namedtuple('Hyper', 'name seed model optimizer optimizer_kwargs lr_schedule preprocess', defaults=(False,))
Result = collections.namedtuple('Result', 'hyper model epochs train_metrics validation_metrics')


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


def get_schedule(lr_points):
    lr_schedule = []
    epoch = 0
    for item in lr_points:
        for i in range(item[0]):
            epoch += 1
            lr_schedule.append((epoch, item[1]))
    return lr_schedule

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
        optimizer = hyper.optimizer(model.parameters(), lr=hyper.lr_schedule[0][1], **hyper.optimizer_kwargs)

        epochs = []
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

        def run_epoch(epoch, lr, train):
            epochs.append(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

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
        
        run_epoch(0, hyper.lr_schedule[0][1], train=False)

        lr_schedule = get_schedule(hyper.lr_schedule)
        
        for (epoch, lr) in tqdm(lr_schedule, desc="Epochs"):
            run_epoch(epoch + 1, lr, train=True)

        return Result(hyper=hyper, model=model, epochs=epochs, train_metrics=train_metrics, validation_metrics=validation_metrics)
