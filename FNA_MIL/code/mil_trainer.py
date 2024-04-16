from utils import AverageMeter, save_model, plot_roc_curve, plot_pr_curve
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch
import os
import sys
import json
import copy


class ClassificationTrainer:

    def __init__(self, model, optimizer, train_loader, test_loader, test_interval=5, batch_size=8,
                 epochs=50, patience=10, negative_control=None, lq_loss=None, scheduler=None, schedule_type=None,
                 run_name=None, test_data_index=-1):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_type = schedule_type
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_interval = test_interval
        self.batch_size = batch_size
        self.num_train = len(train_loader)
        self.num_test = len(test_loader) if test_loader is not None else 0
        self.epochs = epochs
        self.curr_epoch = 0
        self.use_gpu = next(self.model.parameters()).is_cuda
        # TODO: this assumes a global learning rate
        self.lr = self.optimizer.param_groups[0]['lr']
        self.patience = patience
        self.beta = 0.95
        if lq_loss is None:
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            self.criterion = lambda y_pred, y_true: self.lq_loss(lq_loss, y_pred, y_true)
        self.run_name = run_name
        self.test_data_index = test_data_index

    def train(self):
        print(f"\n[*] Train on {self.num_train} samples, test on {self.num_test} samples")
        best_test_acc = 0
        epochs_since_best = 0

        for epoch in range(self.epochs):
            self.curr_epoch = epoch
            print(f'\nEpoch {epoch+1}/{self.epochs} -- lr = {self.lr}')
            train_loss, train_acc = self.run_one_epoch(training=True)
            test_inference_results = self.get_inference_results(self.test_loader)
            test_acc = self.get_acc(test_inference_results)
            if self.test_data_index == -1:
                test_auc = self.get_auc(test_inference_results)
                test_aps = self.get_aps(test_inference_results)
                test_precision = self.get_precision(test_inference_results)
                test_recall = self.get_recall(test_inference_results)
            else:
                test_auc = -1
                test_aps = -1
            msg = f'train loss {train_loss:.3f} train acc {train_acc:.3f} -- test acc {test_acc:.3f} test pre {test_precision:.3f} test rec {test_recall:.3f} test auc {test_auc:.3f} test aps {test_aps:.3f}'
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'test_auc': test_auc,
                'test_aps': test_aps,
                'test_pre': test_precision,
                'test_rec': test_recall
            }
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                save_model(self.model, self.run_name)
                test_inference_results_json = self.tensor_to_json(test_inference_results)
                test_inference_results_json_write_path = os.path.join('/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/models', self.run_name, 'test_inference.json')
                json.dump(test_inference_results_json, open(test_inference_results_json_write_path, 'w'))
                labels = [values['label'] for values in test_inference_results.values()]
                predictions = [np.median(values['predictions'].cpu().detach().numpy()) for values in test_inference_results.values()]
                if self.test_data_index == -1:
                    plot_roc_curve(ground_truth=labels, scores=predictions, model_id=self.run_name)
                    plot_pr_curve(ground_truth=labels, scores=predictions, test_aps=test_aps, model_id=self.run_name)
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            if epochs_since_best == 0:
                msg += '[*]'
                results_write_path = os.path.join('/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/models', self.run_name, 'results')
                with open(results_write_path, 'w') as out:
                    print(msg, file=out)
                    self.print_auc_order(test_inference_results, out)
                    

            for param_group in self.optimizer.param_groups:
                curr_lr = param_group['lr']
                break
            metrics['curr_lr'] = curr_lr
            print(msg)
            self.print_auc_order(test_inference_results)

            if self.scheduler is not None:
                self.scheduler.step()
                self.lr = curr_lr
            # if epochs_since_best > self.patience:
            #     epochs_since_best = 0
            #     self.lr = self.lr / np.sqrt(10)
            #     for param_group in self.optimizer.param_groups:
            #         param_group['lr'] = self.lr

    def tensor_to_json(self, tensor):
        tensor_to_convert = copy.deepcopy(tensor)
        for k in tensor_to_convert.keys():
            tensor_to_convert[k]['predictions'] = tensor_to_convert[k]['predictions'].cpu().detach().numpy().tolist()
        return tensor_to_convert
    
    @staticmethod
    def avg_soft(tensor):
        mil = torch.mean(tensor, dim=0, keepdim=True)
        mil_soft = torch.nn.functional.softmax(mil, dim=-1)
        return mil_soft


    def run_one_epoch(self, training, testing=False):
        losses = AverageMeter()
        accs = AverageMeter()
        if training:
            # using train set, doing updates
            if testing:
                raise RuntimeError()
            amnt = self.num_train
            loader = self.train_loader
            self.model.train()
        elif testing:
            # evaling test set
            amnt = self.num_test
            loader = self.test_loader
            self.model.eval()
        beta = self.beta ** self.curr_epoch
        simple_loss = torch.nn.CrossEntropyLoss()
        if testing:
            accum_amnt = 1
            max_size = 1
        else:
            accum_amnt = 8
            max_size = 4
        accum_counter = 0
        curr_shape = None
        batch_data = []
        with tqdm(total=amnt * self.batch_size) as pbar:
            for i, data in enumerate(loader):
                batch_data.append(data)
                accum_counter += 1
                if accum_counter < accum_amnt:
                    continue
                accum_counter = 0
                # now stack if possible
                curr_shape = None
                x_batched = []
                y_batched = []
                for x, y, _ in batch_data:
                    if x.shape != curr_shape or len(x_batched) == 0 or len(x_batched[-1]) >= max_size:
                        curr_shape = x.shape
                        x_batched.append([])
                        y_batched.append([])
                    x_batched[-1].append(x)
                    y_batched[-1].append(y)
                x_batched = [torch.stack(xb).squeeze(1) for xb in x_batched]
                y_batched = [torch.stack(yb).squeeze(1) for yb in y_batched]
                total_loss = 0
                total_acc = 0
                if training:
                    self.optimizer.zero_grad()
                for x, y in zip(x_batched, y_batched):
                    if self.use_gpu:
                        x, y, = x.cuda(), y.cuda()
                    if training:
                        # output is going to be a MIL output and a bunch of SIL outputs
                        mil_out, sil_out = self.model(x)
                        simple_mil_loss = (simple_loss(mil_out, y) / accum_amnt)*x.shape[0]
                        simple_mil_loss.backward()
                        total_loss += float(simple_mil_loss.detach().cpu())
                    else:
                        with torch.no_grad():
                            mil_out, sil_out = self.model(x)
                            simple_mil_loss = (simple_loss(mil_out, y) / accum_amnt) * x.shape[0]
                            total_loss += float(simple_mil_loss.detach().cpu())
                    _, preds = torch.max(mil_out, 1)
                    with torch.no_grad():
                        total_acc += float(torch.sum(preds == y.data).float().detach())
                if training:
                    self.optimizer.step()
                
                batch_data = []
                acc = total_acc / accum_amnt

                losses.update(total_loss)
                accs.update(acc)

                pbar.set_description(f" - loss: {losses.avg:.3f} acc {accs.avg:.3f}")
                pbar.update(accum_amnt)

        return losses.avg, accs.avg

    def get_inference_results(self, loader):
        inference_results = {}
        self.model.eval()
        rand = 0
        with torch.no_grad():
            for images, labels, orders in tqdm(loader):
                images = images.cuda()
                results_mil, results_sil = self.model(images)
                rand += 1
                preds = F.softmax(results_mil, dim=1)[0, 1] # TODO: why is it different? not torch.max(mil_out, 1)?
                inference_results[str(rand)] = {
                    'predictions': preds,
                    'label': int(labels[0]),
                    'order': orders[0]
                }

        return inference_results

    @staticmethod
    def get_auc(inference_results):
        labels = [values['label'] for values in inference_results.values()]
        predictions = [np.median(values['predictions'].cpu().detach().numpy()) for values in inference_results.values()]
        auc = roc_auc_score(labels, predictions)
        return auc

    @staticmethod
    def print_auc_order(inference_results, out=None):
        acc_dict = {}
        for values in inference_results.values():
            order = values['order'][:values['order'].find('_',10)]
            if order not in acc_dict.keys():
                acc_dict[order] = {
                    'label': values['label'],
                    'prediction': [np.round(np.median(values['predictions'].cpu().detach().numpy()))]
                }
            else:
                acc_dict[order]['prediction'].append(np.round(np.median(values['predictions'].cpu().detach().numpy())))
        
        for key in acc_dict:
            prediction = int(np.round(np.median(acc_dict[key]['prediction'])))
            label = acc_dict[key]['label']
            result = 'O' if prediction == label else 'X'
            if out is None:
                print(f'{key} pred {prediction} label {label} result {result}')
            else:
                print(f'{key} pred {prediction} label {label} result {result}', file=out)

    @staticmethod
    def get_aps(inference_results):
        labels = [values['label'] for values in inference_results.values()]
        predictions = [np.median(values['predictions'].cpu().detach().numpy()) for values in inference_results.values()]
        aps = average_precision_score(labels, predictions)
        return aps
    
    @staticmethod
    def get_precision(inference_results):
        labels = [values['label'] for values in inference_results.values()]
        predictions = [np.median(values['predictions'].cpu().detach().numpy()) for values in inference_results.values()]
        predictions = np.array(predictions)
        predictions[predictions>=0.5] = 1
        predictions[predictions<0.5] = 0
        precision = precision_score(labels, predictions)
        return precision
    
    @staticmethod
    def get_recall(inference_results):
        labels = [values['label'] for values in inference_results.values()]
        predictions = [np.median(values['predictions'].cpu().detach().numpy()) for values in inference_results.values()]
        predictions = np.array(predictions)
        predictions[predictions>=0.5] = 1
        predictions[predictions<0.5] = 0
        recall = recall_score(labels, predictions)
        return recall
    
    @staticmethod
    def get_acc(inference_results):
        labels = [values['label'] for values in inference_results.values()]
        predictions = [np.median(values['predictions'].cpu().detach().numpy()) for values in inference_results.values()]
        acc = np.sum(np.round(predictions) == labels) / len(labels)
        return acc