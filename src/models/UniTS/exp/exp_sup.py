from data.ATHOMEDataset import ATHOMEFeaturesUniTS
from data.PREATHOMEDataset import PREATHOMEFeaturesUniTS
from data_provider.data_factory import data_provider
from utils.pot import pot_eval
from utils.tools import adjust_learning_rate, cal_accuracy, adjustment
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.losses import mape_loss, mase_loss, smape_loss, FocalLoss
from utils.dataloader import BalancedDataLoaderIterator
from utils.layer_decay import param_groups_lrd
from utils.ddp import get_world_size, is_main_process, gather_tensors_from_all_gpus

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, cohen_kappa_score

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist

import os
import time
import warnings
import numpy as np
import yaml
import sys
import pandas as pd
from tqdm import tqdm

from data.ATHOME.constants import ANOMALIES as ATHOME_ANOMALIES
from data.PREATHOME.constants import ANOMALIES as PREATHOME_ANOMALIES

warnings.filterwarnings('ignore')


def apply_random_mask_for_imputation(x, patch_len, mask_rate):
    """
    Apply a random mask to the input tensor.

    Parameters:
    x (torch.Tensor): The input tensor with shape [B, T, N].
    patch_len (int): The length of each patch.
    mask_rate (float): The proportion of the tensor to be masked.

    Returns:
    torch.Tensor: The masked input tensor.
    torch.Tensor: The mask tensor.
    """
    B, T, N = x.shape
    num_keep = int((T // patch_len) * (1 - mask_rate))

    # Generate random noise and sort it
    noise = torch.rand(B, T // patch_len, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Select indices to keep
    ids_keep = ids_shuffle[:, :num_keep]
    mask = torch.zeros([B, T], device=x.device)
    mask[:, :num_keep] = 1
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # Expand the mask to the original shape
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_len).view(B, T)
    mask = mask.unsqueeze(-1).repeat(1, 1, N)

    # Apply the mask
    x_masked = x.masked_fill(mask == 0, 0)

    return x_masked, mask


def custom_print_decorator(func):
    def wrapper(*args, **kwargs):
        text = ' '.join(map(str, args))
        if 'file' not in kwargs or kwargs['file'] is None:
            sys.stdout.write(text + '\n')
        else:
            kwargs['file'].write(text + '\n')

        if 'folder' in kwargs and kwargs['folder']:
            with open(f'{kwargs["folder"]}/finetune_output.log', 'a') as log_file:
                log_file.write(text + '\n')
        if 'folder' in kwargs:
            del kwargs['folder']
        if 'file' in kwargs:
            del kwargs['file']

    return wrapper


# Replace print to save all print into log files
print = custom_print_decorator(print)


def read_task_data_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    task_dataset_config = config.get('task_dataset', {})
    # If dataset is ATHOME, take it and map to all the anomalies
    if 'ATHOME' in task_dataset_config:
        print("ATHOME dataset detected. Processing anomalies...")
        # Create a new task dataset config for each anomaly
        patient_id = task_dataset_config["ATHOME"]["patient_id"]
        new_task_dataset_config = {}
        for anomaly in ATHOME_ANOMALIES:
            # Create a new task dataset config for each anomaly
            new_task_dataset_config[f"ATHOME_{patient_id}_{anomaly}"] = task_dataset_config["ATHOME"].copy()
            new_task_dataset_config[f"ATHOME_{patient_id}_{anomaly}"]["dataset"] = f"ATHOME_{anomaly}"
            new_task_dataset_config[f"ATHOME_{patient_id}_{anomaly}"]["anomaly_type"] = anomaly
            # Get the features for the anomaly type
            data_set = ATHOMEFeaturesUniTS(
                win_size=16,
                flag="train",
                k=1,
                anomaly_type=anomaly,
                patient_id=patient_id
            )
            new_task_dataset_config[f"ATHOME_{patient_id}_{anomaly}"]["enc_in"] = len(data_set.main_features_idxs)
            new_task_dataset_config[f"ATHOME_{patient_id}_{anomaly}"]["main_features_idxs"] = data_set.main_features_idxs
            new_task_dataset_config[f"ATHOME_{patient_id}_{anomaly}"]["column_names"] = data_set.column_names
            new_task_dataset_config[f"ATHOME_{patient_id}_{anomaly}"]["scaler"] = data_set.scaler
            print(f"- {anomaly}: {len(data_set.main_features_idxs)} features")
        # Remove the original ATHOME dataset
        task_dataset_config.pop("ATHOME")
        # Add the new task dataset configs
        task_dataset_config.update(new_task_dataset_config)
    # If dataset is PREATHOME, take it and map to all the anomalies
    elif 'PREATHOME' in task_dataset_config:
        print("PREATHOME dataset detected. Processing anomalies...")
        # Create a new task dataset config for each anomaly
        new_task_dataset_config = {}
        for anomaly in PREATHOME_ANOMALIES:
            # Create a new task dataset config for each anomaly
            new_task_dataset_config[f"PREATHOME_{anomaly}"] = task_dataset_config["PREATHOME"].copy()
            new_task_dataset_config[f"PREATHOME_{anomaly}"]["dataset"] = f"PREATHOME_{anomaly}"
            new_task_dataset_config[f"PREATHOME_{anomaly}"]["anomaly_type"] = anomaly
            # Get the features for the anomaly type
            data_set = PREATHOMEFeaturesUniTS(
                win_size=16,
                flag="train",
                k=1,
                anomaly_type=anomaly,
            )
            new_task_dataset_config[f"PREATHOME_{anomaly}"]["enc_in"] = len(data_set.main_features_idxs)
            new_task_dataset_config[f"PREATHOME_{anomaly}"]["main_features_idxs"] = data_set.main_features_idxs
            new_task_dataset_config[f"PREATHOME_{anomaly}"]["scaler"] = data_set.scaler
            print(f"- {anomaly}: {len(data_set.main_features_idxs)} features")
        # Remove the original PREATHOME dataset
        task_dataset_config.pop("PREATHOME")
        # Add the new task dataset configs
        task_dataset_config.update(new_task_dataset_config)
    return task_dataset_config


def get_task_data_config_list(task_data_config, default_batch_size=None):
    task_data_config_list = []

    for task_name, task_config in task_data_config.items():
        task_config['max_batch'] = default_batch_size
        task_data_config_list.append([task_name, task_config])

    return task_data_config_list


def init_and_merge_datasets(data_loader_list):
    dataloader = BalancedDataLoaderIterator(data_loader_list)
    train_steps = dataloader.__len__()
    return dataloader, train_steps


class Exp_All_Task(object):
    def __init__(self, args, infer=False, task_data_config=None):
        super(Exp_All_Task, self).__init__()

        self.args = args
        self.loss_weights = None
        if not infer:
            self.ori_task_data_config = read_task_data_config(
                self.args.task_data_config_path
            )
            device_id = dist.get_rank() % torch.cuda.device_count()
        else:
            self.ori_task_data_config = task_data_config
            device_id = 0

        self.ori_task_data_config_list = get_task_data_config_list(
            self.ori_task_data_config, default_batch_size=self.args.batch_size)

        self.task_data_config = self.ori_task_data_config
        self.task_data_config_list = self.ori_task_data_config_list
        self.device_id = device_id
        print("device id", self.device_id)
        self.model = self._build_model(ddp=not infer)

    def get_loss_by_name(self, loss_name):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
        elif loss_name == 'CE':
            print("CE with loss weights:", self.loss_weights)
            return nn.CrossEntropyLoss(
                weight=self.loss_weights, reduction='mean'
            )
        elif loss_name == 'Focal':
            print("Using Focal Loss")
            return FocalLoss(alpha=1, gamma=2, reduction='mean')
        else:
            print("no loss function found!")
            exit()
    
    def _build_model(self, ddp=True):
        import importlib
        module = importlib.import_module("models."+self.args.model)
        model = module.Model(
            self.args, self.task_data_config_list).to(self.device_id)
        if ddp:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device_id],
                                                        find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=False)
        return model

    def _get_data(self, flag, test_anomaly_detection=False, train_ddp=True):
        this_task_data_config = self.task_data_config

        data_set_list = []
        data_loader_list = []

        for task_data_name, task_config in this_task_data_config.items():
            if test_anomaly_detection and (task_config['task_name'] == 'anomaly_detection' or task_config['task_name'] == 'contextual_anomaly_detection'):
                train_data_set, train_data_loader = data_provider(
                    self.args, task_config, flag='train', ddp=False)
                data_set, data_loader = data_provider(
                    self.args, task_config, flag, ddp=False)  # ddp false to avoid shuffle
                data_set_list.append([train_data_set, data_set])
                data_loader_list.append([train_data_loader, data_loader])
                print(task_data_name, len(data_set))
            elif task_config['task_name'] == 'classification':
                data_set, data_loader, loss_weights = data_provider(
                    self.args, task_config, flag, ddp=train_ddp)
                if self.loss_weights is not None:
                    self.loss_weights = loss_weights.to(self.device_id)
                data_set_list.append(data_set)
                data_loader_list.append(data_loader)
                print(task_data_name, len(data_set))
            else:
                data_set, data_loader = data_provider(
                    self.args, task_config, flag, ddp=train_ddp)
                data_set_list.append(data_set)
                data_loader_list.append(data_loader)
                print(task_data_name, len(data_set))
        return data_set_list, data_loader_list

    def _select_optimizer(self):
        eff_batch_size = self.args.batch_size * self.args.acc_it * get_world_size()
        real_learning_rate = self.args.learning_rate * eff_batch_size / 32
        self.real_learning_rate = real_learning_rate
        print("base lr: %.2e" % (self.args.learning_rate * 32 / eff_batch_size))
        print("actual lr: %.2e" % real_learning_rate)

        print("accumulate grad iterations: %d" % self.args.acc_it)
        print("effective batch size: %d" % eff_batch_size)
        if self.args.layer_decay is not None:
            print("layer decay: %.2f" % self.args.layer_decay)
            model_without_ddp = self.model.module
            param_groups = param_groups_lrd(model_without_ddp, self.args.weight_decay,
                                            no_weight_decay_list=[
                                                'prompts', 'mask_tokens', 'cls_tokens', 'category_tokens'],
                                            layer_decay=self.args.layer_decay
                                            )
            model_optim = optim.Adam(param_groups, lr=real_learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(
            ), lr=real_learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self, config_list):
        criterion_list = []
        for each_config in config_list:
            if 'loss' in each_config[1]:
                loss_name = each_config[1]['loss']
            else:
                if each_config[1]['task_name'] == 'anomaly_detection':
                    loss_name = 'MSE'
                elif each_config[1]['task_name'] == 'contextual_anomaly_detection':
                    loss_name = 'MSE'
                elif each_config[1]['task_name'] == 'classification':
                    loss_name = 'Focal'
                else:
                    print("this task has no loss now!", folder=self.path)
                    exit()
            criterion_list.append(self.get_loss_by_name(loss_name))

        return criterion_list

    def choose_training_parts(self, prompt_tune=False):
        for name, param in self.model.named_parameters():
            if prompt_tune:
                if 'prompt_token' in name or 'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name or 'cls_token' in name or 'category_token' in name:
                    param.requires_grad = True
                    print("trainable:", name)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

        # if not prompt_tune:
        #     print("all trainable.")

    def train(self, setting, train_ddp=True):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path

        # Load pretrained weights (Optional)
        if self.args.pretrained_weight is not None:
            if self.args.pretrained_weight == 'auto':
                pretrain_weight_path = os.path.join(
                    self.path, 'pretrain_checkpoint.pth')
            else:
                pretrain_weight_path = self.args.pretrained_weight
            print('loading pretrained model:',
                  pretrain_weight_path, folder=self.path)
            if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                state_dict = torch.load(
                    pretrain_weight_path, map_location='cpu')['student']
                ckpt = {}
                for k, v in state_dict.items():
                    if not ('cls_prompts' in k):
                        ckpt[k] = v
            else:
                ckpt = torch.load(pretrain_weight_path, map_location='cpu')
            msg = self.model.load_state_dict(ckpt, strict=False)
            print(msg, folder=self.path)

        # Data
        _, train_loader_list = self._get_data(flag='train', train_ddp=train_ddp)
        # Since some datasets do not have val set, we use test set and report the performance of last epoch instead of the best epoch.
        test_data_list, test_loader_list = self._get_data(
            flag='test', test_anomaly_detection=True)
        data_loader_cycle, train_steps = init_and_merge_datasets(
            train_loader_list)

        # Model param check
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print("Parameters number for all {} M".format(
            pytorch_total_params/1e6), folder=self.path)
        model_param = []
        for name, param in self.model.named_parameters():
            if ('prompts' in name and 'prompt2forecat' not in name) or 'prompt_token' in name or \
                    'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name or 'cls_token' in name or 'category_token' in name:
                print('skip this:', name)
            else:
                model_param.append(param.numel())
        model_total_params = sum(model_param)
        print("Parameters number for UniTS {} M".format(
            model_total_params/1e6), folder=self.path)

        # Optimizer and Criterion
        model_optim = self._select_optimizer()
        criterion_list = self._select_criterion(self.task_data_config_list)
        scaler = NativeScaler()

        # Set up batch size for each task
        if self.args.memory_check:
            self.memory_check(data_loader_cycle, criterion_list)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if get_world_size() > 1:
            dist.barrier()

        pbar = tqdm(range(self.args.train_epochs), desc='Epochs', position=0, leave=True)

        for epoch in pbar:
            adjust_learning_rate(model_optim, epoch,
                                 self.real_learning_rate, self.args)
            # Prompt learning
            if (epoch+1) <= self.args.prompt_tune_epoch:
                self.choose_training_parts(prompt_tune=True)
            else:
                self.choose_training_parts(prompt_tune=False)

            train_loss = self.train_one_epoch(
                model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler
            )
            pbar.set_postfix({'train_loss': train_loss})
            print(f"** Epoch {epoch+1} train loss: {train_loss}")

            # we report the results of last epoch and not find the best epoch based on val set, since some datasets do not have val set

            if (epoch + 1 % 5 == 0) or (epoch == (self.args.train_epochs+self.args.prompt_tune_epoch - 1)):
                self.test(
                    setting,
                    load_pretrain=False,
                    test_data_list=test_data_list,
                    test_loader_list=test_loader_list,
                    epoch=epoch
                )

        return self.model

    def train_one_epoch(self, model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler):
        current_device = torch.cuda.current_device()
        train_loss_set = []
        acc_it = self.args.acc_it
        max_norm = self.args.clip_grad

        self.model.train()
        epoch_time = time.time()
        self.model.zero_grad(set_to_none=True)
        loss_sum = 0

        pbar = tqdm(enumerate(data_loader_cycle), total=len(data_loader_cycle), desc='Batches', position=1, leave=False)

        for i, (sample_init, task_id) in pbar:

            task_name = self.task_data_config_list[task_id][1]['task_name']
            small_batch_size = self.task_data_config_list[task_id][1]['max_batch']
            if small_batch_size != self.args.batch_size:
                sample_list = self.split_batch(
                    sample_init, small_batch_size, task_name)
                len_sample_list = len(sample_list)
            else:
                sample_list = [sample_init]
                len_sample_list = 1

            for sample_idx in range(len_sample_list):
                sample = sample_list[sample_idx]
                if task_name == 'anomaly_detection':
                    loss = self.train_anomaly_detection(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0
                elif task_name == 'contextual_anomaly_detection':
                    loss = self.train_contextual_anomaly_detection(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0
                elif task_name == 'classification':
                    loss = self.train_classification(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0
                else:
                    print("Something wrong with the task name!", folder=self.path)
                    exit()

                loss /= acc_it
                loss /= len_sample_list
                if sample_idx < len_sample_list-1:
                    norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                        parameters=self.model.parameters(), create_graph=False, update_grad=False)
            loss_display = loss.item()*len_sample_list*acc_it
            train_loss_set.append(loss_display)

            norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                parameters=self.model.parameters(), create_graph=False, update_grad=((i + 1) % acc_it == 0))

            if (i+1) % acc_it == 0:
                model_optim.zero_grad()
            torch.cuda.synchronize()

            loss_sum += loss_display
            loss_sum_display = loss_sum

            del sample_init
            del sample_list
            if torch.cuda.memory_reserved(current_device) > 30*1e9:
                torch.cuda.empty_cache()

            # if is_main_process():
            #     print({
            #         'train_loss_'+self.task_data_config_list[task_id][0]: loss_display, 
            #         'norm_value': norm_value,
            #         'loss_sum': loss_sum_display/(i+1)
            #     })
            pbar.set_postfix({
                # 'norm_value': norm_value,
                'loss_sum': loss_sum_display/(i+1)
            })

            if (i + 1) % 100 == 0:
                if norm_value == None:
                    norm_value = -1
                # if is_main_process():
                #     print("\titers: {0}, epoch: {1} | norm: {2:.2f} | loss: {3:.7f} | current_loss: {4} |current task: {5}".format(
                #         i + 1, epoch + 1, norm_value, loss_sum_display/(i+1), loss_display, task_name, folder=self.path))

        # print("Epoch: {} cost time: {}".format(
        #     epoch + 1, time.time() - epoch_time), folder=self.path)
        train_loss = np.average(train_loss_set)
        torch.cuda.synchronize()
        if get_world_size() > 1:
            dist.barrier()

        return train_loss
    
    def train_classification(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']

        batch_x, label = this_batch

        batch_x = batch_x.float().to(self.device_id)
        padding_mask = None
        label = label.to(self.device_id)
        with torch.cuda.amp.autocast():
            outputs = model(batch_x, padding_mask, None,
                            None, task_id=task_id, task_name=task_name)
            if outputs.shape[0] == label.shape[0]:
                loss = criterion(outputs, label.long().squeeze(-1))
            else:
                label = label.repeat(outputs.shape[0]//label.shape[0], 1)
                loss = criterion(outputs, label.long().squeeze(-1))

        return loss

    def train_anomaly_detection(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        features = config['features']

        batch_x, _ = this_batch

        batch_x = batch_x.float().to(self.device_id)

        with torch.cuda.amp.autocast():
            outputs = model(batch_x, None, None,
                            None, task_id=task_id, task_name=task_name)
            f_dim = -1 if features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            loss = criterion(outputs, batch_x)

        return loss

    def train_contextual_anomaly_detection(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        features = config['features']
        seq_to_impute = config['seq_to_impute']
        main_features_idxs = config['main_features_idxs']

        batch_x, _ = this_batch
        batch_x = batch_x.float().to(self.device_id)

        # Mask out the future signals (last seq_to_impute samples) while keeping the context
        mask = torch.ones_like(batch_x).to(self.device_id)

        # Set to zero only the main features
        mask[:, -seq_to_impute:, main_features_idxs] = 0

        # Mask the context
        inp = batch_x * mask

        with torch.cuda.amp.autocast():
            outputs = model(inp, None, None,
                            None, task_id=task_id, mask=mask, task_name=task_name)
        f_dim = -1 if features == 'MS' else 0
        outputs = outputs[:, :, f_dim:]

        # Get the imputed data
        gen_data = outputs[:, -seq_to_impute:, main_features_idxs]
        true_data = batch_x[:, -seq_to_impute:, main_features_idxs]

        # Get nan values from true data and set to zero on both true and gen data
        nan_mask = torch.isnan(true_data)
        true_data[nan_mask] = 0
        gen_data[nan_mask] = 0
        
        # Compute the loss
        loss = criterion(gen_data, true_data)

        return loss

    def test(self, setting, load_pretrain=False, test_data_list=None, test_loader_list=None, epoch=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path) and is_main_process():
            os.makedirs(self.path)
        if test_data_list is None or test_loader_list is None:
            test_data_list, test_loader_list = self._get_data(
                flag='test', test_anomaly_detection=True)
        if load_pretrain:
            if os.path.exists(self.args.pretrained_weight):
                pretrain_weight_path = self.args.pretrained_weight
                print('loading pretrained model:',
                      pretrain_weight_path, folder=self.path)
                if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                    state_dict = torch.load(
                        pretrain_weight_path, map_location='cpu')['student']
                    ckpt = {}
                    for k, v in state_dict.items():
                        if not ('cls_prompts' in k):
                            ckpt[k] = v
                else:
                    ckpt = torch.load(pretrain_weight_path, map_location='cpu')
                msg = self.model.load_state_dict(ckpt, strict=False)
                print(msg)
            else:
                print("no ckpt found!")
                exit()

        total_dict = {}
        avg_anomaly_f_score = []
        avg_classification_acc = []
        for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            data_task_name = self.task_data_config_list[task_id][0]
            if task_name == 'anomaly_detection':
                f_score, precision, recall, accuracy = self.test_anomaly_detection(
                    setting, test_data, test_loader, data_task_name, task_id, epoch=epoch
                )
                # f_score, precision, recall, accuracy = self.test_anomaly_detection(
                #     setting, test_data, test_loader, data_task_name, task_id, epoch=epoch, ignore=True
                # )
                total_dict[data_task_name] = {
                    'f_score': f_score, 'precision': precision, 'recall': recall
                }
                avg_anomaly_f_score.append(f_score)
            elif task_name == 'contextual_anomaly_detection':
                f_score, precision, recall, accuracy = self.test_contextual_anomaly_detection(
                    self.task_data_config_list[task_id][1], setting, test_data, test_loader, data_task_name, task_id, epoch=epoch
                )
                total_dict[data_task_name] = {
                    'f_score': f_score, 'precision': precision, 'recall': recall
                }
                avg_anomaly_f_score.append(f_score)
            elif task_name == 'classification':
                acc = self.test_classification(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {'acc': acc}
                avg_classification_acc.append(acc)
            else:
                print("Something wrong with the task name!", folder=self.path)
                exit()

        avg_anomaly_f_score = np.average(avg_anomaly_f_score)
    
    def test_classification(self, setting, test_data, test_loader, data_task_name, task_id):
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device_id)
                padding_mask = None
                label = label.to(self.device_id)

                outputs = self.model(
                    batch_x, padding_mask, None, None, task_id=task_id, task_name='classification')
                outputs = torch.nn.functional.softmax(outputs)

                predictions = torch.argmax(outputs, dim=1)
                preds.append(predictions.detach())
                trues.append(label)

        preds = gather_tensors_from_all_gpus(
            preds, self.device_id, to_numpy=False)
        trues = gather_tensors_from_all_gpus(
            trues, self.device_id, to_numpy=False)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)

        predictions = preds.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        
        # Calculate overall accuracy
        accuracy = cal_accuracy(predictions, trues)
        
        # Get number of classes from predictions
        num_classes = max(max(predictions), max(trues)) + 1
        
        # Compute comprehensive metrics
        conf_matrix = confusion_matrix(trues, predictions, labels=range(int(num_classes)))
        
        # Calculate per-class accuracy/recall
        per_class_recall = recall_score(trues, predictions, labels=range(int(num_classes)), average=None)
        per_class_precision = precision_score(trues, predictions, labels=range(int(num_classes)), average=None)
        per_class_f1 = f1_score(trues, predictions, labels=range(int(num_classes)), average=None)
        
        # Calculate weighted/macro metrics
        weighted_precision = precision_score(trues, predictions, average='weighted')
        weighted_recall = recall_score(trues, predictions, average='weighted')
        weighted_f1 = f1_score(trues, predictions, average='weighted')
        macro_precision = precision_score(trues, predictions, average='macro')
        macro_recall = recall_score(trues, predictions, average='macro')
        macro_f1 = f1_score(trues, predictions, average='macro')
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(trues, predictions)
        
        # Get detailed classification report
        class_report = classification_report(trues, predictions, labels=range(int(num_classes)), output_dict=True)
        
        # Format results
        print(f'Data task: {data_task_name} - Overall accuracy: {accuracy:.4f}', folder=self.path)
        print(f'Cohen\'s Kappa: {kappa:.4f}', folder=self.path)
        print('\nPer-class metrics:', folder=self.path)
        
        for i in range(int(num_classes)):
            print(f'Class {i}: Precision={per_class_precision[i]:.4f}, Recall={per_class_recall[i]:.4f}, F1={per_class_f1[i]:.4f}', folder=self.path)
        
        print('\nConfusion Matrix:', folder=self.path)
        print(conf_matrix, folder=self.path)
        
        # If we're at a specific epoch, include that in the metrics
        metrics = {
            'accuracy': accuracy,
            'kappa': kappa,
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # Save metrics to a readable format
        if is_main_process():
            os.makedirs(f"{self.path}/metrics", exist_ok=True)
            with open(f"{self.path}/metrics/{data_task_name}_classification_metrics.txt", 'a') as f:
                f.write(f"\n--- Metrics for {data_task_name} ---\n")
                f.write(f"Overall accuracy: {accuracy:.4f}\n")
                f.write(f"Cohen's Kappa: {kappa:.4f}\n\n")
                f.write("Per-class metrics:\n")
                for i in range(int(num_classes)):
                    f.write(f"Class {i}: Precision={per_class_precision[i]:.4f}, Recall={per_class_recall[i]:.4f}, F1={per_class_f1[i]:.4f}\n")
                f.write(f"\nWeighted - Precision: {weighted_precision:.4f}, Recall: {weighted_recall:.4f}, F1: {weighted_f1:.4f}\n")
                f.write(f"Macro    - Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}\n")
                f.write("\nConfusion Matrix:\n")
                f.write(str(conf_matrix))
                f.write("\n\n")
    
        del predictions
        del trues
        torch.cuda.empty_cache()
        return accuracy

    def test_anomaly_detection(self, setting, test_data, test_loader_set, data_task_name, task_id, ar=None, epoch=None, ignore=False):
        train_loader, test_loader = test_loader_set
        attens_energy = []
        anomaly_criterion = nn.MSELoss(reduce=False)

        if ignore:
            epoch += 1

        self.model.eval()
        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Testing (compute energy on train set)', position=1, leave=False):
                batch_x = batch_x.float().to(self.device_id)
                # reconstruction
                outputs = self.model(
                    batch_x, None, None, None, task_id=task_id, task_name='anomaly_detection'
                )
                # criterion
                score = anomaly_criterion(batch_x, outputs)
                score = test_data[0].features_per_anomaly(score, ignore=ignore) # Get only relevant features for the anomaly
                score = torch.mean(score, dim=-1)
                score = score.detach().cpu()
                attens_energy.append(score)

            # If distributed, gather the tensors from all gpus
            if get_world_size() > 1:
                attens_energy = gather_tensors_from_all_gpus(
                    attens_energy, self.device_id, to_numpy=True)
        train_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        reconstructed = []
        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing (compute energy on test set)', position=1, leave=False):
            batch_x = batch_x.float().to(self.device_id)
            # reconstruction
            outputs = self.model(batch_x, None, None, None,
                                 task_id=task_id, task_name='anomaly_detection')
            reconstructed.append(outputs.detach().cpu().numpy())
            
            # criterion
            score = anomaly_criterion(batch_x, outputs)
            score = test_data[0].features_per_anomaly(score, ignore=ignore)
            score = torch.mean(score, dim=-1)
            score = score.detach().cpu()
            attens_energy.append(score)
            test_labels.append(batch_y)

        # If distributed, gather the tensors from all gpus
        if get_world_size() > 1:
            attens_energy = gather_tensors_from_all_gpus(
                attens_energy, self.device_id, to_numpy=True)
        test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        # AUC metrics
        auc_roc = roc_auc_score(gt, test_energy)
        auc_pr = average_precision_score(gt, test_energy)

        print({
            "AUC_ROC": auc_roc,
            "AUC_PR": auc_pr,
            "epoch": epoch
        })

        # Test for different anomaly thresholds
        for anomaly_ratio in tqdm(range(1, int(self.args.anomaly_ratio)), desc='Testing (different anomaly ratios)', position=1, leave=False):
            threshold = np.percentile(combined_energy, 100 - anomaly_ratio)

            # (3) evaluation on the test set
            pred = (test_energy > threshold).astype(int)

            # print("pred:   ", pred.shape)
            # print("gt:     ", gt.shape)

            # (4) detection adjustment
            if not ar == anomaly_ratio:
                gt, pred = adjustment(gt, pred)

            pred = np.array(pred)
            gt = np.array(gt)

            tp = np.sum((gt == 1) & (pred == 1))
            tn = np.sum((gt == 0) & (pred == 0))
            fp = np.sum((gt == 0) & (pred == 1))
            fn = np.sum((gt == 1) & (pred == 0))

            # Metrics
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_score = 2 * precision * recall / (precision + recall)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            MIoU = 0.5 * (tp / (tp + fp + fn) + tn / (tn + fn + fp))
            fnr = fn / (fn + tp)
            tnr = tn / (tn + fp)
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)

            # print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            #     accuracy, precision,
            #     recall, f_score))

            print({
                f"f1_score_ar{anomaly_ratio}": f_score,
                f"precision_ar{anomaly_ratio}": precision,
                f"recall_ar{anomaly_ratio}": recall,
                f"accuracy_ar{anomaly_ratio}": accuracy,
                f"MIoU_ar{anomaly_ratio}": MIoU,

                f"threshold_ar{anomaly_ratio}": threshold,

                f"TP_ar{anomaly_ratio}": tp,
                f"TN_ar{anomaly_ratio}": tn,
                f"FP_ar{anomaly_ratio}": fp,
                f"FN_ar{anomaly_ratio}": fn,

                f"fnr_ar{anomaly_ratio}": fnr,
                f"fpr_ar{anomaly_ratio}": fpr,
                f"tpr_ar{anomaly_ratio}": tpr,
                f"tnr_ar{anomaly_ratio}": tnr,

                "epoch": epoch
            })

            if ar == anomaly_ratio:
                print("WARNING: inference mode is on, only return the results for anomaly ratio", ar)
                reconstructed = np.concatenate(reconstructed, axis=0)
                return f_score, precision, recall, accuracy, gt, pred, reconstructed, test_energy, threshold

        return f_score, precision, recall, accuracy

    def test_contextual_anomaly_detection(self, config, setting, test_data, test_loader_set, data_task_name, task_id, epoch=None):
        train_loader, test_loader = test_loader_set
        attens_energy = []
        anomaly_criterion = nn.MSELoss(reduce=False)
        seq_to_impute = config['seq_to_impute']
        main_features_idxs = config['main_features_idxs']

        self.model.eval()
        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Testing (compute energy on train set)', position=1, leave=False):
                batch_x = batch_x.float().to(self.device_id)

                # Mask out the future signals (last seq_to_impute samples) while keeping the context
                mask = torch.ones_like(batch_x).to(self.device_id)

                # Set to zero only the main features
                mask[:, -seq_to_impute:, main_features_idxs] = 0

                # apply the mask
                inp = batch_x * mask

                # reconstruction
                outputs = self.model(
                    inp, None, None, None, task_id=task_id, mask=mask, task_name='contextual_anomaly_detection')

                # Get the imputed data
                gen_data = outputs[:, -seq_to_impute:, main_features_idxs]
                true_data = batch_x[:, -seq_to_impute:, main_features_idxs]

                # Get nan values from true data and set to zero on both true and gen data
                nan_mask = torch.isnan(true_data)
                true_data[nan_mask] = 0
                gen_data[nan_mask] = 0

                # criterion
                score = torch.mean(anomaly_criterion(gen_data, true_data), dim=-1)
                score = score.detach().cpu()
                attens_energy.append(score)
        attens_energy = gather_tensors_from_all_gpus(
            attens_energy, self.device_id, to_numpy=True)
        train_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        test_datap = []

        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing (compute energy on test set)', position=1, leave=False):
            batch_x = batch_x.float().to(self.device_id)

            # Mask out the future signals (last seq_to_impute samples) while keeping the context
            mask = torch.ones_like(batch_x).to(self.device_id)

            # Set to zero only the main features
            mask[:, -seq_to_impute:, main_features_idxs] = 0

            # apply the mask
            inp = batch_x * mask

            # reconstruction
            outputs = self.model(
                inp, None, None, None, task_id=task_id, mask=mask, task_name='contextual_anomaly_detection')

            # Get the imputed data
            gen_data = outputs[:, -seq_to_impute:, main_features_idxs]
            true_data = batch_x[:, -seq_to_impute:, main_features_idxs]

            # Get nan values from true data and set to zero on both true and gen data
            nan_mask = torch.isnan(true_data)
            true_data[nan_mask] = 0
            gen_data[nan_mask] = 0

            # criterion
            score = torch.mean(anomaly_criterion(gen_data, true_data), dim=-1)
            score = score.detach().cpu()
            attens_energy.append(score)
            test_labels.append(batch_y)
            datap = batch_x.detach().cpu().numpy()
            test_datap.append(datap[:, -seq_to_impute:, :])

        attens_energy = gather_tensors_from_all_gpus(
            attens_energy, self.device_id, to_numpy=True)
        test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        # Test using POT
        anomaly_type = test_data[1].anomaly_type
        dataset_name = config['data']
        print(f"Testing using POT for {anomaly_type} in {dataset_name}")
        scores, pred = pot_eval(train_energy, test_energy, gt, dataset_name, anomaly_type)
        print({
            "f1": scores["f1"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "accuracy": scores["accuracy"],
            "TP": scores["TP"],
            "TN": scores["TN"],
            "FP": scores["FP"],
            "FN": scores["FN"],
            "ROC/AUC": scores["ROC/AUC"],
            "PR/AUC": scores["PR/AUC"],
            "threshold": scores["threshold"],
            "fnr": scores["fnr"],
            "fpr": scores["fpr"],
            "epoch": epoch
        })

        # Save the results
        print(f"Saving results for {data_task_name} in {self.path}/results")
        fist_date: pd.Timestamp = test_data[1].first_date
        os.makedirs(f"{self.path}/results/{data_task_name}", exist_ok=True)
        datap = np.concatenate(test_datap, axis=0)

        # Inverse transform the data
        for i in range(datap.shape[1]):
            datap[:, i] = test_data[1].scaler.inverse_transform(datap[:, i])

        # Save the data
        np.save(f"{self.path}/results/{data_task_name}/data.npy", datap)
        np.save(f"{self.path}/results/{data_task_name}/pred.npy", pred)
        np.save(f"{self.path}/results/{data_task_name}/energy.npy", test_energy)
        np.save(f"{self.path}/results/{data_task_name}/gt.npy", gt)
        np.save(f"{self.path}/results/{data_task_name}/threshold.npy", scores["threshold"])

        # Save the first date
        with open(f"{self.path}/results/{data_task_name}/first_date.txt", 'w') as f:
            f.write(fist_date.strftime("%Y-%m-%d %H:%M:%S"))

        return scores["f1"], scores["precision"], scores["recall"], scores["accuracy"]

    def split_batch(self, batch, small_batch_size, task_name):
        def split_tensor(tensor, size):
            return [tensor[i:min(i + size, tensor.size(0))] for i in range(0, tensor.size(0), size)]
        if task_name == 'anomaly_detection':
            batch_x, batch_y = batch
            split_batch_x = split_tensor(batch_x, small_batch_size)
            split_batch_y = split_tensor(batch_y, small_batch_size)
            return list(zip(split_batch_x, split_batch_y))
        elif task_name == 'contextual_anomaly_detection':
            batch_x, batch_y = batch
            split_batch_x = split_tensor(batch_x, small_batch_size)
            split_batch_y = split_tensor(batch_y, small_batch_size)
            return list(zip(split_batch_x, split_batch_y))

    def save_model(self, setting):
        epoch = self.args.train_epochs+self.args.prompt_tune_epoch
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path
        torch.save(
            self.model.state_dict(),
            os.path.join(path, 'checkpoint_%d.pth' % epoch)
        )
        print('model saved in', path)
        return path
