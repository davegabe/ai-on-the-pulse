from data.PREATHOMEDataset import PREATHOMEFeaturesUniTS  # nopep8
from data.ATHOMEDataset import ATHOMEFeaturesUniTS  # nopep8
from data.ATHOMESleepDataset import ATHOMESleepUniTS  # nopep8

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler


def random_subset(dataset, pct, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=generator)
    return Subset(dataset, idx[:int(len(dataset) * pct)].long().numpy())


def data_provider(args, config, flag, ddp=False):  # args,
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if 'anomaly_detection' in config['task_name']:  # working on one gpu
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        # freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        # freq = args.freq

    # Get the window size
    win_size = args.win_size
    
    if 'anomaly_detection' in config['task_name']:
        drop_last = False

        # In case of contextual anomaly detection
        if args.confounding:
            # we need to set the window size to the patch length
            win_size = args.patch_len

        # Load the data
        print(f"Loading data for {config['data']}")
        if 'ATHOME' == config['data']:
            data_set = ATHOMEFeaturesUniTS(
                win_size=win_size,
                flag=flag,
                k=args.k,
                anomaly_type=config['anomaly_type'],
                patient_id=config["patient_id"]
            )
            config['seq_to_impute'] = data_set.seq_to_impute
            config['main_features_idxs'] = data_set.main_features_idxs
        elif 'PREATHOME' == config['data']:
            data_set = PREATHOMEFeaturesUniTS(
                win_size=win_size,
                flag=flag,
                k=args.k,
                anomaly_type=config['anomaly_type'],
            )
            config['seq_to_impute'] = data_set.seq_to_impute
            config['main_features_idxs'] = data_set.main_features_idxs
        else:
            raise ValueError(f"Data {config['data']} not supported")

        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        return data_set, data_loader
    
    elif 'classification' in config['task_name']:
        # Sleep phase classification task
        print(f"Loading classification data for {config['data']}")
        
        if 'ATHOME_SLEEP' == config['data']:
            # Check if binary classification is requested
            is_binary = config.get('binary_classification', False)
            
            data_set = ATHOMESleepUniTS(
                win_size=win_size,
                flag=flag,
                k=args.k,
                patient_id=-1, #TODO: decide if use every patient or not
                binary=is_binary  # Pass binary flag from config
            )
            
            # Only calculate class weights for multi-class classification, not for binary
            loss_weights = None if is_binary else data_set.get_class_weights()
            
        else:
            raise ValueError(f"Data {config['data']} not supported for sleep classification")
            
        if args.subsample_pct is not None and flag == "train":
            data_set = random_subset(
                data_set, args.subsample_pct, args.fix_seed)
                
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)
        
        if config['task_name'] == 'classification':
            # Return the class weights for the classification task (None for binary)
            return data_set, data_loader, loss_weights
        else:
            # For other tasks, return the data loader only
            return data_set, data_loader
