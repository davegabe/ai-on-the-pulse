# Import dataloader from main directory
import random
import sys  # nopep8
import os  # nopep8
base_path = os.getenv("BASE_PATH")  # nopep8
sys.path.append(os.path.join(base_path, "src/data"))  # nopep8
from FeaturesDataset import FeaturesTranAD  # nopep8

import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import color, plot_accuracies, adjustment
from src.dlutils import ComputeLoss
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from time import time
import wandb
from multiprocessing import Pool

device = "cpu"
VERSION = os.getenv("VERSION", "none")
WANDB_USER = os.getenv("WANDB_USER", "none")
FIXED_SEED = 444


def convert_to_windows(data, model):
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(
            w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))
    return torch.stack(windows)


def load_dataset(dataset: str, k: int, source: str):
    train_set = FeaturesTranAD(
        dataset=dataset, flag="train", k=k, source=source)
    test_set = FeaturesTranAD(dataset=dataset, flag="test", k=k, source=source)
    train_loader = DataLoader(train_set, batch_size=len(train_set))
    test_loader = DataLoader(test_set, batch_size=len(test_set))
    labels = test_set.labels
    return train_loader, test_loader, labels


def save_model(model, optimizer, scheduler, epoch, accuracy_list):
    folder = f'checkpoints/{args.model}_{args.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list}, file_path)


def load_model(modelname, dims, lr):
    import src.models
    model_class = getattr(src.models, modelname)
    model = model_class(dims).double().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{args.model}_{args.dataset}/model.ckpt'
    if os.path.exists(fname) and (not args.retrain or args.test):
        print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list


def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    l = nn.MSELoss(reduction='mean' if training else 'none')
    feats = dataO.shape[1]
    if 'DAGMM' in model.name:
        l = nn.MSELoss(reduction='none')
        compute = ComputeLoss(model, 0.1, 0.005, device, model.n_gmm)
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        l2s = []
        if training:
            for d in data:
                d = d.double().to(device)
                _, x_hat, z, gamma = model(d)
                l1, l2 = l(x_hat, d), l(gamma, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1) + torch.mean(l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(
                f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s = []
            for d in data:
                d = d.double().to(device)
                _, x_hat, _, _ = model(d)
                ae1s.append(x_hat)
            ae1s = torch.stack(ae1s)
            y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(ae1s, data)[:, data.shape[1] -
                                 feats:data.shape[1]].view(-1, feats)
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    if 'Attention' in model.name:
        l = nn.MSELoss(reduction='none')
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        res = []
        if training:
            for d in data:
                d = d.double().to(device)
                ae, ats = model(d)
                # res.append(torch.mean(ats, axis=0).view(-1))
                l1 = l(ae, d)
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # res = torch.stack(res); np.save('ascores.npy', res.detach().numpy())
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            ae1s, y_pred = [], []
            for d in data:
                d = d.double().to(device)
                ae1 = model(d)
                y_pred.append(ae1[-1])
                ae1s.append(ae1)
            ae1s, y_pred = torch.stack(ae1s), torch.stack(y_pred)
            loss = torch.mean(l(ae1s, data), axis=1)
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    elif 'OmniAnomaly' in model.name:
        if training:
            mses, klds = [], []
            for i, d in enumerate(data):
                d = d.type(torch.float64).to(device)
                y_pred, mu, logvar, hidden = model(d, hidden if i else None)
                MSE = l(y_pred, d)
                KLD = -0.5 * torch.sum(1 + logvar -
                                       mu.pow(2) - logvar.exp(), dim=0)
                loss = MSE + model.beta * KLD
                mses.append(torch.mean(MSE).item())
                klds.append(model.beta * torch.mean(KLD).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(
                f'Epoch {epoch},\tMSE = {np.mean(mses)},\tKLD = {np.mean(klds)}')
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            y_preds = []
            for i, d in enumerate(data):
                d = d.type(torch.float64).to(device)
                y_pred, _, _, hidden = model(d, hidden if i else None)
                y_preds.append(y_pred)
            y_pred = torch.stack(y_preds)
            MSE = l(y_pred, data)
            return MSE.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    elif 'USAD' in model.name:
        l = nn.MSELoss(reduction='none')
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d in data:
                d = d.double().to(device)
                ae1s, ae2s, ae2ae1s = model(d)
                l1 = (1 / n) * l(ae1s, d) + (1 - 1/n) * l(ae2ae1s, d)
                l2 = (1 / n) * l(ae2s, d) - (1 - 1/n) * l(ae2ae1s, d)
                l1s.append(torch.mean(l1).item())
                l2s.append(torch.mean(l2).item())
                loss = torch.mean(l1 + l2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            tqdm.write(
                f'Epoch {epoch},\tL1 = {np.mean(l1s)},\tL2 = {np.mean(l2s)}')
            return np.mean(l1s)+np.mean(l2s), optimizer.param_groups[0]['lr']
        else:
            ae1s, ae2s, ae2ae1s = [], [], []
            for d in data:
                d = d.double().to(device)
                ae1, ae2, ae2ae1 = model(d)
                ae1s.append(ae1)
                ae2s.append(ae2)
                ae2ae1s.append(ae2ae1)
            ae1s, ae2s, ae2ae1s = torch.stack(
                ae1s), torch.stack(ae2s), torch.stack(ae2ae1s)
            y_pred = ae1s[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = 0.1 * l(ae1s, data) + 0.9 * l(ae2ae1s, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    elif model.name in ['GDN', 'MTAD_GAT', 'MSCRED', 'CAE_M']:
        l = nn.MSELoss(reduction='none')
        n = epoch + 1
        w_size = model.n_window
        l1s = []
        if training:
            for i, d in enumerate(data):
                d = d.double().to(device)
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, h if i else None)
                else:
                    x = model(d)
                loss = torch.mean(l(x, d))
                l1s.append(torch.mean(loss).item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            tqdm.write(f'Epoch {epoch},\tMSE = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            xs = []
            data = data.double().to(device)
            for d in data:
                if 'MTAD_GAT' in model.name:
                    x, h = model(d, None)
                else:
                    x = model(d)
                xs.append(x)
            xs = torch.stack(xs)
            y_pred = xs[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            loss = l(xs, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    elif 'GAN' in model.name:
        l = nn.MSELoss(reduction='none')
        bcel = nn.BCELoss(reduction='mean')
        msel = nn.MSELoss(reduction='mean')
        real_label, fake_label = torch.tensor(
            [0.9]), torch.tensor([0.1])  # label smoothing
        real_label, fake_label = real_label.type(
            torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
        real_label, fake_label = real_label.to(device), fake_label.to(device)
        data = data.double().to(device)
        n = epoch + 1
        w_size = model.n_window
        mses, gls, dls = [], [], []
        if training:
            for d in data:
                # training discriminator
                model.discriminator.zero_grad()
                _, real, fake = model(d)
                dl = bcel(real, real_label) + bcel(fake, fake_label)
                dl.backward()
                model.generator.zero_grad()
                optimizer.step()
                # training generator
                z, _, fake = model(d)
                mse = msel(z, d)
                gl = bcel(fake, real_label)
                tl = gl + mse
                tl.backward()
                model.discriminator.zero_grad()
                optimizer.step()
                mses.append(mse.item())
                gls.append(gl.item())
                dls.append(dl.item())
                # tqdm.write(f'Epoch {epoch},\tMSE = {mse},\tG = {gl},\tD = {dl}')
            tqdm.write(
                f'Epoch {epoch},\tMSE = {np.mean(mses)},\tG = {np.mean(gls)},\tD = {np.mean(dls)}')
            return np.mean(gls)+np.mean(dls), optimizer.param_groups[0]['lr']
        else:
            outputs = []
            for d in data:
                d = d.double().to(device)
                z, _, _ = model(d)
                outputs.append(z)
            outputs = torch.stack(outputs)
            y_pred = outputs[:, data.shape[1] -
                             feats:data.shape[1]].view(-1, feats)
            loss = l(outputs, data)
            loss = loss[:, data.shape[1]-feats:data.shape[1]].view(-1, feats)
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
    elif 'TranAD' in model.name:
        l = nn.MSELoss(reduction='none')
        data_x = torch.tensor(data, dtype=torch.double)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        if training:
            for d, _ in dataloader:
                d = d.to(device)
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = l(z, elem) if not isinstance(z, tuple) else (
                    1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
                if isinstance(z, tuple):
                    z = z[1]
                l1s.append(torch.mean(l1).item())
                loss = torch.mean(l1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step()
            tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
            return np.mean(l1s), optimizer.param_groups[0]['lr']
        else:
            for d, _ in dataloader:
                d = d.to(device)
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, bs, feats)
                z = model(window, elem)
                if isinstance(z, tuple):
                    z = z[1]
            loss = l(z, elem)[0]
            return loss.cpu().detach().numpy(), z.cpu().detach().numpy()[0]
    else:
        data = torch.tensor(data, dtype=torch.float64).to(device)
        y_pred = model(data)
        loss = l(y_pred, data)
        if training:
            tqdm.write(f'Epoch {epoch},\tMSE = {loss}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            return loss.item(), optimizer.param_groups[0]['lr']
        else:
            return loss.cpu().detach().numpy(), y_pred.cpu().detach().numpy()


def run_model(model_name: str, source: str, lr: float):
    # For each fold of the dataset, train and test the model
    results_avg = []
    for k in range(5):
        # Set seed to ensure reproducibility
        random.seed(FIXED_SEED)
        torch.manual_seed(FIXED_SEED)
        np.random.seed(FIXED_SEED)

        # Initialize wandb
        name = f"{model_name}-{source}-lr{lr}"
        tags = [f"lr{lr}", model_name, source]

        # Add the name with suffixes (but not the fold number) to the tags
        group = name
        tags.append(group)

        # Add the fold number to the name
        name += f"-fold_{k}"
        project_name = f"{dataset}-{model_name}-kfold-{VERSION}"

        args.model = model_name
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            entity=WANDB_USER,
            # config=params,
            name=name,
            tags=tags,
            group=group
        )

        train_loader, test_loader, labels = load_dataset(
            dataset=dataset, k=k, source=source)
        if args.model in ['MERLIN']:
            eval(
                f'run_{args.model.lower()}(test_loader, labels, args.dataset)')
        model, optimizer, scheduler, epoch, accuracy_list = load_model(
            args.model, labels.shape[1], lr)

        # Prepare data
        trainD, testD = next(iter(train_loader)), next(
            iter(test_loader))
        trainO, testO = trainD, testD
        if model.name in ['Attention', 'DAGMM', 'USAD', 'MSCRED', 'CAE_M', 'GDN', 'MTAD_GAT', 'MAD_GAN'] or 'TranAD' in model.name:
            trainD, testD = convert_to_windows(
                trainD, model), convert_to_windows(testD, model)

        # Training phase
        if not args.test:
            num_epochs = 20
            e = epoch + 1
            start = time()
            for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
                # Training phase
                lossT, _ = backprop(
                    e, model, trainD, trainO, optimizer, scheduler)
                accuracy_list.append((lossT, lr))

                if e % 5 == 0:
                    # Testing phase
                    torch.zero_grad = True
                    model.eval()
                    loss, y_pred = backprop(
                        0, model, testD, testO, optimizer, scheduler, training=False)

                    # ### Plot curves
                    # if not args.test:
                    #     if 'TranAD' in model.name: testO = torch.roll(testO, 1, 0)
                    #     plotter(f'{args.model}_{args.dataset}', testO, y_pred, loss, labels)

                    # Scores
                    lossT, _ = backprop(
                        0, model, trainD, trainO, optimizer, scheduler, training=False)

                    lossTfinal = np.mean(lossT, axis=1)
                    lossFinal = np.mean(loss, axis=1)
                    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0

                    combined_energy = np.concatenate([lossTfinal, lossFinal], axis=0)

                    # AUC metrics
                    auc_roc = roc_auc_score(labelsFinal, lossFinal)
                    auc_pr = average_precision_score(labelsFinal, lossFinal)

                    wandb.log({
                        "AUC_ROC": auc_roc,
                        "AUC_PR": auc_pr
                    })

                    # Test for different anomaly thresholds
                    for anomaly_ratio in range(1, 7):
                        threshold = np.percentile(combined_energy, 100 - anomaly_ratio)
                        
                        # (3) evaluation on the test set
                        pred = (lossFinal > threshold).astype(int)

                        # print("pred:   ", pred.shape)
                        # print("gt:     ", gt.shape)

                        # (4) detection adjustment
                        gt, pred = adjustment(labelsFinal, pred)

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
                        MIoU = 0.5 * (tp / (tp + fp + fn) +
                                      tn / (tn + fn + fp))

                        # print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                        #     accuracy, precision,
                        #     recall, f_score))

                        wandb.log({
                            f"f1_score_ar{anomaly_ratio}": f_score,
                            f"precision_ar{anomaly_ratio}": precision,
                            f"recall_ar{anomaly_ratio}": recall,
                            f"accuracy_ar{anomaly_ratio}": accuracy,
                            f"MIoU_ar{anomaly_ratio}": MIoU,

                            f"threshold_ar{anomaly_ratio}": threshold,

                            f"TP_ar{anomaly_ratio}": tp,
                            f"TN_ar{anomaly_ratio}": tn,
                            f"FP_ar{anomaly_ratio}": fp,
                            f"FN_ar{anomaly_ratio}": fn
                        })
            # End of training
            wandb.finish()


if __name__ == '__main__':
    # Dataset
    dataset = "HCI"

    # For each model
    models = ['TranAD', 'LSTM_AD', 'CAE_M', 'DAGMM', 'USAD', 'OmniAnomaly', 'MTAD_GAT', 'GDN', 'MAD_GAN', 'MSCRED']

    # For each source
    sources = ['ECG', 'BVP']
    if dataset == 'DREAMER' or dataset == 'HCI':
        sources.remove('BVP')

    # For each learning rate
    lrs = [0.005, 0.001, 0.0005, 0.0001, 0.00005]

    # Run the models
    combined = [(model, source, lr)
                for model in models for source in sources for lr in lrs]

    if device == "cuda":
        for c in combined:
            run_model(*c)
    else:
        with Pool(2) as p:
            p.starmap(run_model, combined)
