import wandb
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn.functional as F

from datasets.esc50 import get_test_set, get_training_set
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT_part1, AugmentMelSTFT_part2, AugmentMelSTFT_part2_v2
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup
from torch import nn

class MelModel(nn.Module):
    def __init__(self, model, stft, mel):
        super().__init__()
        self.model = model
        self.mel = mel
        self.stft = stft

    def mel_requires_grad(self, requires_grad = True):        
        for param in self.mel.parameters():
            param.requires_grad = requires_grad

    def model_requires_grad(self, requires_grad = True):
        for param in self.model.parameters():
            param.requires_grad = requires_grad

    def train_only_classifier(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 'classifier' in name:        
                param.requires_grad = True

    def forward(self, x):
        old_shape = x.size()
        # reshape from: batch,1,samples -> batch,samples (1 is number of channels)
        x = x.reshape(-1, old_shape[2])
        x = self.stft(x)
        x = x.unsqueeze(1)
        x = self.mel(x)
        x = self.model(x)       
        return x        

def train(args):
    # Train Models for Acoustic Scene Classification

    # logging is done using wandb
    wandb.init(
        project="ESC50",
        notes="Fine-tune Models on ESC50.",
        tags=["Environmental Sound Classification", "Fine-Tuning"],
        config=args,
        mode='offline',
        name=args.experiment_name
    )

    device = 'cuda' if  torch.cuda.is_available() else 'cpu'

    # model to preprocess waveform into mel spectrograms
    stft = AugmentMelSTFT_part1(n_mels=args.n_mels,
                        sr=args.resample_rate,
                        win_length=args.window_size,
                        hopsize=args.hop_size,
                        n_fft=args.n_fft,
                        freqm=args.freqm,
                        timem=args.timem,
                        fmin=args.fmin,
                        fmax=args.fmax,
                        fmin_aug_range=args.fmin_aug_range,
                        fmax_aug_range=args.fmax_aug_range
                        )

    # AugmentMelSTFT_part2_v2 AugmentMelSTFT_part2
    mel = AugmentMelSTFT_part2_v2(n_mels=args.n_mels,
                         sr=args.resample_rate,
                         win_length=args.window_size,
                         hopsize=args.hop_size,
                         n_fft=args.n_fft,
                         freqm=args.freqm,
                         timem=args.timem,
                         fmin=args.fmin,
                         fmax=args.fmax,
                         fmin_aug_range=args.fmin_aug_range,
                         fmax_aug_range=args.fmax_aug_range
                         )
    

    # load prediction model
    model_name = args.model_name
    pretrained_name = model_name if args.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and args.pretrained else args.model_width
    if model_name.startswith("dymn"):
        _model = get_dymn(width_mult=width, pretrained_name=pretrained_name,
                         pretrain_final_temp=args.pretrain_final_temp,
                         num_classes=50)
    else:
        _model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                              head_type=args.head_type, se_dims=args.se_dims,
                              num_classes=50)
    
    stft.to(device)
    mel.to(device)
    _model.to(device)
    torch.save(mel.state_dict(), '/content/mel_start.pt')
    model = MelModel(_model, stft, mel)
    # load saved model
    if args.model_local != '':
        state_dict = torch.load(f = args.model_local,  map_location=device)        
        model.load_state_dict(state_dict)
    # train filter banks
    if args.no_train_mel:            
        print('Filter banks NOT optimized!')
        model.mel_requires_grad(requires_grad=False)
    else:
        model.mel_requires_grad(requires_grad=True)
        print('Filter banks optimized!')
    # train model parameteres
    if args.no_train_model:            
        print('Model NOT optimized!')
        model.model_requires_grad(requires_grad=False)
    else:
        model.model_requires_grad(requires_grad=True)
        print('Model optimized!')
    # train only classfier
    if args.train_only_classifier:
        print('train only classifier')
        model.train_only_classifier()

    # # DEBUG : check we are really training all
    # for param in model.parameters():
    #     param.requires_grad = True
    model.to(device)

    # dataloader
    dl = DataLoader(dataset=get_training_set(resample_rate=args.resample_rate,
                                             roll=False if args.no_roll else True,
                                             wavmix=False if args.no_wavmix else True,
                                             gain_augment=args.gain_augment,
                                             fold=args.fold),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True)

    # evaluation loader
    eval_dl = DataLoader(dataset=get_test_set(resample_rate=args.resample_rate, fold=args.fold),
                         worker_init_fn=worker_init_fn,
                         num_workers=args.num_workers,
                         batch_size=args.batch_size)
                         
    ORIGINAL = True
    # optimizer & scheduler

    if ORIGINAL:
        lr = args.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
        schedule_lambda = \
            exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    name = None
    accuracy, val_loss = float('NaN'), float('NaN')

    for epoch in range(args.n_epochs):
        model.train()
        train_stats = dict(train_loss=list())
        pbar = tqdm(dl)
        pbar.set_description("Epoch {}/{}: accuracy: {:.4f}, val_loss: {:.4f}"
                             .format(epoch + 1, args.n_epochs, accuracy, val_loss))
        for batch in pbar:
            x, f, y = batch
            bs = x.size(0)
            x, y = x.to(device), y.to(device)

            # if args.mixup_alpha:
            #     rn_indices, lam = mixup(bs, args.mixup_alpha)
            #     lam = lam.to(x.device)
            #     x = x * lam.reshape(bs, 1, 1, 1) + \
            #         x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
            #     y_hat, _ = model(x)
            #     samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
            #                     F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
            #                                 1. - lam.reshape(bs)))

            # else:
            # with torch.autograd.detect_anomaly(check_nan=False):
            y_hat, _ = model(x)
                    
            samples_loss = F.cross_entropy(y_hat, y, reduction="none")
            # loss
            loss = samples_loss.mean()
            # append training statistics
            train_stats['train_loss'].append(loss.detach().cpu().numpy())

            # Update Model
            loss.backward()
            # threshold = 100000.0
            # for p in model.mel.parameters():
            #     print(f'grad_norm={p.grad.norm().item()}')
            #     if p.grad.norm() < threshold:
            #         # print(p.grad.norm())
            #         torch.nn.utils.clip_grad_norm_(p, threshold)

            optimizer.step()
            # for param in model.mel.parameters():
            #     param.data.clamp_(min = 0.00001, max = 1.1)

            optimizer.zero_grad()
        # Update learning rate
        if ORIGINAL:        
            scheduler.step()

        # evaluate
        accuracy, val_loss = _test(model, eval_dl, device)

        # log train and validation statistics
        wandb.log({"train_loss": np.mean(train_stats['train_loss']),
                   "accuracy": accuracy,
                   "val_loss": val_loss
                   })

        # remove previous model (we try to not flood your hard disk) and save latest model
        if name is not None:
            os.remove(os.path.join(wandb.run.dir, name))
        name = f"mn{str(width).replace('.', '')}_esc50_epoch_{epoch}_acc_{int(round(accuracy*1000))}.pt"
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))

def _test(model, eval_loader, device):
    model.eval()

    targets = []
    outputs = []
    losses = []
    pbar = tqdm(eval_loader)
    pbar.set_description("Validating")
    for batch in pbar:
        x, f, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.cross_entropy(y_hat, y).cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    accuracy = metrics.accuracy_score(targets.argmax(axis=1), outputs.argmax(axis=1))
    return accuracy, losses.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="ESC50")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--fold', type=int, default=1)

    # training
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default="mn10_as")
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--no_roll', action='store_true', default=False)
    parser.add_argument('--no_wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=12)
    parser.add_argument('--weight_decay', type=int, default=0.0)

    # lr schedule
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--warm_up_len', type=int, default=10)
    parser.add_argument('--ramp_down_start', type=int, default=10)
    parser.add_argument('--ramp_down_len', type=int, default=65)
    parser.add_argument('--last_lr_value', type=float, default=0.01)

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    # custom, different to original repo
    parser.add_argument('--model_local', type=str, default='')
    parser.add_argument('--no_train_mel', action='store_true', default=False)
    parser.add_argument('--no_train_model', action='store_true', default=False)
    parser.add_argument('--train_only_classifier', action='store_true', default=False)

    args = parser.parse_args()
    train(args)
