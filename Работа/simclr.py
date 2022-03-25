import logging
import os
import sys
import copy
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, device, *args, **kwargs):
        self.args = kwargs['args']
        self.device = device
        self.model = kwargs['model'].to(self.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.checkpoint = self.args.checkpoint
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / 0.07 #softmax temperature (default: 0.07)
        return logits, labels

    def train(self, train_loader):
        PATH = './checkpoint_simcrl'  # '/content/drive/MyDrive/Работа/checkpoint_simcrl'
        start = 1
        scaler = GradScaler(enabled=True)  # optional
        if self.checkpoint:
            try:
                checkpoint = torch.load(PATH)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start = checkpoint['epoch']
                print('Found checkpoint - ' + str(start))
            except:
                print('No checkpoints available')
        else:
            print('No checkpoints')
        # save config file
        save_config_file(self.writer.log_dir, self.args)
        best_acc = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        n_iter = 0
        print(f"Start SimCLR training for {self.args.simcrl_epochs - start + 1} epochs.")
        for epoch_counter in range(start, self.args.simcrl_epochs + 1):
            top1_mean = 0
            cnt = 0
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.device)

                with autocast(enabled=True):  # optional
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                top1 = accuracy(logits, labels)
                top1_mean+=top1[0]
                cnt+=1
                if n_iter % self.args.log_every_n_steps == 0:
                    # , topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
            if self.checkpoint and epoch_counter % 10 == 0:
                torch.save({
                    'epoch': epoch_counter,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, PATH)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            print(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 mean accuracy: {top1_mean/cnt}")
            if top1_mean/cnt > best_acc:
                best_acc = top1_mean/cnt
            best_model_wts = copy.deepcopy(self.model.state_dict())
        logging.info("Training has finished.")
        self.model.load_state_dict(best_model_wts)