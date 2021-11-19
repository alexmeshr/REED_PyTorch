from torch import linalg as LA, optim
import torch
import sys
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from transformer import *
import torch.nn.functional as F
from training import validate_model
def generate_graph(matr_in, treshold):
    matr_out = [[nn.ReLU(F.cosine_similarity(matr_in[i], matr_in[j], dim=0) - treshold)
                 for j in range(len(matr_in))]for i in range(len(matr_in))]
    return matr_out

def class_balanced_sampler(targets, nclasses):
    counts = [0] * (nclasses+1)
    for item in targets:
            counts[item] += 1
    weight_per_class = [0.] * (nclasses+1)
    N = float(sum(counts))
    for i in range(nclasses):
        weight_per_class[i] = N/float(counts[i])
    weights = [0] * len(targets)
    for i in range(len(targets)):
        weights[i] = weight_per_class[targets[i]]#if targets[i] == -1 then weights[i] will be 0
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

class GraphStructuredR():
        def __init__(self, matr_for_graph, args):
            self.matr_for_graph = matr_for_graph
            self.lamdLU = args.lamdLU
            self.lamdUU = args.lamdUU
            self.T = args.Temperature
            self.graph_treshold = args.graph_treshold
        def __call__(self, outputs, targets, index):#vector = cat(labeled, unlabeled)
            batch_size = len(outputs)//2
            """outputs_l = outputs[:batch_size]
            outputs_u = outputs[batch_size:]
            index_l = index[:batch_size]
            index_u = index[batch_size:]
            """ #it is already sharpened in mixmatch
            sum1 = 0
            sum2 = 0
            relu = nn.ReLU()
            for i in range(batch_size):
                for j in range(batch_size, 2*batch_size):

                    a1 = relu(F.cosine_similarity(self.matr_for_graph[index[i]], self.matr_for_graph[index[j]], dim=0) - self.graph_treshold)
                    sum1+=a1*(LA.vector_norm(outputs[j] - targets[i], ord=2)**2)
                    a2 = relu(F.cosine_similarity(self.matr_for_graph[index[i+batch_size]], self.matr_for_graph[index[j]], dim=0) - self.graph_treshold)
                    sum2+=a2*(LA.vector_norm(outputs[j] - outputs[i+batch_size], ord=2)**2)
            return self.lamdLU*sum1+self.lamdUU*sum2

def MixMatch(net, data, p_matr, args, test_loader):
    print("MM")
    R = GraphStructuredR(matr_for_graph=p_matr, args=args)
    print("Graph done")
    criterion = SemiLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    labeled_trainloader = DataLoader(dataset=data,
                                  sampler=class_balanced_sampler(data.targets, args.num_classes),
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  drop_last=False)

    unlabeled_trainloader = DataLoader(dataset=data,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     drop_last=False)

    net.train()
    unlabeled_train_iter = iter(unlabeled_trainloader)
    print("start")
    for epoch in range(args.third_stage_epochs):
        running_loss = 0.0
        cnt = 0
        for inputs_x, labels_x, index_x in labeled_trainloader:
            try:
                inputs_u, _, index_u = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, _, index_u = unlabeled_train_iter.next()
            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1, 1), 1)

            inputs_x, labels_x, index_x = inputs_x.cuda(), labels_x.cuda(), index_x.cuda()
            inputs_u, index_u = inputs_u.cuda(), index_u.cuda()

            with torch.no_grad():
                outputs_u = net(inputs_u)
                pu = torch.softmax(outputs_u, dim=1)
                ptu = pu ** (1 / args.Temperature)  # temparature sharpening
                targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                targets_u = targets_u.detach()

                outputs_x = net(inputs_x)
                px = torch.softmax(outputs_x, dim=1)
                ptx = px ** (1 / args.Temperature)  # temparature sharpening
                targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                targets_x = targets_x.detach()

            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)
            all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
            all_targets = torch.cat([labels_x, targets_u], dim=0)
            all_index = torch.cat([index_x, index_u], dim=0)
            R_outputs = torch.cat([targets_x, targets_u], dim=0)
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net(mixed_input)
            logits_x = logits[:batch_size]
            logits_u = logits[batch_size:]

            Lx, Lu = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:])


            loss = Lx + args.MMlamb * Lu + R(R_outputs, labels_x, all_index)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            cnt+=1
        print(' Epoch [%3d/%3d]\t CE-loss: %.4f' % (epoch, args.third_stage_epochs, running_loss/cnt))
        validate_model(net, test_loader)
    return net


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu


"""
test_data = torchvision.datasets.CIFAR10(root='data/', train=True, download=True, transform=transform_train("cifar10"))
test_loader = DataLoader(dataset=test_data,sampler=TestClassBalancedSampler(test_data.targets),
                                  batch_size=4,
                                  shuffle=False,
                                  num_workers=1,
                                  drop_last=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    freeze_support()
    for i in range(10):
        inputs, labels = next(iter(test_loader))
        print(labels)
"""