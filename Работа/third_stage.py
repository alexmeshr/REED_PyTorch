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
    counts = [0] * (nclasses + 1)
    for item in targets:
        counts[item] += 1
    weight_per_class = [0.] * (nclasses + 1)
    N = float(sum(counts))
    for i in range(nclasses):
        weight_per_class[i] = N / float(counts[i])
    weights = [0] * len(targets)
    for i in range(len(targets)):
        weights[i] = weight_per_class[targets[i]]  # if targets[i] == -1 then weights[i] will be 0
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    print("class_balanced_sampler done")
    return sampler


"""@torch.jit.script
def create_matrix_A(p_matr, graph_treshold):
    size = len(p_matr)
    print("begin A: {}x{}".format(size, size))
    A = torch.zeros([size,size], dtype=torch.float32)#, device = 'cuda:0')
    for i in range(size):
        for j in range(size):
            A[i][j] = F.relu(F.cosine_similarity(p_matr[i], p_matr[j], dim=0) - graph_treshold)
    print("done A")
    return A
    """


# print(create_matrix_A.code)
@torch.jit.script
def GraphStructuredR(outputs, y, index, p_matr, lamdLU, lamdUU, graph_treshold):  # vector = cat(labeled, unlabeled)
    batch_size = len(outputs) // 2
    """outputs_l = outputs[:batch_size]
    outputs_u = outputs[batch_size:]
    index_l = index[:batch_size]
    index_u = index[batch_size:]
    """  # it is already sharpened in mixmatch

    sum1 = torch.zeros(1, dtype=torch.float32, device='cuda')
    sum2 = torch.zeros(1, dtype=torch.float32, device='cuda')
    # print("begin GSR")
    for i in range(batch_size):
        for j in range(batch_size, 2 * batch_size):
            a1 = F.relu(F.cosine_similarity(p_matr[index[i]], p_matr[index[j]], dim=0) - graph_treshold)
            sum1 += a1 * (sum((outputs[j] - y[i]) ** 2))
            a2 = F.relu(F.cosine_similarity(p_matr[index[i + batch_size]], p_matr[index[j]], dim=0) - graph_treshold)
            sum2 += a2 * (sum((outputs[j] - outputs[i + batch_size]) ** 2))
            # print("a1 = {} a2 = {} s1 = {} s2 ={}\n".format(a1, a2, sum((outputs[j] - y[i])**2), sum((outputs[j] - outputs[i+batch_size])**2)))
    # print("end GSR\n\n")
    return lamdLU * sum1 + lamdUU * sum2


# print(GraphStructuredR.code)


def MixMatch(net, data, h_matr, args, test_loader, use_cuda):
    criterion = SemiLoss()
    start = 1
    PATH = './checkpoint_third'
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    try:
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start = checkpoint['epoch']
        print('Found checkpoint - ' + str(start))
    except:
        print('No checkpoints available')
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
    # A_matr = create_matrix_A(p_matr, torch.tensor(args.graph_treshold))
    print("start")
    for epoch in range(start, args.third_stage_epochs + 1):
        print(epoch)
        running_loss = 0.0
        running_Rloss = 0.0
        cnt = 0
        inputs_u = []
        index_u = []
        for inputs_x, labels_x, index_x in labeled_trainloader:
            try:
                inputs_u, _, index_u = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, _, index_u = unlabeled_train_iter.next()
            batch_size = inputs_x.size(0)
            if use_cuda:
                labels_x = torch.zeros(batch_size, args.num_classes).scatter_(1, labels_x.view(-1, 1), 1)
                inputs_x, labels_x, index_x = inputs_x.cuda(), labels_x.cuda(non_blocking=True), index_x.cuda()
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
            # print("\n outputs_u = {}\n targets_u = {}\n outputs_x = {}\n targets_x = {}\n".format(outputs_u,targets_u, outputs_x, targets_x))
            # mixmatch
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1 - l)

            all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
            all_targets = torch.cat([labels_x, targets_u], dim=0)
            all_index = torch.cat([index_x, index_u], dim=0)
            R_outputs = torch.cat([targets_x, targets_u], dim=0)
            idx = torch.randperm(all_inputs.size(0))
            # print("\n outputs_R = {}\n targets_+ = {}{}\n".format(R_outputs,targets_x,targets_u))
            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            logits = net(mixed_input)
            logits_x = logits[:batch_size]
            logits_u = logits[batch_size:]

            Lx, Lu = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:])
            if use_cuda:
                R_outputs = R_outputs.cuda()
            test_r = GraphStructuredR(R_outputs, labels_x, all_index, h_matr, torch.tensor(args.lamdLU),
                                      torch.tensor(args.lamdUU), torch.tensor(args.graph_treshold))
            loss = Lx + args.MMlamb * Lu + test_r
            # print( "R = {}\nloss = {}\n".format(test_r, loss)  )
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_Rloss += test_r
            cnt += 1
            if cnt % 1000 == 0:
                print("#", end="")
        print(' Epoch [%3d/%3d]\t CE-loss: %.4f R-loss: %.4f' % (
        epoch, args.third_stage_epochs, running_loss / cnt, running_Rloss / cnt))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, PATH)
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