import torch
import numpy as np
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from scipy.special import softmax
import matplotlib.pyplot as plt
import torch.optim as optim

def warmup(net , dataloader,device, args):
    CEloss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    net.train()
    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        if args.noise_type == 'symmetric':
            L = loss
        elif args.noise_type == 'symmetric':
            penalty = NegEntropy(outputs)
            L = loss + penalty
        L.backward()
        optimizer.step()

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def sort_data(model, dataloader, device, args):
    model = model.to(device)
    print("new9")
    for i in range(args.warm_up):
        print("  ", i)
        warmup(model, dataloader, device, args)
    was_training = model.training
    model.eval()
    CE = nn.CrossEntropyLoss(reduction='none')
    losses = torch.zeros(len(dataloader.dataset.data))
    # p_i = torch.zeros(args.num_classes, len(dataloader.dataset.data))
    # index_i = torch.zeros(args.num_classes)
    p_max = torch.zeros(len(dataloader.dataset.data))
    answers = torch.zeros(len(dataloader.dataset.data))
    p_array = np.zeros((len(dataloader.dataset.data), args.num_classes))
    with torch.no_grad():
        p_index = 0
        for inputs, targets, index in dataloader:
            inputs, targets, index = inputs.to(device), targets.to(device), index.to(device)
            outputs = model(inputs)
            outputs_p = torch.softmax(outputs, dim=1)
            outputs = torch.tensor(outputs).to(device)
            loss = CE(outputs, targets)
            predictions, nums = torch.max(outputs_p, 1)
            for b in range(inputs.size(0)):
                losses[index[b]] = loss[b]
                # p_i[int(nums[b])][int(index_i[int(nums[b])])] = predictions[b]
                # index_i[int(nums[b])] += 1
                p_array[index[b]] = outputs_p[b]
                p_max[index[b]] = predictions[b]
                answers[index[b]] = nums[b]
    # p_i = [i[i!=0] for i in p_i]
    print(p_array)
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    input_loss = losses.reshape(-1, 1)
    # p_i = [i.reshape(-1,1) for i in p_i]
    gmm1 = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm1.fit(input_loss)
    prob1 = gmm1.predict_proba(input_loss)
    prob1 = prob1[:, gmm1.means_.argmin()]
    p_clean = (prob1 > args.p_clean)
    """p_noise = (prob1 <= args.p_clean)
    gmm2 = [GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4) for x in range(args.num_classes)]
    for i in range(args.num_classes):
        print(p_i[i].shape)
        if len(p_i[i]) > 0:
            gmm2[i].fit(p_i[i])

    prob2 = torch.zeros(len(dataloader.dataset.data))
    for j in range(len(dataloader.dataset.data)):
        prob2[j] = gmm2[int(answers[j])].predict_proba([[p_max[j]]])[0][gmm2[int(answers[j])].means_.argmin()]
    """
    gmm2 = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm2.fit(p_array)
    prob2 = gmm2.predict_proba(p_array)
    m = np.array([np.mean(x) for x in gmm2.means_])
    prob2 = prob2[:, m.argmin()]
    p_right = (prob2 > args.p_right)
    #p_wrong = (prob2 <= args.p_right)
    #print("p_right: ", prob2)
    #fig, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    #x = [x for x in range(1000)]
    #ax2.scatter(x=x, y=p_max[p_right][:1000], c='y', label='p_right')
    #ax2.scatter(x=x, y=p_max[p_wrong][:1000], c='b', label='not p_right')
    #plt.legend(loc='lower right')
    #plt.show()
    good = 0
    bad = 0
    correct = dataloader.dataset.original_targets
    new_targets = np.zeros(len(dataloader.dataset.data))
    for i in range(len(dataloader.dataset.data)):
        if p_clean[i] or ((p_right[i]) and (answers[i] == dataloader.dataset.targets[i])):
          new_targets[i] = answers[i]
          if ((not p_clean[i]) and (p_right[i]) and (answers[i] == dataloader.dataset.targets[i])):
            if new_targets[i] == correct[i]:
              good+=1
            else:
              bad+=1
        else:
            new_targets[i] = -1
    print("good: ", good)
    print("bad: ", bad)
    if args.testing:
        correct = dataloader.dataset.original_targets
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(dataloader.dataset.data)):
            if new_targets[i] != -1:
                if new_targets[i] == correct[i]:
                    TP += 1
                else:
                    FP += 1
            else:
                if dataloader.dataset.targets[i] == correct[i]:
                    FN += 1
                else:
                    TN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * (recall * precision) / (recall + precision)
        print("acc: ", TP + TN, (TP + TN) / len(dataloader.dataset.data))
        print("precision: ", precision)
        print("recall: ", recall)
        print("F1: ", F1)
        model.train(mode=was_training)
    return (TP + TN) / len(dataloader.dataset.data), precision, recall, F1
