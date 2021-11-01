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
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = CEloss(outputs, labels)
        L = loss  # + penalty
        L.backward()
        optimizer.step()



def sort_data(model, dataloader, device, args):
    model = model.to(device)
    for i in range(args.warm_up):
        print("  ", i)
        warmup(model, dataloader, device, args)
    was_training = model.training
    model.eval()
    CE = nn.CrossEntropyLoss(reduction='none')
    losses = torch.zeros(len(dataloader.dataset.data))
    #p_i = torch.zeros(args.num_classes, len(dataloader.dataset.data))
    #index_i = torch.zeros(args.num_classes)
    p_max = torch.zeros(len(dataloader.dataset.data))
    answers = torch.zeros(len(dataloader.dataset.data))
    #p_array = np.array([])
    with torch.no_grad():
        index = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).cpu().numpy()
            outputs_p = np.array([softmax(output) for output in outputs])
            #for output in outputs_p:
            #  p_array = np.append(p_array, output)
            outputs_p =torch.tensor(outputs_p).to(device)
            outputs = torch.tensor(outputs).to(device)
            loss = CE(outputs, targets)
            predictions, nums = torch.max(outputs_p, 1)
            for b in range(inputs.size(0)):
                losses[index] = loss[b]
                #p_i[int(nums[b])][int(index_i[int(nums[b])])] = predictions[b]
                #index_i[int(nums[b])] += 1
                p_max[index] = predictions[b]
                #answers[index] = nums[b]
                index += 1
    #p_i = [i[i!=0] for i in p_i]
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    input_loss = losses.reshape(-1, 1)
    #p_i = [i.reshape(-1,1) for i in p_i]
    gmm1 = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm1.fit(input_loss)
    prob1 = gmm1.predict_proba(input_loss)
    prob1 = prob1[:, gmm1.means_.argmin()]
    p_clean = (prob1 > args.p_clean)
    p_noise = (prob1 <= args.p_clean)
    #print("p_clean: ", prob1)
    #gmm2 = [GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4) for x in range(args.num_classes)]
    #for i in range(args.num_classes):
    #    gmm2[i].fit(p_i[i])
    #prob2 = torch.zeros(len(dataloader.dataset.data))
    #for j in range(len(dataloader.dataset.data)):
    #    prob2[j] = gmm2[int(answers[j])].predict_proba([[p_max[j]]])[0][gmm2[int(answers[j])].means_.argmin()]
    #prob2 = gmm2.predict_proba(p_max)
    #prob2 = prob2[:, gmm2.means_.argmin()]
    #p_right = (prob2 > args.p_right)
    #p_wrong = (prob2 <= args.p_right)
    #print("p_right: ", prob2)
    #for x in range(10000):
    #    if p_right[x]:
    #        print(x, prob2[x], p_max[x])
    fig, (ax2, ax) = plt.subplots(2,1, figsize=(10, 8))
    x = [x for x in range(1000)]
    #ax.scatter(x = x, y=p_max[p_right][:1000], c = 'g', label='P_right')
    #ax.scatter(x = x, y=p_max[p_wrong][:1000], c = 'r', label='not P_right')
    #plt.legend(loc='upper right')
    ax2.scatter(x = x, y=p_max[p_clean][:1000], c = 'b', label='P_clean')
    ax2.scatter(x = x, y=p_max[p_noise][:1000], c = 'y', label='not P_clean')
    plt.legend(loc='upper right')
    plt.show()
    new_targets = np.zeros(len(dataloader.dataset.data))
    for i in range(len(dataloader.dataset.data)):
        if p_clean[i]:  # or ((not p_right[i]) and (answers[i] == dataloader.dataset.targets[i])):
            new_targets[i] = dataloader.dataset.targets[i]
        else:
            new_targets[i] = -1
    if args.testing:
        correct = dataloader.dataset.original_targets
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for i in range(len(dataloader.dataset.data)):
            if new_targets[i] != -1:
                if new_targets[i] == correct[i]:
                    TP+=1
                else:
                    FP+=1
            else:
                if dataloader.dataset.targets[i] == correct[i]:
                    FN+=1
                else:
                    TN+=1
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        F1 = 2 * (recall * precision) / (recall + precision)
        print("acc: ", TP + TN, (TP + TN) / len(dataloader.dataset.data))
        print("precision: ", precision)
        print("recall: ", recall)
        print("F1: ", F1)
        model.train(mode=was_training)
    return (TP + TN) / len(dataloader.dataset.data), precision, recall, F1
