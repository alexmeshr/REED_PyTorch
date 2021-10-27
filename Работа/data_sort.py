import torch
import numpy as np
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from scipy.special import softmax
import matplotlib.pyplot as plt

def sort_data(model, dataloader, device, args):
    was_training = model.training
    model.eval()
    losses = torch.zeros(len(dataloader.dataset.data))
    p_max = torch.zeros(len(dataloader.dataset.data))
    p_array = np.array([])
    CE = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        index = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs).cpu().numpy()
            outputs = np.array([softmax(output) for output in outputs])
            for output in outputs:
              p_array = np.append(p_array, output)
            outputs =torch.tensor(outputs).cuda()
            loss = CE(outputs, targets)
            _, predictions = torch.max(outputs, 1)
            for b in range(inputs.size(0)):
                losses[index] = loss[b]
                p_max[index] = predictions[b]
                index += 1
    losses = (losses - losses.min()) / (losses.max() - losses.min())
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [x for x in range(1000)]
    ax.scatter(x = x, y=losses[:1000])
    plt.show()
    print(losses)
    input_loss = losses.reshape(-1, 1)
    p_max = p_max.reshape(-1, 1)
    print("p_max:", p_max, p_max.shape)
    # fit a two-component GMM to the loss
    gmm1 = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm1.fit(input_loss)
    prob1 = gmm1.predict_proba(input_loss)
    prob1 = prob1[:, gmm1.means_.argmin()]
    p_clean = (prob1 > args.p_threshold)
    print("p_clean: ", prob1)
    p_array = torch.tensor(p_array)
    p_array = p_array.reshape(-1, 1)
    print("p_array: ", p_array, p_array.shape)
    gmm2 = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm2.fit(p_array)
    prob2 = gmm2.predict_proba(p_max)
    prob2 = prob2[:, gmm2.means_.argmin()]
    p_right = (prob2 > args.p_right)
    print("p_right: ", prob2)
    for x in range(1000):
        if prob2[x] < 0.9:
            print(x, prob2[x], p_max[x])
    all_data = np.array(list(zip(dataloader.dataset.data, dataloader.dataset.targets)))
    correct = np.zeros(len(dataloader.dataset.data), dtype=np.bool_)
    i = 0
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    """
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if (pred[i]):
                    all_data[i] = prediction


                answ = (label == prediction)
                if answ:
                    if dataloader.dataset.data.noise_or_not[i]:
                        TP += 1
                    else:
                        FP += 1

                else:
                    if dataloader.dataset.data.noise_or_not[i]:
                        FN += 1
                    else:
                        TN += 1
                correct[i] = answ
                i += 1
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    F1 = 2 * (recall * precision) / (recall + precision)
    print("acc: ", TP + TN, (TP + TN) / i)
    print("precision: ", precision)
    print("recall: ", recall)
    print("F1: ", F1)"""
    model.train(mode=was_training)
    return (TP + TN)# / i, precision, recall, F1
