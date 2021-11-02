import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler


def generate_graph(matr_in, treshold):
    sim = nn.CosineSimilarity(dim=len(matr_in[0]), eps=1e-6)
    matr_out = [[nn.ReLU(sim(matr_in[i], matr_in[j]) - treshold)
                 for j in range(len(matr_in))]for i in range(len(matr_in))]
    return matr_out

def ClassBalancedSampler(targets, nclasses):
    counts = [0] * nclasses
    for item in targets:
        counts[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(counts))
    for i in range(nclasses):
        weight_per_class[i] = N/float(counts[i])
    weights = [0] * len(targets)
    for i in range(len(targets)):
        weights[i] = weight_per_class[targets[i]]
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler

def graph_structured_r(graph,args):#lamdLU, lambUU, Temperature
    pass
