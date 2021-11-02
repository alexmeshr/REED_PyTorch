import torch.nn as nn
def generate_graph(matr_in, treshold):
    sim = nn.CosineSimilarity(dim=len(matr_in[0]), eps=1e-6)
    matr_out = [[nn.ReLU(sim(matr_in[i], matr_in[j]) - treshold)
                 for j in range(len(matr_in))]for i in range(len(matr_in))]
    return matr_out
