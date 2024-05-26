import torch.nn as nn
import torch
class MetaNet(nn.Module):
    def __init__(self, features_size=2048, embed_dim=128, out_size=14, temperature=1):
        super(MetaNet, self).__init__()
        # self.hiden = nn.Sequential(
        #     nn.Linear(clip_size, 1024),
        #     nn.ReLU(True),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(True)
        # )
        self.temperature = temperature
        self.pseudo_labels = nn.Linear(features_size, out_size)
        self.weight = nn.Linear(features_size + embed_dim, 1)
        self.cls_emb = nn.Embedding(out_size, embed_dim)

        # nn.init.xavier_uniform_(self.pseudo_labels[-1].weight)
        nn.init.xavier_uniform_(self.cls_emb.weight)
        # self.weight.bias.data = 10 * torch.ones(1)

    def forward(self, features, labels):
        labels = self.cls_emb(labels)

        labels = torch.cat([features, labels], -1)
        weights = self.weight(labels).sigmoid()

        pseudo_labels = self.pseudo_labels(features)
        pseudo_labels /= self.temperature
        pseudo_labels = pseudo_labels.softmax(-1)
        return pseudo_labels, weights

class HeadsNet(nn.Module):
    def __init__(self, head, head2, head3):
        super(HeadsNet, self).__init__()
        self.head = head
        self.head2 = head2
        self.head3 = head3

    def forward(self, x):
        return self.head(x), self.head2(x), self.head3(x)