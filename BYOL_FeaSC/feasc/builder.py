import torch
import torch.nn as nn
import copy


class BYOL_FeaSC(nn.Module):
    """
    Build a BYOL_FeaSC model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(BYOL_FeaSC, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer


        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad = False


    def forward(self, x1, x2, gamma=0.2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """

        rep1, rep1_l = self.encoder(x1, return_feat=True, gamma=gamma)
        rep2, rep2_l = self.encoder(x2, return_feat=True, gamma=gamma)
        z1 = self.predictor(rep1) # NxC
        z2 = self.predictor(rep2) # NxC

        z1_l = self.predictor(rep1_l)
        z2_l = self.predictor(rep2_l)

        with torch.no_grad():
            p1 = self.teacher(x1)
            p2 = self.teacher(x2)

        return z1, z2, z1_l, z2_l, p1.detach(), p2.detach()
