import torch
import torch.nn as nn
import torchvision.models as models
from lcd.netvlad import NetVLAD


class ExtractDeepVladFeature(nn.Module):
    def __init__(self, net_vlad_ckp):

        super(ExtractDeepVladFeature, self).__init__()

        # model for feature extraction, consisting of encoder and netvlad two parts
        self.model = nn.Module()

        # vgg16 encoder
        encoder = models.vgg16(pretrained=False)

        # capture only feature part and remove last relu and maxpool
        layers = list(encoder.features.children())[:-2]

        # do not need to update parameters
        for l in layers[:-5]:
            for p in l.parameters():
                p.requires_grad = False

        encoder = nn.Sequential(*layers)

        # add the vgg16 encoder to the model
        self.model.add_module('encoder', encoder)

        # NetVlad
        netvlad = NetVLAD()

        # add netvlad to the model
        self.model.add_module('pool', netvlad)

        # load pre-trained weights into the model
        # ckp = 'model_zoo/net_vlad/checkpoint.pth.tar'
        check_point = torch.load(net_vlad_ckp)
        print('==> Initialize NetVlad with [{}]'.format(net_vlad_ckp))
        self.model.load_state_dict(check_point['state_dict'])

        self.model.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.model.encoder(x)
            x = self.model.pool(x)

        return x
