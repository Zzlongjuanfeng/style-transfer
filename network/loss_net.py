from collections import namedtuple
import torch
import torchvision.models.vgg as vgg

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)


def CreateLossNetwork(opt):
    vgg_model = vgg.vgg16(pretrained=True)
    if torch.cuda.is_available():
        vgg_model.cuda(opt.gpu_ids[0])
    loss_network = LossNetwork(vgg_model)
    loss_network.eval()
    del vgg_model
    return loss_network