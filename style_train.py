import numpy as np
import torch
import time
import glob
import os
import matplotlib as plt
from PIL import Image
from PIL import ImageFile

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable

from option.base_options import BaseOptions
from network.loss_net import CreateLossNetwork
from network.transformer_net import TransformerNet
from utils import CalculateGram, recover_image

def InitRand(opt):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

def CreateDataLoader(opt):
    transform = transforms.Compose([transforms.Resize([opt.loadSize,opt.loadSize]),
    #                                 transforms.CenterCrop(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                                    ])
    # http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder
    train_dataset = datasets.ImageFolder(opt.dataroot, transform)
    # http://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize,
                              shuffle=not opt.serial_batches, num_workers=opt.nThreads)
    return train_dataset, train_loader

def LoadStyleImage(opt):
    style_img = Image.open(opt.style_image).convert('RGB')
    style_img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
    )(style_img).unsqueeze(0)
    # assert np.sum(style_img - recover_image(style_img_tensor.numpy())[0].astype(np.uint8)) < 3 * style_img_tensor.size()[2] * style_img_tensor.size()[3]
    if torch.cuda.is_available():
        style_img_tensor = style_img_tensor.cuda(opt.gpu_ids[0])

    # plt.imshow(recover_image(style_img_tensor.cpu().numpy())[0])
    return style_img_tensor

def GetGram(loss_network, style_img_tensor):
    # http://pytorch.org/docs/master/notes/autograd.html#volatile
    style_loss_features = loss_network(Variable(style_img_tensor, volatile=True))
    gram_style = [Variable(CalculateGram(y).data, requires_grad=False) for y in style_loss_features]

    print(np.mean(gram_style[3].data.cpu().numpy()))
    print(np.mean(style_loss_features[3].data.cpu().numpy()))
    print(gram_style[0].numel())

    return style_loss_features, gram_style

def save_debug_image(tensor_orig, tensor_transformed, filename):
    assert tensor_orig.size() == tensor_transformed.size()
    result = Image.fromarray(recover_image(tensor_transformed.cpu().numpy())[0])
    orig = Image.fromarray(recover_image(tensor_orig.cpu().numpy())[0])
    new_im = Image.new('RGB', (result.size[0] * 2 + 5, result.size[1]))
    new_im.paste(orig, (0,0))
    new_im.paste(result, (result.size[0] + 5,0))
    new_im.save(filename)

def model_test(transformer, opt):
    transformer = transformer.eval()
    fnames = glob.glob(opt.dataroot + r"/*/*")
    data_len = len(fnames)
    img = Image.open(fnames[20 % data_len]).convert('RGB')
    transform = transforms.Compose([transforms.Resize([opt.loadSize, opt.loadSize]),
                                    #                                 transforms.CenterCrop(IMAGE_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                                    ])
    img_tensor = transform(img).unsqueeze(0)
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda(opt.gpu_ids[0])

    img_output = transformer(Variable(img_tensor, volatile=True))
    # plt.imshow(recover_image(img_tensor.cpu().numpy())[0])
    save_testImage_path = "{}/{}_{}.jpg".format(os.path.join(opt.checkpoints_dir, opt.exp_name),opt.epochs, 'test')
    print("save test image to {}".format(save_testImage_path))
    save_debug_image(img_tensor, img_output.data, save_testImage_path)

def model_save(transformer, opt):
    save_model_path = os.path.join(opt.checkpoints_dir, opt.exp_name, "model_udnie.pth")
    print("save model to {}".format(save_model_path))
    torch.save(transformer.state_dict(), save_model_path)
    print("done!!")

def model_train(opt):
    InitRand(opt)
    dataset, data_loader = CreateDataLoader(opt)
    loss_network = CreateLossNetwork(opt)
    style_img_tensor = LoadStyleImage(opt)
    _, gram_style = GetGram(loss_network, style_img_tensor)

    transformer = TransformerNet()
    # l1_loss = torch.nn.L1Loss()
    if torch.cuda.is_available():
        transformer.cuda(opt.gpu_ids[0])
    transformer.train()
    mse_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(transformer.parameters(), opt.lr)

    for epoch in range(opt.epochs):
        agg_content_loss = 0.
        agg_style_loss = 0.
        agg_reg_loss = 0.
        count = 0
        # for batch_id, (x, _) in tqdm_notebook(enumerate(data_loader), total=len(data_loader)):
        for batch_id, (x, _) in enumerate(data_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = Variable(x)
            if torch.cuda.is_available():
                x = x.cuda(opt.gpu_ids[0])

            y = transformer(x)
            xc = Variable(x.data, volatile=True)

            features_y = loss_network(y)
            features_xc = loss_network(xc)

            f_xc_c = Variable(features_xc[opt.content_layer].data, requires_grad=False)

            content_loss = opt.content_weight * mse_loss(features_y[opt.content_layer], f_xc_c)

            reg_loss = opt.regularization_weight * (
                torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = gram_style[m]
                gram_y = CalculateGram(features_y[m])
                #             style_loss += STYLE_WEIGHT * mse_loss(gram_y, gram_s)
                style_loss += opt.style_weight * mse_loss(gram_y, gram_s.expand_as(gram_y))

            total_loss = content_loss + style_loss + reg_loss
            total_loss.backward()
            optimizer.step()

            agg_content_loss += content_loss.data[0]
            agg_style_loss += style_loss.data[0]
            agg_reg_loss += reg_loss.data[0]

            if (batch_id + 1) % opt.display_freq == 0:
                mesg = "{} [{}/{}] content: {:.6f}  style: {:.6f}  reg: {:.6f}  total: {:.6f}".format(
                    time.ctime(), count, len(dataset),
                    agg_content_loss / opt.display_freq,
                    agg_style_loss / opt.display_freq,
                    agg_reg_loss / opt.display_freq,
                    (agg_content_loss + agg_style_loss + agg_reg_loss) / opt.display_freq
                )
                print(mesg)
                agg_content_loss = 0
                agg_style_loss = 0
                agg_reg_loss = 0
                transformer.eval()
                y = transformer(x)
                save_debug_image(x.data, y.data,
                                 "{}/{}_{}.jpg".format(os.path.join(opt.checkpoints_dir, opt.exp_name),
                                 epoch, count))
                transformer.train()
    return transformer

if __name__ == "__main__":
    opt = BaseOptions().parse()
    transformer = model_train(opt)
    model_test(transformer, opt)
    model_save(transformer, opt)