import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device=torch.device("cuda" if torch.cuda.is_available else "cpu")

imsize=(512,512)

loader=transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor(),
])
def image_loader(image_file):
    image=Image.open(image_file)
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)



# from test import *


unloader=transforms.ToPILImage()

def imshow(tensor,title=None):
    image=tensor.cpu().clone()
    image=image.squeeze(0)
    image=unloader(image)
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.set_size_inches(imsize[0]/100, imsize[1]/100)
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.imshow(image)
    ax.imshow(image, aspect='auto')
    fig.savefig("styled_image_output.jpg")
    if title is not None:
        plt.title(title)
    plt.savefig("styled_image_output.jpg")


# plt.figure()
# imshow(style_img,title="style_img")

# plt.figure()
# imshow(content_img,title="content_img")
    
# return input itself , act as transparent layer
class ContentLoss(nn.Module):
    def __init__(self,target):
        super(ContentLoss,self).__init__()
        self.target=target.detach()

    def forward(self,input):
        self.loss=F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a,b,c,d=input.size()
    features = input.view(a*b,c*d)
    G=torch.mm(features,features.t())
    return G.div(a*b*c*d)


class StyleLoss(nn.Module):
    def __init__(self,target_feature):
        super(StyleLoss, self).__init__()
        self.target=gram_matrix(target_feature).detach()
    def forward(self,input):
        G=gram_matrix(input)
        self.loss=F.mse_loss(G,self.target)
        return input


cnn=models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self,mean,std):
        super(Normalization, self).__init__()
        self.mean=torch.tensor(mean).view(-1,1,1)
        self.std=torch.tensor(std).view(-1,1,1)

    def forward(self,img):
        return (img-self.mean)/self.std



content_layer_defaults = ['conv_2', 'conv_4']
style_layer_defaults = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn,normalization_mean,normalization_std,
                               style_img,content_img,
                               content_layers=content_layer_defaults,style_layers=style_layer_defaults
                               ):
    print(content_layers,style_layers)
    cnn=copy.deepcopy(cnn)
    normalization=Normalization(normalization_mean,normalization_std)

    content_losses=[]
    style_losses=[]

    model=nn.Sequential(normalization)

    i=0
    for layer in cnn.children():
        if isinstance(layer,nn.Conv2d):
            i+=1
            name='conv_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            name='relu_{}'.format(i)
            layer=nn.ReLU(inplace=False)
        elif isinstance(layer,nn.MaxPool2d):
            name='pool_{}'.format(i)
        elif isinstance(layer,nn.BatchNorm2d):
            name='bn_{}'.format(i)
        else:
            raise RuntimeError("Unrecognised layer: {}".format(layer.__class__.__name__))

        model.add_module(name,layer)

        if name in content_layers:
            target=model(content_img).detach()
            content_loss=ContentLoss(target)
            model.add_module('content_loss_{}'.format(i),content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_features=model(style_img).detach()
            style_loss=StyleLoss(target_features)
            model.add_module('style_loss_{}'.format(i),style_loss)
            style_losses.append(style_loss)

    for i in range(len(model)-1,-1,-1):
        if isinstance(model[i],ContentLoss) or isinstance(model[i],StyleLoss):
            break;

    model=model[:(i+1)]

    return model, content_losses, style_losses






# input_img = content_img.clone()
# plt.figure()
# imshow(input_img,title="Input")

def get_input_optimizer(input_img):
    optimizer= optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn,normalization_mean,normalization_std,content_img, style_img, input_img,num_steps=300, style_weight=10000, content_weight=5,content_layer=content_layer_defaults,style_layer=style_layer_defaults):
    print("Building style transfer Model")

    model, content_losses, style_losses=get_style_model_and_losses(cnn,normalization_mean,normalization_std,style_img,content_img,content_layers=content_layer,style_layers=style_layer)
    optimizer=get_input_optimizer(input_img)

    print("Optimizing")
    run=[0]
    while run[0]<=num_steps:
        def closure():
            nonlocal  run
            input_img.data.clamp_(0,1)
            optimizer.zero_grad()

            model(input_img)
            style_score=0
            content_score=0

            for sl in style_losses:
                style_score+=sl.loss
            for cl in content_losses:
                content_score+=cl.loss

            style_score*=style_weight
            content_score*=content_weight

            loss=style_score+content_score
            loss.backward()
            run[0]+=1

            if run[0]%50==0:
                print("run ",run)
                print("Style loss : {:4f} Content loss: {:4f}".format(style_score.item(),content_score.item()))
                imshow(input_img)
            return style_score+content_score


        optimizer.step(closure)


    input_img.data.clamp_(0,1)
    plt.close('all')
    return input_img
if __name__=="__main__":
  content_layers=content_layer_defaults
  style_layers=style_layer_defaults
  style_img = image_loader("styles/3.jpg")
  content_img = image_loader("/home/chandragupta/study/data/my_faces/MY_FACE/00/face54.jpg")
  # input_img = torch.randn(content_img.data.size(), device=device)
  # input_img = content_img.clone()
  input_img = style_img.clone()
  output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                              content_img, style_img, input_img,content_layer=content_layers,style_layer=style_layers)
  plt.figure()
  imshow(output, title='Output Image')

