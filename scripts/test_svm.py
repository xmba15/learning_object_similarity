#!/usr/bin/env python
# -*- coding: utf-8 -*-


from triplet_network import TripletNetWork
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import model_net

from new_augmentation import SquareZeroPadding
from config import Config
from utils import imshow, show_plot
import torch.nn.functional as F
import torchvision

from PIL import Image

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

import torchvision.transforms as transforms
_preprocess = transforms.Compose([
    SquareZeroPadding(),
    transforms.Resize((224, 224), 2),
    transforms.ToTensor(),
    normalize
])

def load_image(img_path):

    img = Image.open(img_path)
    img = _preprocess(img)

    return img.unsqueeze(0)

if __name__=="__main__":

    # Net = model_net.ResnetBased
    # model = Net(normalize = True)
    # tnet = TripletNetWork(model).cuda()
    # tnet.eval()
    # tnet.load_state_dict(torch.load(Config.model_dir + "/resnet_fc256__triplet_3.pth"))

    tnet = torchvision.models.resnet50(pretrained = True).cuda()
    tnet.eval()
    import time

    start = time.time()
    # img_path_list = [Config.image_dir + "/roi_" + str(i) + ".jpg" for i in range(9)]

    img_path_list = [Config.image_dir + "/n1.png", Config.image_dir + "/apple.jpg"]
    # img_path_list = [Config.image_dir + "/fries.jpg"]    
    
    
    img_list = [load_image(img_path) for img_path in img_path_list]
    feature_list = []
    
    instance_img_path = Config.image_dir + "/n2.png"
    instance_img_torch = Variable(load_image(instance_img_path)).cuda()
    instance_output = tnet(instance_img_torch)
    # instance_output, _, _ = tnet(instance_img_torch, instance_img_torch, instance_img_torch)    
    instance_output = F.normalize(instance_output, p = 2, dim = 1)
    
    for i, img in enumerate(img_list):
        img_torch = Variable(img).cuda()
        output = tnet(img_torch)
        # output, _, _ = tnet(img_torch, img_torch, img_torch)
        output = F.normalize(output, p = 2, dim = 1)
        euclidean_distance = F.pairwise_distance(output, instance_output)
        print euclidean_distance
        # feature_list.append(output.cpu().data.numpy())
        
    print time.time() - start
