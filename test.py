
import time
import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from math import sqrt
import numpy as np
import argparse,os
import pdb
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import cv2
import torch.utils.data as data 
import matplotlib.pyplot as plt
import model_224
from model_224 import *
import sys
device = model_224.device

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

dataset = 'CEERI_test_one_vid'
pwd = os.getcwd()
result_save_dir = os.path.join(pwd,'results_test','maps')
if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)

class Args:
  cuda = 'store_true'
  resume = 'model.pth'
  pretrained = ''

def get_n_depth_image_x(image_path,videoname):
    frames_total = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
    image_id = np.random.randint(0, frames_total-10)
    img_ret, map_ret = [], []
    for i in range(10):
        map_x = np.zeros((32, 32))
        image_name =  videoname.split('/')[-1]+'_'+str(image_id+i).zfill(4) + '_face.png'

        
        # RGB
        image_name = os.path.join(image_path, image_name)
        image_x_temp = cv2.imread(image_name)
        image_name=''
        img_ret.append((cv2.resize(image_x_temp, (224, 224)))) 

    return np.array(img_ret)

def get_sample(root_dir, videoname):
    image_path = os.path.join(root_dir, videoname)

    image_x = get_n_depth_image_x(image_path, videoname)
    new_image_x  =[]

    for img in image_x:
        new_image_x.append((img - 127.5)/128 )    # [-1,1]
    image_x = np.array(new_image_x)[:,:,:,::-1].transpose((3, 0, 1, 2))
    sample = torch.from_numpy(np.expand_dims(image_x,axis=0).astype(np.float)).float()
    return sample
  
def FeatureMap2Heatmap( x, feature1, feature2, feature3,feature4, map_x,label,phase,epoch):
    ## initial images 
    feature_first_frame = x[0,:,1,:,:].cpu()    ## the middle frame
    # result_save_dir += phase+'/' 
    if not os.path.exists(os.path.join(result_save_dir,phase,str(epoch))):
        os.makedirs(os.path.join(result_save_dir,phase,str(epoch)))

    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    results_save = os.path.join(result_save_dir,phase,str(epoch))
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(results_save+'/'+ '_x_visual.jpg')
    plt.close()


    ## first feature
    feature_first_frame = feature1[0,1,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(results_save+'/'+ '_x_Block1_visual.jpg')
    plt.close()
    
    ## second feature
    feature_first_frame = feature2[0,1,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(results_save+'/'+'_x_Block2_visual.jpg')
    plt.close()
    
    ## third feature
    feature_first_frame = feature3[0,1,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(results_save+'/'+ '_x_Block3_visual.jpg')
    plt.close()
    
        ## fourth feature
    feature_first_frame = feature4[0,1,:,:,:].cpu()    ## the middle frame 
    heatmap = torch.zeros(feature_first_frame.size(1), feature_first_frame.size(2))
    for i in range(feature_first_frame.size(0)):
        heatmap += torch.pow(feature_first_frame[i,:,:],2).view(feature_first_frame.size(1),feature_first_frame.size(2))

    heatmap = heatmap.data.numpy()
    
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    plt.imshow(heatmap)
    plt.colorbar()
    plt.savefig(results_save+'/'+ '_x_Block4_visual.jpg')
    plt.close()


    ## third feature
    for ii in range(map_x.shape[0]):
        heatmap2 = torch.pow(map_x[ii,:,:],2)    ## the middle frame 
        heatmap2 = heatmap2.data.cpu().numpy()        
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        plt.imshow(map_x[ii,:,:].data.cpu().numpy())
        plt.colorbar()
        plt.savefig(results_save+'/'+ str(ii)+'_x_DepthMav_visual.jpg')
        plt.close()
    plt.close('all')


# args=Args()
def main():
    global opt, model,beta,samples_per_cls,nsamples_per_cls

    opt = Args()
    cuda = device
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print('===> Building model')

    model = get_model(sample_size = 224, sample_duration = 16, num_classes=2)


    print("The model has : '{}' Parameters ".format(count_parameters(model)))

    samples_per_cls = 2
    
    print('===> Setting GPU')
    model = model.to(device)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume,map_location = 'cpu')   
            model.load_state_dict(checkpoint['model'].state_dict())
        else:

            print("=> no checkpoint found at '{}'".format(opt.resume))
            sys.exit()

    if test(model)==0:
        print('Predicted Result : Spoof')
    else:
        print('Predicted Result : Live')
    

def test(model):
    model.eval()
    print("====================Testing =================>") 
    root_dir, videoname = '','057_A9E4_VA01_02_01'
    input = get_sample(root_dir, videoname)
    input =input.to(device)
    with torch.no_grad():
        # features,
        pred,map_pred,x_Block1_32x32,x_Block2_32x32,x_Block3_32x32,x_Block4_32x32 = model(input)

    predictions = pred.softmax(dim = 1)
    predictions = torch.argmax(input=pred)
    predictions  = predictions.type(torch.int32)
    
    coun = 1
    for map_ in map_pred.data.cpu().numpy():
        write_path = os.path.join(result_save_dir,'pred_maps')
        if not os.path.exists(write_path) :
            os.makedirs(write_path)
        cv2.imwrite(os.path.join(write_path,str(coun)+'_'+str(predictions.cpu().detach().numpy())+'.png'),map_*255)
        coun+=1
    
    FeatureMap2Heatmap(input,x_Block1_32x32,x_Block2_32x32,x_Block3_32x32,x_Block4_32x32, map_pred,predictions.cpu().detach().numpy(),'Test',0)
    return predictions.cpu().detach().numpy()
   



if __name__ == '__main__':
    main()
