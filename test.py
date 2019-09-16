from train import ToTensor, Color_Generator_Rep
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time

CHECKPOINTS_DIR='./Models/Main/'
Test_DIR = './Test_Images/'
result_dir = './Test_Results/'

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

device = 'cuda:0'        

ch = 1

color_net = Color_Generator_Rep()
checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,"netG.pt"))
color_net.load_state_dict(checkpoint['model_state_dict'])
color_net.eval()
color_net.to(device)

if __name__ =='__main__':

    st = time.time()

    m = "0020.png"

    img=cv2.imread(Test_DIR + str(m))

    img = img.astype(np.float32)
    h,w,c=img.shape

    img=img/255.0

    train_x = np.zeros((1, ch, h, w)).astype(np.float32)

    train_x[0,0,:,:] = img[:,:,0]

    Y_channel = img*255.0

    dataset_torchx = torch.from_numpy(train_x)

    dataset_torchx=dataset_torchx.to(device)

    output=color_net(dataset_torchx)

    output=output*255.0
    output = output.cpu()
    a=output.detach().numpy()

    res = a[0,:,:,:].transpose((1, 2, 0))

    print(res.shape)

    result = cv2.merge((np.uint8(Y_channel[:,:,0]), np.uint8(res[:,:,0]), np.uint8(res[:,:,1])))

    result = cv2.cvtColor(result, cv2.COLOR_YCR_CB2BGR)

    final_image = np.uint8(result)

    cv2.imwrite(result_dir + str(m),final_image)
    print(' 		{')
    print(' 			saved image ', str(m), ' at ', str(os.path.dirname(result_dir)))
    print(' 			image height ', str(final_image.shape[1]))
    print(' 			image width ', str(final_image.shape[0]))
    print(' 		}\n')
    
    end = time.time()
    print(' 		Total time taken in secs : '+str(end-st))
