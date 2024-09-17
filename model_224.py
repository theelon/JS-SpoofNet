
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
device = torch.device("cpu")
class SA(nn.Module):
    """
    input:N*C*D*H*W
    """
    def __init__(self,features_=512,name='st'):#, in_put
        super().__init__()
        # self.N = 32#in_put[0] 
        # self.C = 1024#in_put[1]
        # self.D = 1#in_put[2]
        # self.H = 7#in_put[3]
        # self.W = 7#in_put[4]
        self.gama = nn.Parameter(torch.tensor([0.0]))
        # self.gama_s = nn.Parameter(torch.tensor([0.0]))
        self.sp_ = name
        self.in_ch = features_#1920(201) 1024(121)
        self.out_ch = features_#1920
        
        # self.conv3d_3 = nn.Sequential(
        #     # Conv3d input:N*C*D*H*W
        #     # Conv3d output:N*C*D*H*W
        #     nn.Conv3d(in_channels=self.in_ch, out_channels=self.out_ch,kernel_size=(3,3,3), padding=1),
        #     nn.BatchNorm3d(self.out_ch),
        #     nn.ReLU(inplace=True),
        # )
        self.compress = 8
        self.conv3d_d = nn.Sequential(
        # Conv3d input:N*C*D*H*W
        # Conv3d output:N*C*D*H*W
        nn.Conv3d(in_channels=features_//self.compress, out_channels=self.out_ch,kernel_size=(3,3,3), padding=1),
        nn.BatchNorm3d(self.out_ch),
        nn.ReLU(inplace=True),
        )

        self.conv3d_s = nn.Sequential(
        # Conv3d input:N*C*D*H*W
        # Conv3d output:N*C*D*H*W
        nn.Conv3d(in_channels=features_//self.compress, out_channels=self.out_ch,kernel_size=(3,3,3), padding=1),
        nn.BatchNorm3d(self.out_ch),
        nn.ReLU(inplace=True),
        )

        self.conv3d_v = nn.Sequential(
        # Conv3d input:N*C*D*H*W
        # Conv3d output:N*C*D*H*W
        nn.Conv3d(in_channels=self.in_ch, out_channels=features_//self.compress, kernel_size=(1, 1, 1)),
        nn.BatchNorm3d(features_//self.compress),
        nn.ReLU(inplace=True), 
        )
        self.conv3d_k = nn.Sequential(
        # Conv3d input:N*C*D*H*W
        # Conv3d output:N*C*D*H*W
        nn.Conv3d(in_channels=self.in_ch, out_channels=features_//self.compress, kernel_size=(1, 1, 1)),
        nn.BatchNorm3d(features_//self.compress),
        nn.ReLU(inplace=True), 
        )
        self.conv3d_q = nn.Sequential(
        # Conv3d input:N*C*D*H*W
        # Conv3d output:N*C*D*H*W
        nn.Conv3d(in_channels=self.in_ch, out_channels=features_//self.compress, kernel_size=(1, 1, 1)),
        nn.BatchNorm3d(features_//self.compress),
        nn.ReLU(inplace=True), 
        )

        # self.conv3d_h = nn.Sequential(
        #     # Conv3d input:N*C*D*H*W
        #     # Conv3d output:N*C*D*H*W
        #     nn.Conv3d(in_channels=self.in_ch, out_channels=self.in_ch, kernel_size=(1, 1, 1)),
        #     nn.BatchNorm3d(self.out_ch),
        #     nn.ReLU(inplace=True), 
        # )


    @classmethod
    def Cal_Patt(cls, k_x, q_x, v_x, N, C, D, H, W):
        """
        input:N*C*D*H*W
        """
        k_x_flatten = k_x.reshape((N, C, D, 1, H * W))
        q_x_flatten = q_x.reshape((N, C, D, 1, H * W))
        v_x_flatten = v_x.reshape((N, C, D, 1, H * W))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1,2,4,3), k_x_flatten)
        r_x = F.softmax(sigma_x, dim=4)
        # print(r_x.shape)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Patt = torch.matmul(v_x_flatten, r_x).reshape(N, C, D, H, W)
        return Patt
    
    @classmethod
    def Cal_Datt(cls, k_x, q_x, v_x, N, C, D, H, W):
        """
        input:N*C*D*H*W
        """
        # k_x_transpose = k_x.permute(0, 1, 3, 4, 2)
        # q_x_transpose = q_x.permute(0, 1, 3, 4, 2)
        # v_x_transpose = v_x.permute(0, 1, 3, 4, 2)
        k_x_flatten = k_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        q_x_flatten = q_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        v_x_flatten = v_x.permute(0, 1, 3, 4, 2).reshape((N, C, H, W, 1, D))
        sigma_x = torch.mul(q_x_flatten.permute(0, 1, 2, 3, 5, 4), k_x_flatten)
        r_x = F.softmax(sigma_x, dim=5)
        # print("r_x----------------------",r_x.shape)
        # r_x = F.softmax(sigma_x.float(), dim=4)
        Datt = torch.matmul(v_x_flatten, r_x).reshape(N, C, H, W, D)
        # print("Datt----------------------",Datt.shape)
        return Datt.permute(0, 1, 4, 2, 3)

   
    def forward(self, x):
        v_x = self.conv3d_v(x)
        k_x = self.conv3d_k(x)
        # print("k_x",k_x.shape[0])
        q_x = self.conv3d_q(x)
        if self.sp_ == "s":
            Patt = self.Cal_Patt(k_x, q_x, v_x, k_x.shape[0], k_x.shape[1],k_x.shape[2], k_x.shape[3], k_x.shape[4])
            Patt_c = self.conv3d_s(Patt)
            Y = self.gama*Patt_c + x
        elif self.sp_ == "t":
            Datt = self.Cal_Datt(k_x, q_x, v_x, k_x.shape[0], k_x.shape[1],k_x.shape[2], k_x.shape[3], k_x.shape[4])
            Datt_c = self.conv3d_d(Datt)
            Y = self.gama*Datt_c + x
        elif self.sp_ == "st":
            Patt = self.Cal_Patt(k_x, q_x, v_x, k_x.shape[0], k_x.shape[1],k_x.shape[2], k_x.shape[3], k_x.shape[4])
            Patt_c = self.conv3d_s(Patt)
            Datt = self.Cal_Datt(k_x, q_x, v_x, k_x.shape[0], k_x.shape[1],k_x.shape[2], k_x.shape[3], k_x.shape[4])
            Datt_c = self.conv3d_d(Datt)
            Y = self.gama*(Datt_c+Patt_c) + x
            
            # Y = self.gama*Datt + x
        # print("gama",self.gama.cpu().detach().numpy()[0])
        # write_SA_Lrate("{}\n ".format(self.gama.cpu().detach().numpy()[0]))
        return Y


class C3D(nn.Module):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=600):

        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        self.group_map1 = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU())          
        self.group_map2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU())
        self.group_map3 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU())
        self.group_map4 = nn.Sequential(
            nn.Conv3d(512, 128, kernel_size=1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU())

        self.lastconv1 = nn.Sequential(
            nn.Conv3d(128*4, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 1, kernel_size=3, padding=1),
            nn.BatchNorm3d(1),
            nn.ReLU())

        self.attn1 = SA(features_=128)
        self.attn2 = SA(features_=256)
        self.attn3 = SA(features_=512)
        self.attn4 = SA(features_=512)
        self.downsample32x32 = nn.Upsample(size=(1,32, 32), mode='trilinear',align_corners=True)

        last_duration = int(math.floor(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.fc1 = nn.Sequential(
            nn.Linear((32768) , 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(4096, num_classes))         
        self.attn = SA(features_=512)
        

    def forward(self, x):
        out = self.group1(x)
        # print("group1",x.shape,out.shape)
        out = self.group2(out)
        # print("group2",out.shape)
        x_Block1_32x32 = self.downsample32x32(self.group_map1(out))
        # print("x_Block1_32x32",x_Block1_32x32.shape)
        out = self.group3(out)
        # print("group3",out.shape)
        x_Block2_32x32 = self.downsample32x32(self.group_map2(out))
        # print("x_Block2_32x32",x_Block2_32x32.shape)
        out = self.group4(out)
        # print("group4",out.shape)
        x_Block3_32x32 = self.downsample32x32(self.group_map3(out)) 
        # print("x_Block3_32x32",x_Block3_32x32.shape)
        out = self.group5(out)
        # print("group5",out.shape)
        out  = self.attn(out)
        x_Block4_32x32 = self.downsample32x32(self.group_map4(out))
        # print("x_Block4_32x32",x_Block4_32x32.shape)
        x_concat = torch.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32,x_Block4_32x32), dim=1)
        # print("x_concat",x_concat.shape)
        map_x = self.lastconv1(x_concat)
        # print("map_x",map_x.shape)
        # print(out.shape,'>>>>>>>>>>>>')
        out = out.view(out.size(0), -1)
        # print(out.shape,'>>>>>>>>>>>>')
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out,(map_x.squeeze(1)).squeeze(1),x_Block1_32x32,x_Block2_32x32,x_Block3_32x32,x_Block4_32x32


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = C3D(**kwargs)
    return model


if __name__ == '__main__':
    model = get_model(sample_size = 112, sample_duration = 16, num_classes=2)
    model = model
    # model = model.cuda()
    # model = nn.DataParallel(model, device_ids=None)
    # print(model)

    input_var = Variable(torch.randn(8, 3, 10, 224, 224))
    output = model(input_var)
    print(output.shape)