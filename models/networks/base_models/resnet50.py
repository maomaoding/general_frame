#coding:utf8
import torch as t
from torch import nn
import torch.nn.functional as F

channels=[64,128,256,512,1024,2048]
class RESNET(nn.Module):
    def __init__(self):
        super(RESNET, self).__init__()
        self.model_name="SCNN"

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resa_bn = nn.BatchNorm2d(channels[0])
        #self.conv_d = nn.Conv2d(channels[0], channels[0], (1, 9), padding=(0, 4), bias=False)
        #self.conv_u = nn.Conv2d(channels[0], channels[0], (1, 9), padding=(0, 4), bias=False)
        #self.conv_r = nn.Conv2d(channels[0], channels[0], (9, 1), padding=(4, 0), bias=False)
        #self.conv_l = nn.Conv2d(channels[0], channels[0], (9, 1), padding=(4, 0), bias=False)
        # res2---------------------------------------------
        # res2a
        self.res2a_branch1 = nn.Conv2d(channels[0], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2a_branch1 = nn.BatchNorm2d(channels[2])
        self.res2a_branch2a = nn.Conv2d(channels[0], channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2a_branch2a = nn.BatchNorm2d(channels[0])
        self.res2a_branch2b = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2a_branch2b = nn.BatchNorm2d(channels[0])
        self.res2a_branch2c = nn.Conv2d(channels[0], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        #self.bn2a_branch2c = nn.BatchNorm2d(channels[2])
        # res2b
        self.bn2b_bn = nn.BatchNorm2d(channels[2])
        self.res2b_branch2a = nn.Conv2d(channels[2], channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2b_branch2a = nn.BatchNorm2d(channels[0])
        self.res2b_branch2b = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2b_branch2b = nn.BatchNorm2d(channels[0])
        self.res2b_branch2c = nn.Conv2d(channels[0], channels[2], kernel_size=1, stride=1, padding=0, bias=False)

        # res2c
        self.bn2c_bn = nn.BatchNorm2d(channels[2])
        self.res2c_branch2a = nn.Conv2d(channels[2], channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2c_branch2a = nn.BatchNorm2d(channels[0])
        self.res2c_branch2b = nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2c_branch2b = nn.BatchNorm2d(channels[0])
        self.res2c_branch2c = nn.Conv2d(channels[0], channels[2], kernel_size=1, stride=1, padding=0, bias=False)

        # res3---------------------------------------------
        # res3a
        self.bn3a_bn = nn.BatchNorm2d(channels[2])
        self.res3a_branch1 = nn.Conv2d(channels[2], channels[3], kernel_size=1, stride=2, padding=0, bias=False)
        self.bn3a_branch1 = nn.BatchNorm2d(channels[3])
        self.res3a_branch2a = nn.Conv2d(channels[2], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3a_branch2a = nn.BatchNorm2d(channels[1])
        self.res3a_branch2b = nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3a_branch2b = nn.BatchNorm2d(channels[1])
        self.res3a_branch2c = nn.Conv2d(channels[1], channels[3], kernel_size=1, stride=1, padding=0, bias=False)

        # res3b
        self.bn3b_bn = nn.BatchNorm2d(channels[3])
        self.res3b_branch2a = nn.Conv2d(channels[3], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3b_branch2a = nn.BatchNorm2d(channels[1])
        self.res3b_branch2b = nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3b_branch2b = nn.BatchNorm2d(channels[1])
        self.res3b_branch2c = nn.Conv2d(channels[1], channels[3], kernel_size=1, stride=1, padding=0, bias=False)

        # res3c
        self.bn3c_bn = nn.BatchNorm2d(channels[3])
        self.res3c_branch2a = nn.Conv2d(channels[3], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3c_branch2a = nn.BatchNorm2d(channels[1])
        self.res3c_branch2b = nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3c_branch2b = nn.BatchNorm2d(channels[1])
        self.res3c_branch2c = nn.Conv2d(channels[1], channels[3], kernel_size=1, stride=1, padding=0, bias=False)

        # res3d
        self.bn3d_bn = nn.BatchNorm2d(channels[3])
        self.res3d_branch2a = nn.Conv2d(channels[3], channels[1], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3d_branch2a = nn.BatchNorm2d(channels[1])
        self.res3d_branch2b = nn.Conv2d(channels[1], channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3d_branch2b = nn.BatchNorm2d(channels[1])
        self.res3d_branch2c = nn.Conv2d(channels[1], channels[3], kernel_size=1, stride=1, padding=0, bias=False)

        # res4---------------------------------------------
        # res4a
        self.bn4a_bn = nn.BatchNorm2d(channels[3])
        self.res4a_branch1 = nn.Conv2d(channels[3], channels[4], kernel_size=1, stride=2, padding=0, bias=False)
        self.bn4a_branch1 = nn.BatchNorm2d(channels[4])
        self.res4a_branch2a = nn.Conv2d(channels[3], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4a_branch2a = nn.BatchNorm2d(channels[2])
        self.res4a_branch2b = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4a_branch2b = nn.BatchNorm2d(channels[2])
        self.res4a_branch2c = nn.Conv2d(channels[2], channels[4], kernel_size=1, stride=1, padding=0, bias=False)

        # res4b
        self.bn4b_bn = nn.BatchNorm2d(channels[4])
        self.res4b_branch2a = nn.Conv2d(channels[4], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4b_branch2a = nn.BatchNorm2d(channels[2])
        self.res4b_branch2b = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4b_branch2b = nn.BatchNorm2d(channels[2])
        self.res4b_branch2c = nn.Conv2d(channels[2], channels[4], kernel_size=1, stride=1, padding=0, bias=False)

        # res4c
        self.bn4c_bn = nn.BatchNorm2d(channels[4])
        self.res4c_branch2a = nn.Conv2d(channels[4], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4c_branch2a = nn.BatchNorm2d(channels[2])
        self.res4c_branch2b = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4c_branch2b = nn.BatchNorm2d(channels[2])
        self.res4c_branch2c = nn.Conv2d(channels[2], channels[4], kernel_size=1, stride=1, padding=0, bias=False)

        # res4d
        self.bn4d_bn = nn.BatchNorm2d(channels[4])
        self.res4d_branch2a = nn.Conv2d(channels[4], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4d_branch2a = nn.BatchNorm2d(channels[2])
        self.res4d_branch2b = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4d_branch2b = nn.BatchNorm2d(channels[2])
        self.res4d_branch2c = nn.Conv2d(channels[2], channels[4], kernel_size=1, stride=1, padding=0, bias=False)

        # res4e
        self.bn4e_bn = nn.BatchNorm2d(channels[4])
        self.res4e_branch2a = nn.Conv2d(channels[4], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4e_branch2a = nn.BatchNorm2d(channels[2])
        self.res4e_branch2b = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4e_branch2b = nn.BatchNorm2d(channels[2])
        self.res4e_branch2c = nn.Conv2d(channels[2], channels[4], kernel_size=1, stride=1, padding=0, bias=False)

        # res4f
        self.bn4f_bn = nn.BatchNorm2d(channels[4])
        self.res4f_branch2a = nn.Conv2d(channels[4], channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4f_branch2a = nn.BatchNorm2d(channels[2])
        self.res4f_branch2b = nn.Conv2d(channels[2], channels[2], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4f_branch2b = nn.BatchNorm2d(channels[2])
        self.res4f_branch2c = nn.Conv2d(channels[2], channels[4], kernel_size=1, stride=1, padding=0, bias=False)

        # res5---------------------------------------------
        # res5a
        self.bn5a_bn = nn.BatchNorm2d(channels[4])
        self.res5a_branch1 = nn.Conv2d(channels[4], channels[5], kernel_size=1, stride=2, padding=0, bias=False)
        self.bn5a_branch1 = nn.BatchNorm2d(channels[5])
        self.res5a_branch2a = nn.Conv2d(channels[4], channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5a_branch2a = nn.BatchNorm2d(channels[3])
        self.res5a_branch2b = nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=2, padding=2, dilation=2,
                                        bias=False)
        self.bn5a_branch2b = nn.BatchNorm2d(channels[3])

        self.res5a_branch2c = nn.Conv2d(channels[3], channels[5], kernel_size=1, stride=1, padding=0,bias=False)


        # res5b
        self.bn5b_bn = nn.BatchNorm2d(channels[5])
        self.res5b_branch2a = nn.Conv2d(channels[5], channels[3], kernel_size=1, stride=1, padding=0,bias=False)
        self.bn5b_branch2a = nn.BatchNorm2d(channels[3])
        self.res5b_branch2b = nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=2, dilation=2,
                                        bias=False)
        self.bn5b_branch2b = nn.BatchNorm2d(channels[3])
        self.res5b_branch2c = nn.Conv2d(channels[3], channels[5], kernel_size=1, stride=1, padding=0, bias=False)

        # res5c
        self.bn5c_bn = nn.BatchNorm2d(channels[5])
        self.res5c_branch2a = nn.Conv2d(channels[5], channels[3], kernel_size=1, stride=1, padding=0,bias=False)
        self.bn5c_branch2a = nn.BatchNorm2d(channels[3])
        self.res5c_branch2b = nn.Conv2d(channels[3], channels[3], kernel_size=3, stride=1, padding=2, dilation=2,bias=False)
        self.bn5c_branch2b = nn.BatchNorm2d(channels[3])
        self.res5c_branch2c = nn.Conv2d(channels[3], channels[5], kernel_size=1, stride=1, padding=0, bias=False)


        if self.training:
            self.dropout = nn.Dropout2d(0.3)

        self.conv8 = nn.Conv2d(channels[5], 5, 1)
        # self.conv8.weight.requires_grad = False
        # self.conv8.bias.requires_grad = False

        self.fc9 = nn.Linear(5 * 9 * 25, 128)
        self.fc10 = nn.Linear(128, 4)

        # init weight and bias
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         torch.nn.init.xavier_normal_(m.weight, gain=1)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.normal_(0, 1)
        #         m.bias.data.fill_(0)
        #     elif isinstance(m, nn.ConvTranspose2d):
        #         assert (m.weight.data.shape[2] == m.weight.data.shape[3])
        #         f = math.ceil(m.weight.data.shape[3] / 2)
        #         c = (2 * f - 1 - f % 2) / (2 * f)
        #         for i in range(m.weight.data.shape[2]):
        #             for j in range(m.weight.data.shape[3]):
        #                 m.weight.data[:, :, i, j] = (1 - abs(i / f - c)) * (1 - abs(j / f - c))
        #         m.bias.data.fill_(0)
        #
        # nn.init.normal_(self.conv_d.weight, mean=0, std=math.sqrt(2.5 / self.conv_d.weight.numel()))
        # nn.init.normal_(self.conv_u.weight, mean=0, std=math.sqrt(2.5 / self.conv_u.weight.numel()))
        # nn.init.normal_(self.conv_r.weight, mean=0, std=math.sqrt(2.5 / self.conv_r.weight.numel()))
        # nn.init.normal_(self.conv_l.weight, mean=0, std=math.sqrt(2.5 / self.conv_l.weight.numel()))

    def forward(self, x):
        skip_connections=[[],[],[],[],[],[],[],[]]
        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)
        skip_connections[3] = x
        x = self.maxpool1(x)

        x=self.resa_bn(x)

        x=self.relu(x)
        '''for i in range(1, x.shape[2]):
            x[..., i:i + 1, :].add_(F.relu(self.conv_d(x[..., i - 1:i, :])))

        for i in range(x.shape[2] - 2, 0, -1):
            x[..., i:i + 1, :].add_(F.relu(self.conv_u(x[..., i + 1:i + 2, :])))

        for i in range(1, x.shape[3]):
            x[..., i:i + 1].add_(F.relu(self.conv_r(x[..., i - 1:i])))
        for i in range(x.shape[3] - 2, 0, -1):
            x[..., i:i + 1].add_(F.relu(self.conv_l(x[..., i + 1:i + 2])))'''
        # res2---------------------------------------------
        # res2a
        x_2a_b1 = self.res2a_branch1(x)
        #x_2a_b1 = self.bn2a_branch1(x_2a_b1)
        x_2a_b2 = self.res2a_branch2a(x)
        x_2a_b2 = self.bn2a_branch2a(x_2a_b2)
        x_2a_b2 = self.relu(x_2a_b2)
        x_2a_b2 = self.res2a_branch2b(x_2a_b2)
        x_2a_b2 = self.bn2a_branch2b(x_2a_b2)
        x_2a_b2 = self.relu(x_2a_b2)
        x_2a_b2 = self.res2a_branch2c(x_2a_b2)
        #x_2a_b2 = self.bn2a_branch2c(x_2a_b2)
        x = t.add(x_2a_b1, x_2a_b2)
        #x = self.relu(x)
        # res2b
        x_2b = self.bn2b_bn(x)
        x_2b=self.relu(x_2b)
        skip_connections[7] = x
        x_2b = self.res2b_branch2a(x_2b)
        x_2b = self.bn2b_branch2a(x_2b)
        x_2b = self.relu(x_2b)
        x_2b = self.res2b_branch2b(x_2b)
        x_2b = self.bn2b_branch2b(x_2b)
        x_2b = self.relu(x_2b)
        x_2b = self.res2b_branch2c(x_2b)
        x = t.add(x, x_2b)
        #x = self.relu(x)
        # res2c
        x_2c = self.bn2c_bn(x)
        x_2c = self.relu(x_2c)
        x_2c = self.res2c_branch2a(x_2c)
        x_2c = self.bn2c_branch2a(x_2c)
        x_2c = self.relu(x_2c)
        x_2c = self.res2c_branch2b(x_2c)
        x_2c = self.bn2c_branch2b(x_2c)
        x_2c = self.relu(x_2c)
        x_2c = self.res2c_branch2c(x_2c)
        #x_2c = self.bn2c_branch2c(x_2c)
        x = t.add(x, x_2c)
        x = self.bn3a_bn(x)
        x = self.relu(x)
        skip_connections[2] = x
        # res3---------------------------------------------
        # res3a
        x_3a_b1 = self.res3a_branch1(x)
        #x_3a_b1 = self.bn3a_branch1(x_3a_b1)
        x_3a_b2 = self.res3a_branch2a(x)
        x_3a_b2 = self.bn3a_branch2a(x_3a_b2)
        x_3a_b2 = self.relu(x_3a_b2)
        x_3a_b2 = self.res3a_branch2b(x_3a_b2)
        x_3a_b2 = self.bn3a_branch2b(x_3a_b2)
        x_3a_b2 = self.relu(x_3a_b2)
        x_3a_b2 = self.res3a_branch2c(x_3a_b2)
        x = t.add(x_3a_b1, x_3a_b2)


        # res3b
        x_3b = self.bn3b_bn(x)
        x_3b = self.relu(x_3b)
        skip_connections[6] = x
        x_3b = self.res3b_branch2a(x_3b)
        x_3b = self.bn3b_branch2a(x_3b)
        x_3b = self.relu(x_3b)
        x_3b = self.res3b_branch2b(x_3b)
        x_3b = self.bn3b_branch2b(x_3b)
        x_3b = self.relu(x_3b)
        x_3b = self.res3b_branch2c(x_3b)

        x = t.add(x, x_3b)

        # res3c
        x_3c = self.bn3c_bn(x)
        x_3c = self.relu(x_3c)
        x_3c = self.res3c_branch2a(x_3c)
        x_3c = self.bn3c_branch2a(x_3c)
        x_3c = self.relu(x_3c)
        x_3c = self.res3c_branch2b(x_3c)
        x_3c = self.bn3c_branch2b(x_3c)
        x_3c = self.relu(x_3c)
        x_3c = self.res3c_branch2c(x_3c)

        x = t.add(x, x_3c)


        # res3d
        x_3d = self.bn3c_bn(x)
        x_3d = self.relu(x_3d)
        x_3d = self.res3c_branch2a(x_3d)
        x_3d = self.bn3c_branch2a(x_3d)
        x_3d = self.relu(x_3d)
        x_3d = self.res3c_branch2b(x_3d)
        x_3d = self.bn3c_branch2b(x_3d)
        x_3d = self.relu(x_3d)
        x_3d = self.res3c_branch2c(x_3d)

        x = t.add(x, x_3d)
        x = self.bn4a_bn(x)
        x = self.relu(x)
        skip_connections[1]=x
        # res4---------------------------------------------
        # res4a
        x_4a_b1 = self.res4a_branch1(x)
        #x_4a_b1 = self.bn4a_branch1(x_4a_b1)
        x_4a_b2 = self.res4a_branch2a(x)
        x_4a_b2 = self.bn4a_branch2a(x_4a_b2)
        x_4a_b2 = self.relu(x_4a_b2)
        x_4a_b2 = self.res4a_branch2b(x_4a_b2)
        x_4a_b2 = self.bn4a_branch2b(x_4a_b2)
        x_4a_b2 = self.relu(x_4a_b2)
        x_4a_b2 = self.res4a_branch2c(x_4a_b2)
        x = t.add(x_4a_b1, x_4a_b2)

        # res4b
        x_4b = self.bn4b_bn(x)
        x_4b = self.relu(x_4b)
        skip_connections[5] = x
        x_4b = self.res4b_branch2a(x_4b)
        x_4b = self.bn4b_branch2a(x_4b)
        x_4b = self.relu(x_4b)
        x_4b = self.res4b_branch2b(x_4b)
        x_4b = self.bn4b_branch2b(x_4b)
        x_4b = self.relu(x_4b)
        x_4b = self.res4b_branch2c(x_4b)

        x = t.add(x, x_4b)

        # res4c
        x_4c = self.bn4c_bn(x)
        x_4c = self.relu(x_4c)
        x_4c = self.res4c_branch2a(x_4c)
        x_4c = self.bn4c_branch2a(x_4c)
        x_4c = self.relu(x_4c)
        x_4c = self.res4c_branch2b(x_4c)
        x_4c = self.bn4c_branch2b(x_4c)
        x_4c = self.relu(x_4c)
        x_4c = self.res4c_branch2c(x_4c)

        x = t.add(x, x_4c)

        # res4d
        x_4d = self.bn4d_bn(x)
        x_4d = self.relu(x_4d)
        x_4d = self.res4d_branch2a(x_4d)
        x_4d = self.bn4d_branch2a(x_4d)
        x_4d = self.relu(x_4d)
        x_4d = self.res4d_branch2b(x_4d)
        x_4d = self.bn4d_branch2b(x_4d)
        x_4d = self.relu(x_4d)
        x_4d = self.res4d_branch2c(x_4d)

        x = t.add(x, x_4d)

        # res4e
        x_4e = self.bn4e_bn(x)
        x_4e = self.relu(x_4e)
        x_4e = self.res4e_branch2a(x_4e)
        x_4e = self.bn4e_branch2a(x_4e)
        x_4e = self.relu(x_4e)
        x_4e = self.res4e_branch2b(x_4e)
        x_4e = self.bn4e_branch2b(x_4e)
        x_4e = self.relu(x_4e)
        x_4e = self.res4e_branch2c(x_4e)

        x = t.add(x, x_4e)

        # res4f
        x_4f = self.bn4f_bn(x)
        x_4f = self.relu(x_4f)
        x_4f = self.res4f_branch2a(x)
        x_4f = self.bn4f_branch2a(x_4f)
        x_4f = self.relu(x_4f)
        x_4f = self.res4f_branch2b(x_4f)
        x_4f = self.bn4f_branch2b(x_4f)
        x_4f = self.relu(x_4f)
        x_4f = self.res4f_branch2c(x_4f)

        x = t.add(x, x_4f)
        x = self.bn5a_bn(x)
        x = self.relu(x)
        skip_connections[0] = x
        # res5---------------------------------------------
        # res5a
        x_5a_b1 = self.res5a_branch1(x)
        #x_5a_b1 = self.bn5a_branch1(x_5a_b1)
        x_5a_b2 = self.res5a_branch2a(x)
        x_5a_b2 = self.bn5a_branch2a(x_5a_b2)
        x_5a_b2 = self.relu(x_5a_b2)
        x_5a_b2 = self.res5a_branch2b(x_5a_b2)
        x_5a_b2 = self.bn5a_branch2b(x_5a_b2)
        x_5a_b2 = self.relu(x_5a_b2)
        x_5a_b2 = self.res5a_branch2c(x_5a_b2)

        x = t.add(x_5a_b1, x_5a_b2)


        # res5b
        x_5b = self.bn5b_bn(x)
        x_5b = self.relu(x_5b)
        x_5b = self.res5b_branch2a(x_5b)
        x_5b = self.bn5b_branch2a(x_5b)
        x_5b = self.relu(x_5b)
        x_5b = self.res5b_branch2b(x_5b)
        x_5b = self.bn5b_branch2b(x_5b)
        x_5b = self.relu(x_5b)
        x_5b = self.res5b_branch2c(x_5b)

        x = t.add(x, x_5b)

        # res5c
        x_5c = self.bn5c_bn(x)
        x_5c = self.relu(x_5c)
        x_5c = self.res5c_branch2a(x_5c)
        x_5c = self.bn5c_branch2a(x_5c)
        x_5c = self.relu(x_5c)
        x_5c = self.res5c_branch2b(x_5c)
        x_5c = self.bn5c_branch2b(x_5c)
        x_5c = self.relu(x_5c)
        x_5c = self.res5c_branch2c(x_5c)

        x = t.add(x, x_5c)
        x = self.relu(x)

        skip_connections[4] = x

        return skip_connections
        '''if self.training:
            x = self.dropout(x)
        # x = F.dropout2d(x, p=0.1, training=self.training)

        x = self.conv8(x)
        x1 = F.interpolate(x, size=[288, 800], mode='bilinear', align_corners=False)
        # x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x, dim=1)
        #x2 = F.avg_pool2d(x2, 2, stride=2, padding=0)
        x2 = x2.view(-1, x2.numel() // x2.shape[0])
        x2 = self.fc9(x2)
        x2 = F.relu(x2)
        x2 = self.fc10(x2)
        x2 = t.sigmoid(x2)
        print(skip_connections[0])
        return x1, x2'''
