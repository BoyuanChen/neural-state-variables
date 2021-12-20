
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def conv2d_bn_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer

def conv2d_bn_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.Conv2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_sigmoid(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.Sigmoid()
    )
    return convlayer

def deconv_relu(inch,outch,kernel_size,stride=1,padding=1):
    convlayer = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(inch,outch,kernel_size=kernel_size,stride=stride,padding=padding),
        torch.nn.BatchNorm2d(outch),
        torch.nn.ReLU()
    )
    return convlayer



class EncoderDecoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(EncoderDecoder,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=2),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,128,4,stride=2),
            conv2d_bn_relu(128,128,3),
        )
        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(128,128,(3,4),stride=(1,2)),
            conv2d_bn_relu(128,128,3),
        )
        
        self.deconv_5 = deconv_relu(128,64,(3,4),stride=(1,2))
        self.deconv_4 = deconv_relu(67,64,4,stride=2)
        self.deconv_3 = deconv_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_relu(35,16,4,stride=2)
        self.deconv_1 = deconv_sigmoid(19,3,4,stride=2)

        self.predict_5 = torch.nn.Conv2d(128,3,3,stride=1,padding=1)
        self.predict_4 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,(3,4),stride=(1,2),padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )


    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        return conv5_out

    def decoder(self, x):
        deconv5_out = self.deconv_5(x)
        predict_5_out = self.up_sample_5(self.predict_5(x))

        concat_5 = torch.cat([deconv5_out, predict_5_out],dim=1)
        deconv4_out = self.deconv_4(concat_5)
        predict_4_out = self.up_sample_4(self.predict_4(concat_5))

        concat_4 = torch.cat([deconv4_out,predict_4_out],dim=1)
        deconv3_out = self.deconv_3(concat_4)
        predict_3_out = self.up_sample_3(self.predict_3(concat_4))

        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)
        return predict_out
        

    def forward(self,x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out, latent

class EncoderDecoder64x1x1(torch.nn.Module):
    def __init__(self, in_channels):
        super(EncoderDecoder64x1x1,self).__init__()

        self.conv_stack1 = torch.nn.Sequential(
            conv2d_bn_relu(in_channels,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack2 = torch.nn.Sequential(
            conv2d_bn_relu(32,32,4,stride=2),
            conv2d_bn_relu(32,32,3)
        )
        self.conv_stack3 = torch.nn.Sequential(
            conv2d_bn_relu(32,64,4,stride=2),
            conv2d_bn_relu(64,64,3)
        )
        self.conv_stack4 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack5 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack6 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack7 = torch.nn.Sequential(
            conv2d_bn_relu(64,64,4,stride=2),
            conv2d_bn_relu(64,64,3),
        )

        self.conv_stack8 = torch.nn.Sequential(
            conv2d_bn_relu(64, 64, (3,4), stride=(1,2)),
            conv2d_bn_relu(64, 64, 3),
        )
        
        self.deconv_8 = deconv_relu(64,64,(3,4),stride=(1,2))
        self.deconv_7 = deconv_relu(67,64,4,stride=2)
        self.deconv_6 = deconv_relu(67,64,4,stride=2)
        self.deconv_5 = deconv_relu(67,64,4,stride=2)
        self.deconv_4 = deconv_relu(67,64,4,stride=2)
        self.deconv_3 = deconv_relu(67,32,4,stride=2)
        self.deconv_2 = deconv_relu(35,16,4,stride=2)
        self.deconv_1 = deconv_sigmoid(19,3,4,stride=2)

        self.predict_8 = torch.nn.Conv2d(64,3,3,stride=1,padding=1)
        self.predict_7 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_6 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_5 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_4 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_3 = torch.nn.Conv2d(67,3,3,stride=1,padding=1)
        self.predict_2 = torch.nn.Conv2d(35,3,3,stride=1,padding=1)

        self.up_sample_8 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,(3,4),stride=(1,2),padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_7 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_6 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )

        self.up_sample_4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )
        self.up_sample_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(3,3,4,stride=2,padding=1,bias=False),
            torch.nn.Sigmoid()
        )


    def encoder(self, x):
        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        conv6_out = self.conv_stack6(conv5_out)
        conv7_out = self.conv_stack7(conv6_out)
        conv8_out = self.conv_stack8(conv7_out)
        return conv8_out

    def decoder(self, x):

        deconv8_out = self.deconv_8(x)
        predict_8_out = self.up_sample_8(self.predict_8(x))

        concat_7 = torch.cat([deconv8_out, predict_8_out], dim=1)
        deconv7_out = self.deconv_7(concat_7)
        predict_7_out = self.up_sample_7(self.predict_7(concat_7))

        concat_6 = torch.cat([deconv7_out,predict_7_out],dim=1)
        deconv6_out = self.deconv_6(concat_6)
        predict_6_out = self.up_sample_6(self.predict_6(concat_6))

        concat_5 = torch.cat([deconv6_out,predict_6_out],dim=1)
        deconv5_out = self.deconv_5(concat_5)
        predict_5_out = self.up_sample_5(self.predict_5(concat_5))

        concat_4 = torch.cat([deconv5_out,predict_5_out],dim=1)
        deconv4_out = self.deconv_4(concat_4)
        predict_4_out = self.up_sample_4(self.predict_4(concat_4))

        concat_3 = torch.cat([deconv4_out,predict_4_out],dim=1)
        deconv3_out = self.deconv_3(concat_3)
        predict_3_out = self.up_sample_3(self.predict_3(concat_3))

        concat2 = torch.cat([deconv3_out,predict_3_out],dim=1)
        deconv2_out = self.deconv_2(concat2)
        predict_2_out = self.up_sample_2(self.predict_2(concat2))

        concat1 = torch.cat([deconv2_out,predict_2_out],dim=1)
        predict_out = self.deconv_1(concat1)
        return predict_out
        

    def forward(self,x, reconstructed_latent, refine_latent):
        if refine_latent != True:
            latent = self.encoder(x)
            out = self.decoder(latent)
            return out, latent
        else:
            latent = self.encoder(x)
            out = self.decoder(reconstructed_latent)
            return out, latent

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()
    
    def init_weights(self):
        b = 1 / self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)

class RefineDoublePendulumModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineDoublePendulumModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 4)
        self.layer5 = SirenLayer(4, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent

class RefineElasticPendulumModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineElasticPendulumModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 6)
        self.layer5 = SirenLayer(6, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent

class RefineReactionDiffusionModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineReactionDiffusionModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 2)
        self.layer5 = SirenLayer(2, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent

class RefineSwingStickNonMagneticModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineSwingStickNonMagneticModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 4)
        self.layer5 = SirenLayer(4, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent

class RefineSinglePendulumModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineSinglePendulumModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 2)
        self.layer5 = SirenLayer(2, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent


class RefineCircularMotionModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineCircularMotionModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 2)
        self.layer5 = SirenLayer(2, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent

class RefineAirDancerModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineAirDancerModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 8)
        self.layer5 = SirenLayer(8, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent

class RefineLavaLampModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineLavaLampModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 8)
        self.layer5 = SirenLayer(8, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent

class RefineFireModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineFireModel, self).__init__()

        self.layer1 = SirenLayer(in_channels, 128, is_first=True)
        self.layer2 = SirenLayer(128, 64)
        self.layer3 = SirenLayer(64, 32)
        self.layer4 = SirenLayer(32, 24)
        self.layer5 = SirenLayer(24, 32)
        self.layer6 = SirenLayer(32, 64)
        self.layer7 = SirenLayer(64, 128)
        self.layer8 = SirenLayer(128, in_channels, is_last=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        latent = self.layer4(x)
        x = self.layer5(latent)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x, latent


class RefineModelReLU(torch.nn.Module):
    def __init__(self, in_channels):
        super(RefineModelReLU, self).__init__()

        self.layer1 = nn.Linear(in_channels, 128)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 4)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(4, 64)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Linear(64, 128)
        self.relu5 = nn.ReLU()
        self.layer6 = nn.Linear(128, in_channels)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        latent = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(latent))
        x = self.relu5(self.layer5(x))
        x = self.layer6(x)
        return x, latent


class LatentPredModel(torch.nn.Module):
    def __init__(self, in_channels):
        super(LatentPredModel, self).__init__()

        self.layer1 = nn.Linear(in_channels, 32)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 64)
        self.relu3 = nn.ReLU()
        self.layer4 = nn.Linear(64, 64)
        self.relu4 = nn.ReLU()
        self.layer5 = nn.Linear(64, 32)
        self.relu5 = nn.ReLU()
        self.layer6 = nn.Linear(32, in_channels)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.relu3(self.layer3(x))
        x = self.relu4(self.layer4(x))
        x = self.relu5(self.layer5(x))
        x = self.layer6(x)
        return x
