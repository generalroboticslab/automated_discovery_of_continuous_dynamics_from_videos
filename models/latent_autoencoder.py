from models.sub_modules import *
# from torchvision.models.optical_flow import raft_large


class LargeLatentEncoder(torch.nn.Module):
    def __init__(self, in_channels, architecture="default", **kwargs):

        super(LargeLatentEncoder,self).__init__()

        self.architecture = architecture

        if self.architecture == "default" or self.architecture == "diff":

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
        
    def forward(self, x):

        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        return conv5_out

class LargeLatentDecoder(torch.nn.Module):
    def __init__(self, in_channels, architecture="default", **kwargs):

        super(LargeLatentDecoder,self).__init__()

        self.architecture = architecture

        if self.architecture == "default" or self.architecture == "diff":

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
        
    def forward(self, x):

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

class LargeLatentAutoEncoder(torch.nn.Module):
    def __init__(self, channels, dataset, seed, architecture="default", model_name='encoder-decoder', **kwargs):
        super(LargeLatentAutoEncoder,self).__init__()

        self.name = '_'.join([model_name, str(seed)])
        self.dataset = dataset
        self.seed = seed
        self.architecture = architecture

        self.encoder = LargeLatentEncoder(channels, architecture)
        self.decoder = LargeLatentDecoder(channels, architecture)
        

    def forward(self,x):
        
        latent = self.encoder(x)

        output = self.decoder(latent)

        return output, latent



class LatentEncoder(torch.nn.Module):
    def __init__(self, in_channels, architecture="default", **kwargs):

        super(LatentEncoder,self).__init__()

        self.architecture = architecture

        if self.architecture == "default" or self.architecture == "diff":

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
    
    def forward(self, x):

        conv1_out = self.conv_stack1(x)
        conv2_out = self.conv_stack2(conv1_out)
        conv3_out = self.conv_stack3(conv2_out)
        conv4_out = self.conv_stack4(conv3_out)
        conv5_out = self.conv_stack5(conv4_out)
        conv6_out = self.conv_stack6(conv5_out)
        conv7_out = self.conv_stack7(conv6_out)
        conv8_out = self.conv_stack8(conv7_out)

        return conv8_out

class LatentDecoder(torch.nn.Module):
    def __init__(self, out_channels, architecture="default", **kwargs):

        super(LatentDecoder,self).__init__()

        self.architecture = architecture

        if self.architecture == "default" or self.architecture == "diff":
            self.deconv_8 = deconv_relu(64,64,(3,4),stride=(1,2))
            self.deconv_7 = deconv_relu(67,64,4,stride=2)
            self.deconv_6 = deconv_relu(67,64,4,stride=2)
            self.deconv_5 = deconv_relu(67,64,4,stride=2)
            self.deconv_4 = deconv_relu(67,64,4,stride=2)
            self.deconv_3 = deconv_relu(67,32,4,stride=2)
            self.deconv_2 = deconv_relu(35,16,4,stride=2)
            self.deconv_1 = deconv_sigmoid(19,out_channels,4,stride=2)

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
    
    def forward(self, x):

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

class LatentAutoEncoder(torch.nn.Module):
    def __init__(self, channels, dataset, seed, architecture="default", model_name='encoder-decoder-64', **kwargs):
        super(LatentAutoEncoder,self).__init__()

        self.name = '_'.join([model_name, str(seed)])
        self.dataset = dataset
        self.seed = seed
        self.architecture = architecture

        self.encoder = LatentEncoder(channels, architecture)
        self.decoder = LatentDecoder(channels, architecture)
        

    def forward(self,x):
        
        latent = self.encoder(x)

        output = self.decoder(latent)

        return output, latent


