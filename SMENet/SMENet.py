from torch.autograd import Variable
from layers import *
from data import voc
from sub_modules import *

class SMENet(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SMENet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)    # return prior_box[cx,cy,w,h]
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        self.blocks_fusion = [4, 2, 1]
        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])    # Location_para
        self.conf = nn.ModuleList(head[1])   # Confidence_Para

        self.resnet_fusion = ResNet_fusion(self.blocks_fusion)
        self.change_channels = Change_channels()
        self.Erase = nn.ModuleList(Erase())
        self.FBS = nn.ModuleList(OSE_())

        self.Fusion_detailed_information1 = Fusion_detailed_information1()
        self.Fusion_detailed_information2 = Fusion_detailed_information2()
        self.Fusion_detailed_information3 = Fusion_detailed_information3()

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources = list()    # save detected feature maps
        loc = list()
        conf = list()

        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x)
        sources.append(s)  # extract Conv4_3 layer feature map

        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)
        # Eliminate irrelevant information
        erase_sources0 = self.Erase[0](sources[0], sources[2])  # p1'
        erase_sources1 = self.Erase[1](sources[1], sources[3])  # p2'
        # Transmit detailed information
        sources[2] = self.Fusion_detailed_information3(sources[0], sources[1], sources[2])   #1024
        sources[1] = self.Fusion_detailed_information2(sources[0], erase_sources1)   # 1024
        sources[0] = self.Fusion_detailed_information1(erase_sources0)   # 1024

        sources[0], sources[1], sources[2] = self.change_channels(sources[0], sources[1], sources[2])

        # objects saliency enhancement
        # for i in range(len(sources)):
        #     sources[i] = self.FBS[i](sources[i])

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output


    # load weights
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage), strict=False)
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

def vgg(cfg, i, batch_norm=False):
    layers = []         # cave backbone structure
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]    # the ceiling method (rounding up) is adopted to ensure that the dimension of the data after pooling remains unchanged
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                #layers += [conv2d, nn.GroupNorm(num_groups=4, num_channels=v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v     # The number of output channels of the upper layer is used as the number of input channels of the next layer
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]    # Save space and perform overlay operation
    return layers

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v

    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    conf_layer = []
    loc_layer = []
    for i in range(len(cfg)):
        conf_layer.append(
            nn.Conv2d(in_channels=256, out_channels=cfg[i] * num_classes, kernel_size=3, stride=1, padding=1))
        loc_layer.append(nn.Conv2d(in_channels=256, out_channels=cfg[i] * 4, kernel_size=3, padding=1, stride=1))
    return vgg, extra_layers, (loc_layer, conf_layer)

base = {
    '400': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '400': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '400': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


# External call interface function
def build_SMENet(phase, size=400, num_classes=11):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 400:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only  (size=300) is supported!")
        return

    base_, extras_, head_ = multibox(vgg(base[str(size)], 3), add_extras(extras[str(size)], 1024), mbox[str(size)], num_classes)
    return SMENet(phase, size, base_, extras_, head_, num_classes)

