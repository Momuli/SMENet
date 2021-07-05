import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import torch
from torch.autograd import Variable
import numpy as np
import cv2
from SMENet import build_SMENet
from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
import argparse

parser = argparse.ArgumentParser(description= 'SMENet Test')
parser.add_argument('--trained_model', default='../weights/SMENet.pth',
                    type=str, help='Trained state_dict file path to open')
args = parser.parse_args()

net = build_SMENet('test', 400, 11)
net.load_state_dict(torch.load(args.trained_model), strict=False)

testset = VOCDetection(VOC_ROOT, [('2012', 'test')], None, VOCAnnotationTransform())
for img_id in range(len(testset)):
    image = testset.pull_image(img_id)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    x = cv2.resize(image, (400, 400)).astype(np.float32)
    x -= (86.0, 91.0, 82.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    x = torch.from_numpy(x).permute(2, 0, 1)

    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    if torch.cuda.is_available():
        xx = xx.cuda()
    y= net(xx)
    from data import VOC_CLASSES as labels
    top_k=10

    plt.figure(figsize=(10,10))
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    plt.imshow(rgb_image)
    currentAxis = plt.gca()

    detections = y.data
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0

        while detections[0,i,j,0] >= 0.6:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            color = colors[i]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1
plt.show()