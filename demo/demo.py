#-*-coding:utf-8-*-
import torch, cv2
from torch.autograd import Variable
import argparse


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--img_path', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--class_num', default=201,
                    type=int, help='Trained state_dict file path')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def predict(frame):
    height, width = frame.shape[:2]
    x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))
    y = net(x)  # forward pass
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])),
                        FONT, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, args.class_num)    # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    # img_name = "/media/hkuit155/NewDisk/dataset/fake_dataset/real_cityscapes_small/JPEGImages/hamburg_000000_043944_leftImg8bit.png"
    im = cv2.imread(args.img_path)
    result = predict(im)
    cv2.imshow("res", result)
    cv2.waitKey(0)
