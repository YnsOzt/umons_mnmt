import _init_paths
import os
import numpy as np
import cv2
import torch
import time
import argparse
from tqdm import tqdm
from model.utils.config import cfg, cfg_from_file
from model.faster_rcnn.resnet import resnet
from utils import get_image_blob, save_features
from numpy_nms.cpu_nms import cpu_nms


def parse_args():
    parser = argparse.ArgumentParser(description='Extract Bottom-up features')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='config file',
                        default='cfgs/faster_rcnn_resnet101.yml', type=str)
    parser.add_argument('--model', dest='model_file',
                        help='path to pretrained model',
                        default='models/bottomup_pretrained_10_100.pth', type=str)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory with images',
                        default="images")
    parser.add_argument('--out_dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--boxes', dest='save_boxes',
                        help='save bounding boxes',
                        action='store_true')


    parser.add_argument('--extraction_order_file', dest='extraction_order_file',
                        help='txt file for image extraction order')

    parser.add_argument('--out_file_name', dest='out_file_name',
                        help='name of the output file',
                        default="output_file")    

    parser.add_argument('--min_boxes', dest="min_boxes", type=int,
                    help='An integer for the minimum number of bounding boxes',
                    default=36)                    
       
    parser.add_argument('--max_boxes', dest="max_boxes", type=int,
                    help='An integer for the maximum number of bounding boxes',
                    default=36)


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Load arguments.
    args = parse_args()
    MIN_BOXES = args.min_boxes
    MAX_BOXES = args.max_boxes
    N_CLASSES = 1601
    CONF_THRESH = 0.2

    print("min box : ", MIN_BOXES)
    print("max box : ", MAX_BOXES)
    

    #load the default config if no file is given
    if args.cfg_file is not None: 
        cfg_from_file(args.cfg_file)

    #create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available()
    assert use_cuda, 'Works only with CUDA' 
    device = torch.device('cuda') if use_cuda else torch.device('cpu')
    cfg.CUDA = use_cuda
    np.random.seed(cfg.RNG_SEED)
    
    # Load the model.
    fasterRCNN = resnet(N_CLASSES, 101, pretrained=False)
    fasterRCNN.create_architecture()
    fasterRCNN.load_state_dict(torch.load(args.model_file))
    fasterRCNN.to(device)
    fasterRCNN.eval()
    print('Model is loaded.')

    # Load images.
    imglist = os.listdir(args.image_dir)
    
    num_images = sum(1 for line in open(args.extraction_order_file, "r"))
    print('Number of images to extract features from: {}.'.format(num_images))
    
    all_images_feats = []
    all_images_boxes = []
    # Extract features. 
    with open(args.extraction_order_file, "r") as f:
        for im_file in tqdm(f, total=num_images):
            im_file = im_file.strip() #delete the whitespaces in the name of the image
            im = cv2.imread(os.path.join(args.image_dir, im_file))
            blobs, im_scales = get_image_blob(im)
            assert len(im_scales) == 1, 'Only single-image batch is implemented'

            im_data = torch.from_numpy(blobs).permute(0, 3, 1, 2).to(device)
            im_info = torch.tensor([[blobs.shape[1], blobs.shape[2], im_scales[0]]]).to(device)
            gt_boxes = torch.zeros(1, 1, 5).to(device)
            num_boxes = torch.zeros(1).to(device)

            with torch.set_grad_enabled(False):
                rois, cls_prob, _, _, _, _, _, _, \
                pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            boxes = rois.data.cpu().numpy()[:, :, 1:5].squeeze()
            boxes /= im_scales[0]
            cls_prob = cls_prob.data.cpu().numpy().squeeze()
            pooled_feat = pooled_feat.data.cpu().numpy()

            # Keep only the best detections.
            max_conf = np.zeros((boxes.shape[0]))
            for cls_ind in range(1, cls_prob.shape[1]):
                cls_scores = cls_prob[:, cls_ind]
                dets = np.hstack((boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = np.array(cpu_nms(dets, cfg.TEST.NMS))
                max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

            keep_boxes = np.where(max_conf >= CONF_THRESH)[0]
            if len(keep_boxes) < MIN_BOXES:
                keep_boxes = np.argsort(max_conf)[::-1][:MIN_BOXES]
            elif len(keep_boxes) > MAX_BOXES:
                keep_boxes = np.argsort(max_conf)[::-1][:MAX_BOXES]
          
            image_feat = pooled_feat[keep_boxes]
            if args.save_boxes:
                image_bboxes = boxes[keep_boxes]
                all_images_boxes.append(image_bboxes)
            else:
                all_images_boxes = None

            all_images_feats.append(image_feat)

            #torch.cuda.empty_cache()

    #transforms array to numpy array
    all_images_feats = np.asarray(all_images_feats, dtype=np.float16)
    if args.save_boxes:
      all_images_boxes = np.asarray(all_images_boxes, dtype=np.float16)

    #save the files
    output_file = os.path.join(args.output_dir, args.out_file_name+'.npy')
    save_features(output_file, all_images_feats, all_images_boxes)
