from __future__ import division

import numpy as np

# import sys
# sys.path.append('/home/alan/git/al-caffe/python')
import caffe

MAX_BOXES = 50

class ScoreDetections(caffe.Layer):
    """
    * Marks up bbox predictions as true positive/ false positive and missed gtruth bbox as
        true negatives
    * bottom[0] - list of ground truth bbox
        [batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, 0) ]
    * bottom[1] - list of predicted bbox
        [batch_size x max_bbox_per_image x 5 (xl, yt, xr, yb, confidence) ]
    * top[0] - mAP
    * top[1] - precision
    * top[2] - recall

    Example prototxt definition:

    layer {
        type: 'Python'
        name: 'score'
        top: 'mAP'
        top: 'precision'
        top: 'recall'
        bottom: 'gt_bbox_list'
        bottom: 'det_bbox_list'
        python_param {
            module: 'caffe.layers.detectnet.mean_ap_part'
            layer: 'ScoreDetections'
        }
        include: { phase: TEST }
    }
    """

    def setup(self, bottom, top):
        try:
            plist = self.param_str.split(',')
            self.image_size_x = int(plist[0])
            self.image_size_y = int(plist[1])
        except ValueError:
            raise ValueError("Parameter string missing or data type is wrong!")

    def reshape(self, bottom, top):
        # n_images = bottom[0].data.shape[0]
        top[0].reshape(1)
        top[1].reshape(1)
        top[2].reshape(1)
        assert(bottom[0].data.shape[0] == bottom[1].data.shape[0]), "# of images not matching!"

    def forward(self, bottom, top):
        self.avp, self.precision, self.recall = calcmAP(bottom[0].data, bottom[1].data, self.image_size_x, self.image_size_y)
        top[0].data[...] = self.avp
        top[1].data[...] = self.precision
        top[2].data[...] = self.recall

    def backward(self, top, propagate_down, bottom):
        pass


def tp_fp_fn(det, rhs):
    x_overlap = max(0, min(det[2], rhs[2]) - max(det[0], rhs[0]))
    y_overlap = max(0, min(det[3], rhs[3]) - max(det[1], rhs[1]))
    overlap_area = x_overlap * y_overlap

    det_area = (det[2]-det[0])*(det[3]-det[1])
    rhs_area = (rhs[2]-rhs[0])*(rhs[3]-rhs[1])
    unionarea = det_area + rhs_area - overlap_area
    return overlap_area/rhs_area, (det_area - overlap_area)/det_area, (rhs_area - overlap_area)/rhs_area


def divide_zero_is_zero(a, b):
    return float(a)/float(b) if b != 0 else 0


def calcmAP(gt_bbox_list, det_bbox_list, image_size_x, image_size_y):
    matched_bbox = np.zeros([gt_bbox_list.shape[0], MAX_BOXES, 5])

    precision = []
    recall = []
    avp = []

    for k in range(gt_bbox_list.shape[0]):
        # Remove  zeros from detected bboxes
        cur_det_bbox = det_bbox_list[k, :, 0:4]
        cur_det_bbox = np.asarray(filter(lambda a: a.tolist() != [0, 0, 0, 0], cur_det_bbox))

        # Remove  zeros from label bboxes
        cur_gt_bbox = gt_bbox_list[k, :, 0:4]
        cur_gt_bbox = np.asarray(filter(lambda a: a.tolist() != [0, 0, 0, 0], cur_gt_bbox))

        n_det = cur_det_bbox.shape[0]
        n_gt = cur_gt_bbox.shape[0]

        gt_matched = np.zeros([cur_gt_bbox.shape[0]])
        det_matched = np.zeros([cur_det_bbox.shape[0]])

        gt_tps = np.zeros([cur_gt_bbox.shape[0]])
        gt_fns = np.zeros([cur_gt_bbox.shape[0]])
        det_fps = np.zeros([cur_det_bbox.shape[0]])

        for i in range(cur_gt_bbox.shape[0]):
            for j in range(cur_det_bbox.shape[0]):
                tp, fn, fp = tp_fp_fn(cur_det_bbox[j], cur_gt_bbox[i])
                if tp > gt_tps[i]:
                    gt_tps[i] = tp
                    gt_fns[i] = fn
                    det_fps[j] = fp

        if n_det:
            tp_rate = gt_tps.mean()
            fp_rate = det_fps.mean()
        else:
            fp_rate = 0
            tp_rate = 0

        pr = divide_zero_is_zero(tp_rate, tp_rate + fp_rate)
        rc = tp_rate
        av = pr * rc
        precision.append(pr)
        recall.append(rc)
        avp.append(av)
    return np.mean(avp)*100, np.mean(precision)*100, np.mean(recall)*100
