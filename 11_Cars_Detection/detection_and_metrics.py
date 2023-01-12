# ============================== 1 Classifier model ============================

import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import numpy as np
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear, Softmax
from torch.nn.functional import nll_loss 


# def get_cls_model(input_shape):
#     """
#     :param input_shape: tuple (n_rows, n_cols, n_channels)
#             input shape of image for classification
#     :return: nn model for classification
#     """
#     n_rows, n_cols, n_channels = input_shape
#     model = Sequential(
#         Conv2d(n_channels, 2 * n_channels, 5, padding=2),
#         BatchNorm2d(2 * n_channels),
#         ReLU(),
#         MaxPool2d(2),
#         Flatten(),
#         Linear(n_rows * n_cols // 4 * 2 * n_channels, 2),
#         Softmax()
#     )  
#     return model

def get_cls_model(input_shape):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    n_rows, n_cols, n_channels = input_shape
    model = Sequential(
        Flatten(),
        Linear(n_rows * n_cols * n_channels, 50),
        ReLU(),
        Linear(50, 20),
        ReLU(),
        Linear(20, 2),
        Softmax()
    )  
    return model

def fit_cls_model(X, y, testing=True):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """

    MAX_EPOCHS = 50
    if testing:
        MAX_EPOCHS = 4
    batch_size = 5
    model = get_cls_model((40, 100, 1))
    criterion = nll_loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001) #, momentum=0.9)
    trainloader = torch.utils.data.DataLoader(list(zip(X, y)), batch_size=batch_size, shuffle=True)
    for epoch in range(MAX_EPOCHS):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = torch.reshape(outputs, (outputs.shape[0], outputs.shape[1]))
            loss = nll_loss(outputs, labels)
            loss.backward()
            optimizer.step()
    if not testing:
        torch.save(model, "classifier_model.pth")
        print("Saved")
    return model

# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """
    model = Sequential(
        Conv2d(1, 50, kernel_size=(40, 100)),
        ReLU(),
        Conv2d(50, 20, kernel_size=1),
        ReLU(),
        Conv2d(20, 2, kernel_size=1),
        # ReLU(),
        Softmax(),
    )  
    model.eval()
    with torch.no_grad():
        # for layer in model.named_parameters():
        #     print(layer)
        for layer in model:
            print(layer)
        shapes = [(50, 1, 40, 100), (20, 50, 1, 1), (2, 20, 1, 1)]
        children, cls_children = list(model.children()), list(cls_model.children())
        for child in model.children():
            print(child)
            for param in child.parameters():
                param.requires_grad = False
        children[0].weight.data = cls_children[1].weight.reshape((50, 1, 40, 100))
        children[0].bias.data = cls_children[1].bias
        children[2].weight.data = cls_children[3].weight.reshape((20, 50, 1, 1))
        children[2].bias.data = cls_children[3].bias
        children[4].weight.data = cls_children[5].weight.reshape((2, 20, 1, 1))
        children[4].bias.data = cls_children[5].bias
    return model

# get_detection_model(get_cls_model((40, 100, 1)))
# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format
        {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    # your code here \/
    p = 0.8
    answer = {}
    for img_name, raw_img in dictionary_of_images.items():
        img = np.zeros((220, 370, 1))
        if len(raw_img.shape) == 2:
            raw_img = raw_img.reshape((raw_img.shape[0], raw_img.shape[1], 1))
        # print(f"{img[:raw_img.shape[0], :raw_img.shape[1], ...].shape=}, {raw_img.shape=}")
        img[:raw_img.shape[0], :raw_img.shape[1], ...] = raw_img
        features_map = detection_model(torch.tensor(np.transpose(img, axes=(2, 0, 1))[None, ...], dtype=torch.float))  
        features_map = features_map[0, 1, ...]
        mask = features_map > p
        detections = []
        for pos in np.argwhere(mask).T:
            detections.append(np.array([pos[0], pos[1], 40, 100, features_map[pos[0], pos[1]]])) 
        answer[img_name] = detections
        print(img_name)
    return answer
    # your code here /\


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    x1, y1, n1, m1 = first_bbox
    x2, y2, n2, m2 = second_bbox
    x3, y3, x4, y4 = max(x1, x2), max(y1, y2), min(x1 + n1, x2 + n2), min(y1 + m1, y2 + m2)
    inter_area = max(0, (x4 - x3)) * max(0, (y4 - y3))
    iou = (inter_area) / (n1 * m1 + n2 * m2 - inter_area)
    return iou
    # your code here /\

def calc_iou_vect(first_bbox, second_bbox):
    x1, y1, n1, m1 = first_bbox
    x2, y2, n2, m2 = second_bbox
    x3, y3, x4, y4 = np.maximum(x1, x2), np.maximum(y1, y2), np.minimum(x1 + n1, x2 + n2), np.minimum(y1 + m1, y2 + m2)
    inter_area = np.maximum(0, (x4 - x3)) * np.maximum(0, (y4 - y3))
    iou = (inter_area) / (n1 * m1 + n2 * m2 - inter_area)
    return iou

# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """
    iou_thr = 0.5
    tp, fp, all_p = [], [], []
    gt_len = np.sum([len(x) for x in gt_bboxes.values()])
    for name in pred_bboxes.keys():
        pred_boxes = [it for it in pred_bboxes[name]]
        pred_boxes.sort(key=lambda x: x[4], reverse=True)
        for box in pred_boxes:
            if len(gt_bboxes[name]) == 0:
                all_p.append(box[-1])
                continue

            # gt_boxes = np.array(gt_bboxes[name])
            ious = np.array([calc_iou(tuple(box[:-1]), gt_box) for gt_box in gt_bboxes[name]])

            # ious = calc_iou_vect(tuple(box[:-1]), (gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]))
            ind = np.argmax(ious)
            if ious[ind] >= iou_thr:
                tp.append(box[-1]) # name, conf, IoU
                gt_bboxes[name].pop(ind)
            all_p.append(box[-1])
    tp.sort(reverse=True)
    all_p.sort(reverse=True)
    all_p, tp = np.array(all_p), np.array(tp)
    recall_precision = np.zeros((all_p.shape[0] + 1, 2))
    for cur_pos, box in enumerate(all_p):
        c = box
        all_c = np.count_nonzero(all_p >= c)
        if tp.size == 0:
            precision = recall = 0
        else:
            tp_c = np.count_nonzero(tp >= c)
            precision = tp_c / all_c
            recall = tp_c / gt_len
        recall_precision[cur_pos + 1] = np.array([recall, precision])
    recall_precision[0] = np.array([0, 1])
    recall_precision = np.array(recall_precision)
    ans = np.sum((recall_precision[1:, 0] - recall_precision[:-1, 0]) * (recall_precision[1:, 1] + recall_precision[:-1, 1])) / 2
    return ans


# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.3):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """
    # your code here \/
    answer = {}
    for name, detections in detections_dictionary.items():
        pred = np.array(sorted(detections, key=lambda x: x[4], reverse=True))
        i = 0
        while i < pred.shape[0]:
            ious = np.array([calc_iou(tuple(pred[i, :-1]), tuple(pred[j, :-1])) for j in range(i + 1, pred.shape[0])])
            pred = np.concatenate([pred[:i + 1], pred[i + 1:][ious <= iou_thr]])
            i += 1
        answer[name] = list(pred)
    return answer
    # your code here /\
