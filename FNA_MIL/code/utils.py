from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from skimage.util import random_noise
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import numbers
from PIL import ImageDraw, Image
import cv2 as cv
import pwd


class AverageMeter:
    """
    Computes and stores the average and
    current value.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_torch(random_seed, use_gpu, gpu_number=0):
    torch.manual_seed(random_seed)
    torch.set_num_threads(8)
    if use_gpu:
        # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
        torch.cuda.manual_seed(random_seed)


def aggregate_scores(score_list, method='median'):
    """
    :param score_list: confidence scores for a patient, in list or as a np array. Should be for a single patient
    :param method: aggregation method to use (recommend mean or median)
    :return: aggregated score (single float)
    """
    scores_np = np.float_(score_list)
    if method == 'median':
        return np.median(scores_np)
    elif method == 'mean':
        return np.mean(scores_np)
    elif method == 'max':
        return np.max(scores_np)
    elif method == 'min':
        return np.min(scores_np)
    elif method == 'range':
        return np.max(scores_np) - np.min(scores_np)
    elif method == 'chance':
        return 1


def plot_roc_curve(ground_truth, scores, model_id, aggregation_method='mean'):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    # predicted_scores = aggregate_scores(scores, aggregation_method)
    fpr, tpr, thresholds = roc_curve(ground_truth, scores)
    auc = roc_auc_score(ground_truth, scores)
    axs.set_title(f"ROC Curve \n AUC = {auc:.3f}")
    axs.plot(fpr, tpr)
    axs.set_xlabel("False Positive Rate")
    axs.set_ylabel("True Positive Rate")
    plot_dir = '/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/models'
    plot_path = os.path.join(plot_dir, model_id, 'best_roc_curve.png')
    fig.savefig(plot_path)
    plt.close(fig)

def plot_pr_curve(ground_truth, scores, test_aps, model_id):
    fig, axs = plt.subplots(nrows=1, ncols=1)
    precision, recall, thresholds = precision_recall_curve(ground_truth, scores)
    axs.set_title(f"Precision Recall Curve \n APS = {test_aps:.3f}")
    axs.plot(recall, precision)
    axs.set_xlabel("Recall")
    axs.set_ylabel("Precision")
    plot_dir = '/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/models'
    plot_path = os.path.join(plot_dir, model_id, 'best_pr_curve.png')
    fig.savefig(plot_path)
    plt.close(fig)



def save_model(model, model_id):
    model_dir = '/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/models'
    model_path = os.path.join(model_dir, model_id, 'best.pth')
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")


def load_model(model, model_id, strict=True):
    model_dir = '/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/models'
    model_path = os.path.join(model_dir, model_id, 'best.pth')
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict, strict=strict)

    return model


class SpeckleNoise(object):

    def __init__(self, noise_std):
        self.noise_std = noise_std

    def __call__(self, img):
        np_img = np.asarray(img).copy()
        output = random_noise(np_img, mode='speckle', var=self.noise_std)
        # random noise gives a floating point output
        output = (output * 255).astype(np.uint8)
        img_pil = Image.fromarray(output)
        return img_pil


class BlankImage(object):
    def __init__(self, blank_value, probs = 0.3):
        self.blank_value = blank_value
        self.probs = probs

    def __call__(self, img):
        if np.random.random() < self.probs:
            blank = torch.ones_like(img)*self.blank_value
            return blank
        return img

def get_covid_transforms(speckle=0, hue=0, saturation=0):
    point_transforms = []
    if speckle != 0:
        point_transforms.append(SpeckleNoise(speckle))
    if hue != 0 or saturation != 0:
        point_transforms.append(transforms.ColorJitter(brightness=0, contrast=0, saturation=saturation, hue=hue))

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            #*point_transforms,
            #transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            #transforms.RandomAffine(degrees=45),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms


def get_user():
    username = pwd.getpwuid(os.getuid()).pw_name
    return username