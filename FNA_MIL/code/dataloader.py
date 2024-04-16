import json
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import os
import glob
from PIL import Image
from tqdm import tqdm
import cv2 as cv


all_orders = {
    'positive': ['26_adenocarcinoma1', '26_adenocarcinoma2', '26_adenocarcinoma3',
                 '27_adenocarcinoma4', '27_adenocarcinoma5', '27_adenocarcinoma6',
                 '28_adenocarcinoma7', '28_adenocarcinoma8'],
    'negative': ['29_adenocarcinomaNorm1', '29_adenocarcinomaNorm2',
                 '68_adenocarcinomaNorm3', '68_adenocarcinomaNorm4', '68_adenocarcinomaNorm5',
                 '69_adenocarcinomaNorm6', '69_adenocarcinomaNorm7', '69_adenocarcinomaNorm8'],
    'all': ['26_adenocarcinoma1', '26_adenocarcinoma2', '26_adenocarcinoma3',
            '27_adenocarcinoma4', '27_adenocarcinoma5', '27_adenocarcinoma6',
            '28_adenocarcinoma7', '28_adenocarcinoma8',
            '29_adenocarcinomaNorm1', '29_adenocarcinomaNorm2',
            '68_adenocarcinomaNorm3', '68_adenocarcinomaNorm4', '68_adenocarcinomaNorm5',
            '69_adenocarcinomaNorm6', '69_adenocarcinomaNorm7', '69_adenocarcinomaNorm8']
    }

class ImageCache:
    def __init__(self, cache_amnt=-1):
        self.cached = {}

    def get(self, path):
        if path not in self.cached:
            im = Image.open(path).copy()
            self.cached[path] = im
        else:
            return self.cached[path]
        return self.cached[path]


class BagDataset(torch.utils.data.Dataset):
    def __init__(self, bags, data_transforms=None, is_test=False, cache_images=True, random_sample=True, mil_size=8):
        """

        :param bags: list of dicts containing paths/labels/orders
        :param data_transforms: image transforms applied to images
        :param metadata: any extra data the should be stored with the dataset for convenience
        """
        self.bags = bags
        self.data_transforms = data_transforms
        self.is_test = is_test
        self.cache_images = cache_images
        self.random_sample = random_sample
        self.mil_size = mil_size
        if self.cache_images:
            self.image_cache = ImageCache()

    def worker_init_fn(worker_id):
        np.random.seed(torch.initial_seed())

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        image_paths = self.bags[index]['files']
        if not self.is_test and self.random_sample:
            image_paths = np.random.choice(image_paths, min(self.mil_size, len(image_paths)), replace=False)
        label = self.bags[index]['label']
        if self.cache_images:
            images = [self.image_cache.get(image_path) for image_path in image_paths]
        else:
            images = [Image.open(image_path) for image_path in image_paths]
        transformed_images = torch.stack([self.data_transforms(image) for image in images])

        order = self.bags[index]['order']

        return transformed_images, label, order


def split_train_test(x_neg, x_pos, test_data_index, test_data_fold):
    x_neg_n = len(list(x_neg))
    x_pos_n = len(list(x_pos))

    if test_data_index == -1:
        if test_data_fold == -1:
            X_trn_neg, X_tst_neg = train_test_split(all_orders['negative'], test_size=0.3, random_state=0)
            X_trn_pos, X_tst_pos = train_test_split(all_orders['positive'], test_size=0.3, random_state=0)
        else:
            kf = KFold(n_splits=4, shuffle=True, random_state=0)
            for i, (train_index, test_index) in enumerate(kf.split(np.arange(len(all_orders['negative'])))):
                if i == test_data_fold:
                    X_trn_neg = [all_orders['negative'][i] for i in train_index]
                    X_tst_neg = [all_orders['negative'][i] for i in test_index]
            for i, (train_index, test_index) in enumerate(kf.split(np.arange(len(all_orders['positive'])))):
                if i == test_data_fold:
                    X_trn_pos = [all_orders['positive'][i] for i in train_index]
                    X_tst_pos = [all_orders['positive'][i] for i in test_index]
    else:
        one_test = all_orders['all'][test_data_index]
        X_trn_neg = [all_order for all_order in all_orders['negative'] if all_order != one_test]
        X_tst_neg = [all_order for all_order in all_orders['negative'] if all_order == one_test]
        X_trn_pos = [all_order for all_order in all_orders['positive'] if all_order != one_test]
        X_tst_pos = [all_order for all_order in all_orders['positive'] if all_order == one_test]

    temp_neg = [neg for neg in x_neg if neg[:neg.find('_',10)] in X_trn_neg]
    temp_pos = [pos for pos in x_pos if pos[:pos.find('_',10)] in X_trn_pos]
    train_x = temp_neg + temp_pos
    train_y = [0] * len(temp_neg) + [1] * len(temp_pos)

    temp_neg = [neg for neg in x_neg if neg[:neg.find('_',10)] in X_tst_neg]
    temp_pos = [pos for pos in x_pos if pos[:pos.find('_',10)] in X_tst_pos]
    test_x = temp_neg + temp_pos
    test_y = [0] * len(temp_neg) + [1] * len(temp_pos)

    return train_x, train_y, test_x, test_y


def get_patient_orders():
    base_path = '/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/data/data'
    orders = os.listdir(base_path)
    test_results = [True if order[:order.find('_',10)] in all_orders['positive'] else False for order in orders]
    
    positive_images = {}
    negative_images = {}
    all_image_paths = glob.glob(os.path.join(base_path, '**', '*.jpg'), recursive=True)
    for order, label in tqdm(zip(orders, test_results), total=len(orders)):
        image_paths = [ip for ip in all_image_paths if str(order) in ip]
        if label:
            positive_images[str(order)] = image_paths
        else:
            negative_images[str(order)] = image_paths
    # sort by order number, python 3.7 has dictionaries ordered by default
    negative_images = dict(sorted(negative_images.items()))
    positive_images = dict(sorted(positive_images.items()))
    all_images = dict(negative_images, **positive_images)
    neg_pat_count = len(negative_images)
    pos_pat_count = len(positive_images)
    neg_cell_count = sum([len(values) for values in negative_images.values()])
    pos_cell_count = sum([len(values) for values in positive_images.values()])

    print("Data Stats:")
    print(
        f"          - {neg_pat_count} negative patients, {pos_pat_count} positive patients -- {pos_pat_count / (pos_pat_count + neg_pat_count)} positive pat. fraction")
    print(
        f"          - {neg_cell_count} negative FOVs, {pos_cell_count} positive FOVs -- {pos_cell_count / (pos_cell_count + neg_cell_count)} positive FOV fraction")

    return negative_images, positive_images, all_images


def load_orders_into_bags(orders, image_paths, labels):
    bags = []
    for order, label in tqdm(zip(orders, labels)):
        files = image_paths[order]

        bag = {
            "order": order,
            "label": label,
            "files": files
        }
        bags.append(bag)
    return bags


def load_all_patients(train_transforms=None, test_transforms=None, batch_size=8,
                      random_sample=True, mil_size=None, test_data_index=-1, test_data_fold=-1):
    """
    Loads all the data from the Duke COVID +/- dataset
    :param group_by_patient: return one entry per patient (bag style), default false
    :param batch_size: images to load at a time
    :return: iterable dataset objects for training and test
    """
    cache_path = '/media/data1/kanghyun/eagle_cytology/MCAS-Cytology/FNA_MIL/cache/data.pkl'
    import pickle
    if os.path.exists(cache_path):
        print("reading cache")
        with open(cache_path, 'rb') as fp:
            cache = pickle.load(fp)
            negative_image_paths = cache['neg_paths']
            positive_image_paths = cache['pos_paths']
            all_image_paths = cache['all_paths']
    else:
        print("No cache available")
        negative_image_paths, positive_image_paths, all_image_paths = get_patient_orders()
        data = {
            'neg_paths': negative_image_paths,
            'pos_paths': positive_image_paths,
            'all_paths': all_image_paths
        }
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump(data, fp)
    negative_orders = negative_image_paths.keys()
    positive_orders = positive_image_paths.keys()
    # split into train/test
    train_orders, train_labels, test_orders, test_labels = split_train_test(negative_orders, positive_orders, test_data_index, test_data_fold)

    train_bags = load_orders_into_bags(train_orders, all_image_paths, train_labels)
    test_bags = load_orders_into_bags(test_orders, all_image_paths, test_labels)
    training_dataset = BagDataset(train_bags, data_transforms=train_transforms, random_sample=random_sample, mil_size=mil_size)
    test_dataset = BagDataset(test_bags, data_transforms=test_transforms, is_test=True, random_sample=random_sample)

    num_workers = 4
    train_loader = DataLoader(training_dataset, batch_size=batch_size, pin_memory=True, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader