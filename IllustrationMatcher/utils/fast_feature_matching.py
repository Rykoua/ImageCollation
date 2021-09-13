import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from .models import get_conv4_model
from os.path import join, splitext, isfile
from os import listdir
from PIL import Image

from .ransac import Ransac


class FastFeatureMatching:
    def __init__(self, model):
        self.strideNet = 16
        self.minNet = 16
        self.base = 20
        self.model = model
        self.feats1 = None

    def rescale_image(self, I, featMax, featMin=1):
        w, h = I.size
        ratio = float(w) / h
        if ratio < 1:
            feat_h = featMax
            feat_w = max(round(ratio * feat_h), featMin)

        else:
            feat_w = featMax
            feat_h = max(round(feat_w / ratio), featMin)
        resize_w = (feat_w - 1) * self.strideNet + self.minNet
        resize_h = (feat_h - 1) * self.strideNet + self.minNet

        return resize_w, resize_h

    def multi_scale_resize(self, image, feature_sizes):
        images = list()
        for size in feature_sizes:
            w, h = self.rescale_image(image, size)
            images.append(image.resize((w, h)))
        return images

    @staticmethod
    def normalize(vec, axis, eta=1e-7, is_tensor=False):
        if is_tensor:
            return vec.div(torch.norm(vec, p=2, dim=axis).detach() + eta)
        else:
            return vec / (np.linalg.norm(vec, ord=2, axis=axis, keepdims=True) + eta)

    @staticmethod
    def get_sizes(base, abs_range, step=1, scale_type="affine"):
        if scale_type == "log":
            feature_sizes = [int(base * 2 ** (i / abs_range)) for i in range(-abs_range, abs_range + 1, 1)]
        else:
            feature_sizes = [base + step * i for i in range(-abs_range, abs_range + 1, 1)]
        return feature_sizes

    @staticmethod
    def get_2d_idx(i, width):
        w = i % width
        h = i // width
        return w, h

    def compute_multi_scale_descriptors(self, images, feature_sizes):
        descriptors = list()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if torch.cuda.is_available():
            self.model.to('cuda')
        self.model.eval()
        with torch.no_grad():
            print("Computing descriptors...")
            for image in tqdm(images):
                rescaled_images = self.multi_scale_resize(image, feature_sizes)
                cur_features = list()
                for cur_img in rescaled_images:
                    inp_tensor = preprocess(cur_img).unsqueeze(0)
                    if torch.cuda.is_available():
                        inp_tensor = inp_tensor.to('cuda')
                    cur_features.append(np.squeeze(self.model(inp_tensor).cpu().numpy(), axis=0))
                descriptors.append(cur_features)

        return descriptors

    @staticmethod
    def get_feats_tensors(feats1):
        n_features = feats1[0].shape[0]
        max_n_features = np.max([feat1.shape[1]*feat1.shape[2] for feat1 in feats1])
        v = torch.zeros(1, n_features).cuda()#vector that gives the minimum max among all features

        feats1_tensor = torch.Tensor().cuda()
        score_mask = torch.Tensor().cuda()
        indexes = torch.Tensor().type(torch.long).cuda()
        inv_shapes = torch.Tensor().cuda()
        for feat1 in feats1:
            feat1 = torch.from_numpy(feat1).cuda()
            _, feat1_h, feat1_w = feat1.shape
            cur_inv_shapes = torch.from_numpy(np.array([1./feat1_w, 1./feat1_h])).float().cuda().unsqueeze(0)
            inv_shapes = torch.cat((inv_shapes, cur_inv_shapes), dim=0)
            feat1 = FastFeatureMatching.normalize(feat1, axis=0, is_tensor=True)
            feat1 = feat1.permute(1, 2, 0).view(-1, n_features)
            n_features_to_add = max_n_features - feat1_h * feat1_w
            feat1 = torch.cat((feat1, v.repeat(n_features_to_add, 1)), dim=0).unsqueeze(0)
            cur_score_mask = torch.cat((torch.ones(feat1_h * feat1_w, 1).cuda(),
                                        torch.zeros(n_features_to_add, 1).cuda()),
                                       dim=0).unsqueeze(0)

            list_w = (torch.arange(0, feat1_w, 1)).unsqueeze(0).expand(feat1_h, feat1_w).contiguous().view(
                -1).type(torch.long).unsqueeze(1).cuda()
            list_h = (torch.arange(0, feat1_h, 1)).unsqueeze(1).expand(feat1_h, feat1_w).contiguous().view(
                -1).type(torch.long).unsqueeze(1).cuda()
            cur_indexes = torch.cat((list_w, list_h), dim=1)
            cur_indexes = torch.cat((cur_indexes, torch.zeros((n_features_to_add, 2), dtype=torch.long).cuda()), dim=0).unsqueeze(0)
            indexes = torch.cat((indexes, cur_indexes), dim=0)
            score_mask = torch.cat((score_mask, cur_score_mask), dim=0)
            feats1_tensor = torch.cat((feats1_tensor, feat1), dim=0)

        return feats1_tensor, indexes, score_mask, inv_shapes

    @staticmethod
    def compute_feature_matching(feats1, feat2):
        feats1_tensor, indexes, score_mask, inv_shapes = feats1
        feat2 = feat2.cuda()
        n_features, feat2_h, feat2_w = feat2.shape

        feat2 = FastFeatureMatching.normalize(feat2, axis=0, is_tensor=True)
        feat2 = feat2.permute(1, 2, 0).view(-1, n_features)

        score_mask = score_mask.repeat(1, 1, feat2_w*feat2_h)

        score = torch.matmul(feats1_tensor, feat2.transpose(0, 1))

        score = score_mask*score + score_mask-1
        topk0_score, topk0_index = score.topk(k=1, dim=1)
        topk1_score, topk1_index = score.topk(k=1, dim=2)

        index0 = torch.zeros(score.shape[0], score.shape[1], score.shape[2]).cuda().scatter_(1, topk0_index, topk0_score)
        index1 = torch.zeros(score.shape[0], score.shape[1], score.shape[2]).cuda().scatter_(2, topk1_index, topk1_score)

        intersection_score = index0 * index1

        intersection = intersection_score.nonzero()

        list2W = (torch.arange(0, feat2_w, 1)).unsqueeze(0).expand(feat2_h, feat2_w).contiguous().view(
            -1).type(torch.long).unsqueeze(1).cuda()
        list2H = (torch.arange(0, feat2_h, 1)).unsqueeze(1).expand(feat2_h, feat2_w).contiguous().view(
            -1).type(torch.long).unsqueeze(1).cuda()
        indexes2 = torch.cat((list2W, list2H), dim=1)

        inv_shapes2 = torch.from_numpy(np.array([1. / feat2_w, 1. / feat2_h])).float().cuda().unsqueeze(0)

        grid_size = inv_shapes[intersection[:, 0]]
        match1 = (indexes[intersection[:, 0], intersection[:, 1]]+0.5)*grid_size
        match2 = (indexes2[intersection[:, 2]]+0.5)*inv_shapes2
        similarity = torch.sqrt(intersection_score[intersection[:, 0], intersection[:, 1], intersection[:, 2]])

        return match1, match2, similarity, grid_size, feat2_h*feat2_w


def get_file_extension(file):
    return splitext(file)[1][1:].lower()


def list_folder_images(folder):
    image_types = ['jpg', 'tif', 'png', 'bmp']
    return [join(folder, f) for f in listdir(folder) if (isfile(join(folder, f)) and (get_file_extension(f) in image_types))]


def get_image_list(folder):
    images_path = list_folder_images(folder)
    return [Image.open(image_path).convert('RGB') for image_path in images_path]