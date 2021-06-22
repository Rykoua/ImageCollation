import torch
from torchvision import transforms
import numpy as np
from tqdm import tqdm

from IllustrationMatcher.utils.models import get_conv4_model
from os.path import join, splitext, isfile
from os import listdir
from PIL import Image

from IllustrationMatcher.utils.ransac import Ransac


class FeatureMatching:
    def __init__(self, model):
        self.strideNet = 16
        self.minNet = 16
        self.base = 20
        self.model = model

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
        return [torch.from_numpy(feat) for feat in feats1]

    @staticmethod
    def compute_mutual_match(feat1, feat2):
        match1 = []
        match2 = []
        similarity = []
        grid_size = []

        n_features, feat2H, feat2W = feat2.shape
        _, feat1H, feat1W = feat1.shape
        feat1 = FeatureMatching.normalize(feat1, axis=0, is_tensor=True).permute(1, 2, 0).view(-1, n_features)
        feat2 = FeatureMatching.normalize(feat2, axis=0, is_tensor=True).permute(1, 2, 0).view(-1, n_features)
        score = torch.mm(feat1, feat2.transpose(0, 1))
        topk0_score, topk0_index = score.topk(k=1, dim=0)
        topk1_score, topk1_index = score.topk(k=1, dim=1)

        index0 = torch.zeros((score.shape[0], score.shape[1])).scatter_(0, topk0_index,
                                                                        topk0_score)
        index1 = torch.zeros((score.shape[0], score.shape[1])).scatter_(1, topk1_index,
                                                                        topk1_score)

        intersection_score = index0 * index1
        intersection = intersection_score.nonzero()

        for i1, i2 in intersection:
            i1 = i1.item()
            i2 = i2.item()
            w1, h1 = FeatureMatching.get_2d_idx(i1, feat1W)
            w2, h2 = FeatureMatching.get_2d_idx(i2, feat2W)
            match1.append([(w1 + 0.5) / feat1W, (h1 + 0.5) / feat1H])
            match2.append([(w2 + 0.5) / feat2W, (h2 + 0.5) / feat2H])
            similarity.append(intersection_score[i1, i2].item() ** 0.5)
            grid_size.append([1. / feat1W, 1. / feat1H])
        return match1, match2, similarity, grid_size

    @staticmethod
    def compute_feature_matching(feats1, feat2):
        match1 = []
        match2 = []
        similarity = []
        grid_size = []
        _, feat2_h, feat2_w = feat2.shape
        for feat1 in feats1:
            feat1 = FeatureMatching.normalize(feat1, axis=0, is_tensor=True)
            match1_, match2_, similarity_, grid_size_ = FeatureMatching.compute_mutual_match(feat1, feat2)
            match1 += match1_
            match2 += match2_
            similarity += similarity_
            grid_size += grid_size_

        match1 = torch.from_numpy(np.array(match1))
        match2 = torch.from_numpy(np.array(match2))
        similarity = torch.from_numpy(np.array(similarity))
        grid_size = torch.from_numpy(np.array(grid_size))

        return match1, match2, similarity, grid_size, feat2_h*feat2_w




def get_file_extension(file):
    return splitext(file)[1][1:].lower()


def list_folder_images(folder):
    image_types = ['jpg', 'tif', 'png', 'bmp']
    return [join(folder, f) for f in listdir(folder) if (isfile(join(folder, f)) and (get_file_extension(f) in image_types))]


def get_image_list(folder):
    images_path = list_folder_images(folder)
    return [Image.open(image_path).convert('RGB') for image_path in images_path]


if __name__ == "__main__":
    conv4 = get_conv4_model()
    feature_matching = FeatureMatching(conv4)
    feature_sizes = feature_matching.get_sizes(20, 2)

    images1 = get_image_list("../tmp_manuscripts/P2_/illustration")
    images2 = get_image_list("../tmp_manuscripts/P4_/illustration")

    descriptors1 = feature_matching.compute_multi_scale_descriptors(images1, feature_sizes)
    descriptors2 = feature_matching.compute_multi_scale_descriptors(images2, feature_sizes)
    feat1 = torch.from_numpy(descriptors1[6][0])
    feats1 = [torch.from_numpy(feat) for feat in descriptors1[6]]
    feat2 = torch.from_numpy(descriptors2[5][0])

    #match1, match2, similarity, grid_size = feature_matching.compute_mutual_match(feat1, feat2)
    match1, match2, similarity, grid_size, feature_map_size = feature_matching.compute_feature_matching(feats1, feat2)
    ransac = Ransac()
    score, _ = ransac.get_ransac_score(match1, match2, similarity, grid_size, feature_map_size, tolerance=0.1,
                            nb_iter=100,
                            transformation_name="affine", nb_max_iter=100)
    exit()