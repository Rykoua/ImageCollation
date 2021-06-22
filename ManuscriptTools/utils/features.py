from torchvision import transforms
import torch
import numpy as np
from tqdm import tqdm


def compute_features(model, images, size, batch_size, use_cuda=True):
    preprocess = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensors = [preprocess(image) for image in images]
    tensors = torch.stack(tensors, dim=0)
    outputs = []
    if torch.cuda.is_available() and use_cuda:
        model.to('cuda')
    model.eval()
    progress_bar = tqdm(total=len(images))
    with torch.no_grad():
        for i in range(int(np.ceil(tensors.shape[0] / batch_size))):
            input_batch = tensors[i * batch_size:(i + 1) * batch_size]  # take a small batch
            progress_bar.update(input_batch.shape[0])
            if torch.cuda.is_available() and use_cuda:  # move input to gpu
                input_batch = input_batch.to('cuda')
            outputs.append(np.stack(model(input_batch).cpu().numpy(), axis=0))  # add outputs to a list
    model.to('cpu')
    progress_bar.close()
    return np.concatenate(outputs, axis=0)


def normalize_feats(feat):
    return feat/np.linalg.norm(feat, 2, axis=-1, keepdims=True)


def compute_cosine_matrix(feats1, feats2):
    feats1 = normalize_feats(feats1.reshape(feats1.shape[0], -1))
    feats2 = normalize_feats(feats2.reshape(feats2.shape[0], -1))
    return feats1.dot(feats2.T)
