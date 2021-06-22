from utils.DirTools import DirTools
from utils.models import get_resnet50_conv4_model
from utils.features import compute_features, compute_cosine_matrix
from utils.Results import Results

if __name__ == "__main__":
    # specify here the manuscripts path and the result folder where all the pages will be stored
    manuscript1_path = "manuscripts_demo/P2/illustration"
    manuscript2_path = "manuscripts_demo/P3/illustration"
    result_folder = "./P2P3_web_pages"
    nn_number = 10

    # compute basic cosine score matrix
    images1 = DirTools.get_images_list(manuscript1_path)  # gets PIL images list for the 1st manuscript folder
    images2 = DirTools.get_images_list(manuscript2_path)  # gets PIL images list for the 2nd manuscript folder
    batch_size = 10  # batch size for computing the features
    input_image_size = 320  # specify here the image size before using it as an input for the network
    resnet50_conv4 = get_resnet50_conv4_model()  # loads the pretrained resnet50 truncated, with the final layer being the conv4
    print("computing features...")
    feats1 = compute_features(resnet50_conv4, images1, input_image_size, batch_size=batch_size, use_cuda=True)  # computes conv4 features for the 1st manuscript
    print("computing features...")
    feats2 = compute_features(resnet50_conv4, images2, input_image_size, batch_size=batch_size, use_cuda=True)  # computes conv4 features for the 2nd manuscript
    score_matrix = compute_cosine_matrix(feats1, feats2)  # computes the cosine similarity score matrix

    # create results web pages
    print("generating result webpages...")
    results = Results(manuscript1_path, manuscript2_path, score_matrix, with_annotation=True)
    results.save_all_pages(result_folder, number_nn=nn_number)
    print("result web pages saved successfully in the folder {}".format(result_folder))
    exit()
