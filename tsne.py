import argparse
from tqdm import tqdm
import cv2
import torch
import random
import numpy as np
import colorsys
from shutil import copyfile, rmtree
from os import path, mkdir
from os.path import isdir, basename, normpath, join

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from gap_statistic import OptimalK

from dataset import Dataset, collate_skip_empty
from resnet import ResNet101

colors_per_cluster = {}


def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_features(dataset, batch):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # initialize our implementation of ResNet
    model = ResNet101(pretrained=True)
    model.eval()
    model.to(device)

    # read the dataset and initialize the data loader
    dataset = Dataset(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, collate_fn=collate_skip_empty, shuffle=True)

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None

    # we'll also store the image labels and paths to visualize them later
    image_paths = []

    for batch in tqdm(dataloader, desc='Running the model inference'):
        images = batch['image'].to(device)
        image_paths += batch['image_path']

        with torch.no_grad():
            output = model.forward(images)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    return features, image_paths


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_cluster[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, output, plot_size=1000, max_image_size=100):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.savefig(path.join(output, "tsne_images.png"))

def visualize_tsne_points(tx, ty, labels, output):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_cluster:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_cluster[label][::-1]], dtype=float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.savefig(path.join(output, "tsne_points.png"))


def cluster(tsne):
    def scale(x):
        minimum = min(list(x.values()))
        # compute the distribution range
        value_range = (max(list(x.values())) - minimum)

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        temp = {}
        for i in x.keys():
            temp[i] = (x[i] - minimum) / value_range

        # make the distribution fit [0; 1] by dividing by its range
        return temp

    # scale and move the coordinates so they fit [0; 1] range
    # X = scale_to_01_range(tsne[:, 0])
    # y = scale_to_01_range(tsne[:, 1])
    X = scale_to_01_range(tsne)
    k_min = 2
    k_max = 20

    k = range(k_min, k_max + 1)

    y_preds = {}

    # Silhouette Score
    sil = {}
    # DB-Index
    db = {}
    # CH-Index
    ch = {}

    for i in k:
        kmeans = KMeans(n_clusters=i, max_iter=1000, random_state=43).fit(X)
        score = metrics.silhouette_score(X, kmeans.labels_)
        sil[i] = score

        y_pred = KMeans(n_clusters=i, max_iter=1000, random_state=43).fit_predict(X)
        y_preds[i] = y_pred
        score = metrics.davies_bouldin_score(X, y_pred)
        db[i] = -score

        score = metrics.calinski_harabasz_score(X, y_pred)
        ch[i] = score

    sil = scale(sil)
    db = scale(db)
    ch = scale(ch)

    # plt.plot(k, sil, 'o-')
    # plt.xlabel("Value for k")
    # plt.ylabel("Silhouette score")
    # plt.title('Silhouette Method')
    # plt.show()
    #
    # plt.plot(k, db, 'o-')
    # plt.title('DAVIES-BOULDIN')
    # plt.show()
    #
    # plt.plot(k, ch, 'o-')
    # plt.title('CALINSKI-HARABASZ')
    # plt.show()

    # Gap
    optimalK = OptimalK(parallel_backend='None')
    optimalK(X, cluster_array=np.arange(1, k[-1]))

    gap = scale_to_01_range(optimalK.gap_df.gap_value)
    # plt.plot(optimalK.gap_df.n_clusters, gap, linewidth=3)
    #
    # plt.xlabel('Cluster Count')
    # plt.ylabel('Gap Value')
    # plt.title('Gap Values by Cluster Count')
    # plt.show()

    sil_max = 2
    db_max = 2
    ch_max = 2
    gap_max = 2
    ave_max = 2
    ave = {2: (sil[2] + db[2] + ch[2] + gap[2]) / 4}

    for i in range(k_min + 1, k_max + 1):
        if sil[i] > sil[sil_max]:
            sil_max = i
        if db[i] > db[db_max]:
            db_max = i
        if ch[i] > ch[ch_max]:
            ch_max = i
        if gap[i - k_min] > gap[gap_max - k_min]:
            gap_max = i

        ave[i] = (sil[i] + db[i] + ch[i] + gap[i - k_min]) / 4
        if ave[i] > ave[ave_max]:
            ave_max = i

    print(sil_max)
    print(db_max)
    print(ch_max)
    print(gap_max)
    print(ave_max)

    return y_preds[ave_max], ave_max


def visualize_tsne(tsne, images, labels, output, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels, output)

    # visualize the plot: samples as images
    visualize_tsne_images(tx, ty, images, labels, output, plot_size=plot_size, max_image_size=max_image_size)


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default='data')
    parser.add_argument('--output', type=str, default='clusters')
    parser.add_argument('--batch', type=int, default=64)

    args = parser.parse_args()
    
    dataset_name = basename(normpath(args.path))

    output = args.output
    if not isdir(output):
        mkdir(output)
    
    output = join(output, dataset_name)
    if isdir(output):
        rmtree(output)
    if not isdir(output):
        mkdir(output)

    fix_random_seeds()

    features, image_paths = get_features(
        dataset=args.path,
        batch=args.batch
    )

    tsne = TSNE(n_components=2).fit_transform(features)

    clusters, num_clusters = cluster(tsne)

    for i in range(num_clusters):
        hsv = i * 1.0 / num_clusters
        colors_per_cluster[i] = hsv2rgb(hsv, 0.75, 0.75)

        folder = "cluster_" + str(i)

        full_path = path.join(output, folder)

        if not isdir(full_path):
            mkdir(full_path)

    for i in range(len(image_paths)):
        folder = "cluster_" + str(clusters[i])
        file = image_paths[i].split('/')[-1]
        file = path.join(output, folder, file)
        copyfile(image_paths[i], file)

    visualize_tsne(tsne, image_paths, clusters, output)


if __name__ == '__main__':
    main()
