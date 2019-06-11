from tensorflow.examples.tutorials.mnist import input_data

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox


def visualize(embed, x_test):

    # two ways of visualization: scale to fit [0,1] scale
    # feat = embed - np.min(embed, 0)
    # feat /= np.max(feat, 0)

    # two ways of visualization: leave with original scale
    feat = embed
    ax_min = np.min(embed, 0)    # [-11.579187 -9.2086525] 0 means col, 1 means row
    ax_max = np.max(embed, 0)    # [7.0608773 5.296893 ]
    ax_dist_sq = np.sum((ax_max-ax_min)**2)    # 557.86285

    plt.figure()
    ax = plt.subplot(111)
    shown_images = np.array([[1., 1.]])
    for i in range(feat.shape[0]):
        dist = np.sum((feat[i] - shown_images)**2, 1)    # 0-x axis project, 1-y axis
        if np.min(dist) < 3e-4*ax_dist_sq:   # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [feat[i]]]    # concatenate mat
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(x_test[i], zoom=0.6, cmap=plt.cm.gray_r),
            xy=feat[i], frameon=False
        )
        ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    # plt.xticks([]), plt.yticks([])
    plt.title('Embedding from the last layer of the network')
    plt.show()


if __name__ == "__main__":

    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    x_test = mnist.test.images    # shape of mnist.test.images is [10000, 784]
    x_test = x_test.reshape([-1, 28, 28])    # shape of x_test is [10000, 28, 28]

    embed = np.fromfile('embed.txt', dtype=np.float32)    # shape is (20000,)
    embed = embed.reshape([-1, 2])    # shape is (10000, 2)

    visualize(embed, x_test)
