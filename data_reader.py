import numpy as np
import scipy.io as sio
import spectral as spy
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA
import tifffile

class DataReader():
    def __init__(self):
        self.data_cube = None
        self.g_truth = None

    @property
    def cube(self):
        """
        origin data
        """
        return self.data_cube

    @property
    def truth(self):
        return self.g_truth

    @property
    def normal_cube(self):
        """
        normalization data: range(0, 1)
        """
        return (self.data_cube - np.min(self.data_cube)) / (np.max(self.data_cube) - np.min(self.data_cube))
    @property
    def unit_vector_cube(self):
        """
        Perform per-pixel L2 normalization on the data cube.
        Each pixel vector will be normalized to unit length (L2 norm = 1).

        Returns:
            np.ndarray: Normalized data cube of shape (m, n, d)
        """
        m, n, d = self.data_cube.shape
        img = self.data_cube.reshape((m * n, -1))
        
        # Normalize to [0, 1]
        img = img / img.max()
        
        # Compute L2 norm for each pixel vector
        img_temp = np.sqrt(np.sum(img ** 2, axis=1, keepdims=True))
        img_temp[img_temp == 0] = 1  # Avoid division by zero
        
        # Normalize each pixel vector
        img = img / img_temp
        
        # Reshape back to original cube
        return img.reshape((m, n, d))
        #return self.data_cube
    @property
    def normalized_unit_vector_cube(self):
        """
        First performs global min-max normalization to [0, 1],
        then performs per-pixel L2 normalization (unit vector).
        
        Returns:
            np.ndarray: Normalized data cube of shape (m, n, d)
        """
        # Step 1: Min-Max normalization to [0, 1]
        data_min = np.min(self.data_cube)
        data_max = np.max(self.data_cube)
        norm_cube = (self.data_cube - data_min) / (data_max - data_min)

        # Step 2: Per-pixel L2 normalization
        m, n, d = norm_cube.shape
        img = norm_cube.reshape((m * n, -1))
        
        # Compute L2 norm for each pixel vector
        img_norm = np.linalg.norm(img, axis=1, keepdims=True)
        img_norm[img_norm == 0] = 1  # Avoid division by zero
        
        # Normalize each pixel vector
        img = img / img_norm

        return img.reshape((m, n, d))
class PaviaURaw(DataReader):
    def __init__(self):
        super(PaviaURaw, self).__init__()

        raw_data_package = sio.loadmat(r"./data/paviaU.mat")
        self.data_cube = raw_data_package["paviaU"].astype(np.float32)
        truth = sio.loadmat(r"./data/paviaU_gt.mat")
        self.g_truth = truth["paviaU_gt"].astype(np.float32)




# PCA
def apply_PCA(data, num_components=75):
    new_data = np.reshape(data, (-1, data.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    new_data = pca.fit_transform(new_data)
    new_data = np.reshape(new_data, (data.shape[0], data.shape[1], num_components))
    return new_data, pca


def data_info(train_label=None, val_label=None, test_label=None, start=1):
    class_num = np.max(train_label.astype('int32'))
    if train_label is not None and val_label is not None and test_label is not None:

        total_train_pixel = 0
        total_val_pixel = 0
        total_test_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())
        test_mat_num = Counter(test_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", train_mat_num[i], "\t", val_mat_num[i], "\t", test_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
            total_test_pixel += test_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel, "\t", total_test_pixel)

    elif train_label is not None and val_label is not None:

        total_train_pixel = 0
        total_val_pixel = 0
        train_mat_num = Counter(train_label.flatten())
        val_mat_num = Counter(val_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", train_mat_num[i], "\t", val_mat_num[i])
            total_train_pixel += train_mat_num[i]
            total_val_pixel += val_mat_num[i]
        print("total", "    \t", total_train_pixel, "\t", total_val_pixel)

    elif train_label is not None:
        total_pixel = 0
        data_mat_num = Counter(train_label.flatten())

        for i in range(start, class_num + 1):
            print("class", i, "\t", data_mat_num[i])
            total_pixel += data_mat_num[i]
        print("total:   ", total_pixel)

    else:
        raise ValueError("labels are None")


def draw(label, name: str = "default", scale: float = 4.0, dpi: int = 400, save_img=None):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    if save_img:
        foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)


if __name__ == "__main__":
    data = IndianRaw().cube
    data_gt = IndianRaw().truth
    IndianRaw().data_info(data_gt)
    IndianRaw().draw(data_gt, save_img=None)
    print(data.shape)
    print(data_gt.shape)
