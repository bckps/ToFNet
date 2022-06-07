import h5py
import numpy as np
from matplotlib import pyplot as plt


def show_data(mat):

    print(sorted(mat.keys()))
    print(type(np.array(mat['cap'])))
    cap_data = np.squeeze(mat['cap'])
    cap_ideal_data = np.squeeze(mat['cap_ideal'])
    depth_data = np.squeeze(mat['depth_1024'])
    path = np.squeeze(mat['path'])

    plt.figure()
    plt.imshow(depth_data[:,:].T)
    plt.show()
    t = np.array(depth_data[:,:])
    print(t)

    plt.figure()
    plt.subplot(141)
    plt.imshow(cap_data[0,:,:].T)
    plt.axis('off')
    plt.title('40MHz:0')
    plt.subplot(142)
    plt.imshow(cap_data[1,:,:].T)
    plt.axis('off')
    plt.title('40MHz:pi/2')
    plt.subplot(143)
    plt.imshow(cap_data[2,:,:].T)
    plt.axis('off')
    plt.title('70MHz:0')
    plt.subplot(144)
    plt.imshow(cap_data[3,:,:].T)
    plt.axis('off')
    plt.title('70MHz:pi/2')
    plt.show()


    print(cap_data.shape)
    print(cap_ideal_data.shape)
    print(depth_data.shape)
    print(path)

if __name__ == '__main__':
    mat = h5py.File(r'C:\Users\919im\Documents\local-香川研\dataset_shizuoka\bathroom\bathroom_1.mat', 'r')
    show_data(mat)