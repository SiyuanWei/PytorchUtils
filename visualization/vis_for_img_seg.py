import numpy as np
import cv2
import matplotlib.pyplot as plt


def tensor_to_image(tensor,mean, std):
    image = np.transpose(tensor, (1, 2, 0))
    n_channel = image.shape[2]
    for c in range(n_channel):
        image[:, :, c] = (image[:, :, c] * std[c] + mean[c]) * 255
    if n_channel==1:
        image = image.reshape((image.shape[0],image.shape[1]))
    return image.astype(np.uint8)

def save_feature_to_img(feature, name, sigmoid=True):
     feature = feature.clone().cpu().numpy()
     if sigmoid: feature = 1.0 / (1 + np.exp(-1 * feature))
     feature = np.round(feature * 255)
     cv2.imwrite('{}'.format(name), feature)

def visual_comparison_plot(image,groundtruth,label):
    f, axarr = plt.subplots(1, 3)
    for ax in axarr:
        ax._frameon = False  # 边框不显示
        ax.axis('off')
    axarr[0].imshow(image)
    axarr[0].set_title('Image')

    axarr[1].imshow(groundtruth)
    axarr[1].set_title('GroundTruth')

    axarr[2].imshow(label)
    axarr[2].set_title('Predict')
    plt.show()

def color_label(mask_img, color_map):
    assert isinstance(color_map,dict)
    mask_img = np.argmax(mask_img, axis=2)
    h, w = mask_img.shape[0], mask_img.shape[1]
    mask_img = (mask_img).reshape((h, w, 1))

    r_map, g_map, b_map = np.vectorize(color_map[0].get), np.vectorize(color_map[1].get), np.vectorize(color_map[2].get)
    return np.concatenate((r_map(mask_img), g_map(mask_img), b_map(mask_img)), axis=2).astype(np.uint8)