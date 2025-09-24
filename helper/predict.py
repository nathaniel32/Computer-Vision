import matplotlib.pyplot as plt
import cv2

def plot_predictions(image_orig, orig_size, mask_pred, title):
    # resize mask ke ukuran asli
    mask_resized = cv2.resize(mask_pred, orig_size)

    # plot
    plt.figure(figsize=(8,8))
    plt.imshow(image_orig)
    plt.imshow(mask_resized, cmap='jet', alpha=0.5)
    plt.title(title, fontsize=10)
    plt.axis('off')
    plt.show()