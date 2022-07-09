from matplotlib import pyplot as plot 
import numpy as np
import cv2


def overlay_heatmap_on_image(img, mask, use_rgb=False):

    # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_OCEAN)
    # extend dimensions, if not rgb
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]"
        )

    overlay = 1-heatmap + img
    overlay = overlay / np.max(overlay)
    return np.uint8(255 * overlay)


def make_detailed_overlay_img(img, mask_0, mask_1, label, classes, use_rgb=True, save_path=None, show_preview=False):

    heatmap_overlayed_0 = overlay_heatmap_on_image(img, mask_0, use_rgb)
    heatmap_overlayed_1 = overlay_heatmap_on_image(img, mask_1, use_rgb)    
    
    fig, axes = plot.subplots(nrows=3, ncols=1, figsize=(5, 5), dpi=100)
    fig.suptitle(label)

    axes[0].imshow(heatmap_overlayed_0)
    axes[0].set_title(f"Class {classes[0]} Attention")

    axes[1].imshow(img)
    axes[1].set_title("Original Image")

    axes[2].imshow(heatmap_overlayed_1)
    axes[2].set_title(f"Class {classes[1]} Attention")

    if show_preview:
        plot.show()

    if save_path is not None:
        fig.savefig(save_path, dpi=1000)
        print("\nSaved", save_path)

    fig.canvas.draw()
    fig_img = np.array(fig.canvas.renderer.buffer_rgba())
    plot.clf()

    return fig_img


def plot_img(img, grayscale=False):

    plot.clf()
    if grayscale:
        plot.imshow(img, cmap='gray')
    else:
        plot.imshow(img)
    plot.show()
