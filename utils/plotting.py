# https://github.com/NoelShin/reco
# https://github.com/NoelShin/namedmask

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def colourise_label(label: np.ndarray, palette, ignore_index: int = 255) -> np.ndarray:
    h, w = label.shape[-2:]
    coloured_label = np.zeros((h, w, 3), dtype=np.uint8)

    unique_label_ids = np.unique(label)
    for label_id in unique_label_ids:
        if label_id == ignore_index:
            coloured_label[label == label_id] = np.array([255, 255, 255], dtype=np.uint8)
        else:
            coloured_label[label == label_id] = palette[label_id]
    return coloured_label


def render_results(val_img, val_gt, palette):
    pil_img = val_img
    pil_img = pil_img * np.array([0.229, 0.224, 0.225])[:, None, None]
    pil_img = pil_img + np.array([0.485, 0.456, 0.406])[:, None, None]
    pil_img = pil_img * 255.0
    pil_img = np.clip(pil_img, 0, 255)
    ret_img = pil_img.astype(np.uint8).transpose(1, 2, 0)
    val_pil_img: Image.Image = Image.fromarray(ret_img)

    ignore_index = -1
    coloured_gt = colourise_label(val_gt, palette, ignore_index)

    nrows, ncols = 1, 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, figsize=(ncols * 3, nrows * 3))
    for i in range(nrows):
        for j in range(ncols):
            if j == 0:
                ax[i, j].imshow(val_pil_img)
            elif j == 1:
                ax[i, j].imshow(coloured_gt)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout(pad=0.5)
    plt.show()
    return ret_img


def get_legends(unique_labels_gt, palette, label_id_to_cat, num_cols=6, is_voc2012=False, show=True):
    num_colors = len(unique_labels_gt) 
    # Number of columns in each row (you can adjust this)
    num_rows = (num_colors + num_cols - 1) // num_cols  # Calculate the number of rows

    labels = [label_id_to_cat[c] for c in unique_labels_gt]

    if is_voc2012:
        palette = np.array(list(palette.values()))
        # palette = palette[:-1]
    else:
        palette = np.array(palette)
    rgb_array = palette[unique_labels_gt]

    # Calculate figure size based on the number of rows and columns
    fig_width = num_cols * 1.5
    fig_height = num_rows * 1.5

    # Create a figure and axis
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_colors):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        ax.imshow([[rgb_array[i]]])  # Normalize the values to [0, 1]
        ax.axis('off')
        ax.set_title(labels[i])  # Set the title as the label

    # Hide any empty subplots if there are any
    for i in range(num_colors, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])

    # Adjust layout and spacing
    plt.tight_layout()

    if show:
        # Show the RGB colors with labels
        plt.show()
    
    return fig


def render_multiple(val_imgs, preds: dict, palette):
    # pred is a dict of {"name": [pred per image]}

    nrows, ncols = len(preds)+1, len(val_imgs)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    
    for j, val_img in enumerate(val_imgs):
        if j == 0:
            ax[0, j].set_ylabel("Image")
        ax[0, j].imshow(val_img)
        ax[0, j].set_xticks([])
        ax[0, j].set_yticks([])

    for i, (name, pred) in enumerate(preds.items()):
        i = i + 1
        for j, (val_img, val_pred) in enumerate(zip(val_imgs, pred)):
            val_pred = colourise_label(val_pred, palette, -1)
            if j == 0:
                ax[i, j].set_ylabel(name)
            ax[i, j].imshow(val_img)
            ax[i, j].imshow(val_pred, alpha=0.8)
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    plt.tight_layout()
    return fig