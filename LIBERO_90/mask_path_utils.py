import cv2
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString


def add_mask_2d_to_img(img, points, mask_pixels=25):
    img_zeros = np.zeros_like(img)
    for point in points:
        x, y = point.astype(np.int16)
        y_minus, y_plus = (
            int(max(0, y - mask_pixels)),
            int(min(img.shape[0], y + mask_pixels)),
        )
        x_minus, x_plus = (
            int(max(0, x - mask_pixels)),
            int(min(img.shape[1], x + mask_pixels)),
        )
        # example for masking out a square
        img_zeros[y_minus:y_plus, x_minus:x_plus] = img[y_minus:y_plus, x_minus:x_plus]
    return img_zeros


def scale_path(path, min_in, max_in, min_out, max_out):
    return (path - min_in) / (max_in - min_in) * (max_out - min_out) + min_out


def smooth_path_rdp(points, tolerance=0.05):
    """
    Simplifies a line using the Ramer-Douglas-Peucker algorithm.

    :param points: List of (x, y) tuples representing the line points
    :param tolerance: The tolerance parameter to determine the degree of simplification
    :return: List of (x, y) tuples representing the simplified line points
    """
    if len(points[0]) == 2:
        line = LineString(points)
        simplified_line = line.simplify(tolerance, preserve_topology=False)
        return np.array(simplified_line.coords)
    else:
        raise NotImplementedError("Only 2D simplification is supported")


def add_path_2d_to_img(img, path, cmap=None, color=None, line_size=4):
    """
    Add 2D path to image.
    - img (np.ndarray): image
    - path (np.ndarray): 2D path
    """

    # copy image
    img_out = img.copy()

    path_len = len(path)

    # setup color(-map)
    if cmap is not None:
        plt_cmap = getattr(plt.cm, cmap)
        norm = plt.Normalize(vmin=0, vmax=path_len - 1)
    elif color is None:
        color = (255, 0, 0)

    # plot path
    for i in range(path_len - 1):
        # get color
        if cmap is not None:
            color = plt_cmap(norm(i))[:3]
            color = tuple(int(c * 255) for c in color)

        cv2.line(
            img_out,
            (int(path[i][0]), int(path[i][1])),
            (int(path[i + 1][0]), int(path[i + 1][1])),
            color,
            line_size,
        )

    return img_out


def add_path_2d_to_img_gradient(
    image, points, line_size=3, circle_size=0, plot_lines=True
):
    img_out = image.copy()

    if np.all(points <= 1):
        points = points * image.shape[1]

    points = points.astype(int)
    path_len = len(points)

    # Generate gradient from dark red to bright red
    reds = np.linspace(64, 255, path_len).astype(int)
    colors = [tuple(int(r) for r in (r_val, 0, 0)) for r_val in reds]

    for i in range(path_len - 1):
        color = colors[i]
        if plot_lines:
            cv2.line(img_out, tuple(points[i]), tuple(points[i + 1]), color, line_size)
        if circle_size > 0:
            cv2.circle(
                img_out,
                tuple(points[i]),
                max(1, circle_size),
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

    # Draw last point
    if circle_size > 0:
        cv2.circle(
            img_out,
            tuple(points[-1]),
            max(1, circle_size),
            colors[-1],
            -1,
            lineType=cv2.LINE_AA,
        )

    return img_out


def process_mask_obs(
    sample_img,
    mask,
    mask_noise_std=0.01,
    mask_rdp_tolerance=0.1,
):
    # sample_img -> HxWx3
    height, width = sample_img.shape[:2]

    # normalize mask
    min_in, max_in = np.zeros(2), np.array([height, width])
    min_out, max_out = np.zeros(2), np.ones(2)
    mask_scaled = scale_path(
        mask, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out
    )

    # add noise to mask
    noise = np.random.normal(0, mask_noise_std, mask.shape)
    mask_noise = mask_scaled + noise

    # simplify mask
    mask_noise = smooth_path_rdp(mask_noise, tolerance=mask_rdp_tolerance)

    # scale mask to img size
    mask_scaled = scale_path(
        mask_noise, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in
    )

    # filter unique points
    unique_mask = np.unique(mask_scaled.astype(np.uint16), axis=0)

    # set size of the mask pixels, 15% of the height
    mask_pixels = int(height * 0.15)

    # apply mask
    sample_img = add_mask_2d_to_img(sample_img, unique_mask, mask_pixels=mask_pixels)

    return sample_img


def process_path_obs(
    sample_img,
    path,
    path_line_size=3,
    path_rdp_tolerance=0.05,
    path_noise_std=0.01,
    path_color_gradient=True,
):
    height, width = sample_img.shape[-3:-1]

    # add noise to path
    # following HAMSTER
    noise = np.random.normal(0, path_noise_std, path.shape[1:])
    path_noise = path + noise[None]

    # simplify path
    path_noise = smooth_path_rdp(path_noise, tolerance=path_rdp_tolerance)

    # scale path to img size
    min_in, max_in = np.zeros(2), np.array([height, width])
    min_out, max_out = np.zeros(2), np.ones(2)
    path_scaled = scale_path(
        path_noise, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in
    )

    # draw path
    zero_img = np.zeros((height, width, 3), dtype=np.uint8)
    if path_color_gradient:
        # following HAMSTER
        sketch = add_path_2d_to_img_gradient(
            zero_img, path_scaled, line_size=path_line_size, circle_size=0
        )
    else:
        sketch = add_path_2d_to_img(
            zero_img, path_scaled, color=(255, 0, 0), line_size=path_line_size
        )
    sketch = np.repeat(sketch[None], len(sample_img), axis=0)

    # add path to image
    mask = sketch[..., 0] > 0
    sample_img[mask] = sketch[mask]

    return sample_img


def get_mask_and_path_from_h5(
    annotation_path: str,
    task_key: str,
    observation: dict,
    demo_key: str,
):
    """
    Helper function to load annotations (path, mask) from separate hdf5 file.
    :param annotation_path: The path to the hdf5 file containing the annotations.
    :param task_key: The key to the task in the hdf5 file.
    :param demo_key: The key to the demo in the hdf5 file.
    Returns:
        masked_imgs: A list of masked images.
        path_imgs: A list of path images.
        masked_path_imgs: A list of masked path images.
        quests: A list of instructions for the subtask.
    """

    # load annotations
    f_annotation = h5py.File(annotation_path, "r", swmr=True)[task_key][demo_key][
        "primary"
    ]
    w, h = f_annotation["masked_frames"].shape[-2:]
    # get sub-traj paths + instructions
    paths = []
    quests = []
    traj_split_indices = f_annotation["traj_splits_indices"][:]
    for split_idx in range(1, len(traj_split_indices)):
        start_idx = traj_split_indices[split_idx - 1]
        end_idx = traj_split_indices[split_idx]
        if split_idx == len(traj_split_indices) - 1:
            end_idx += 1
        curr_path = np.array(
            f_annotation["gripper_positions"][start_idx:end_idx]
        ).copy()
        min_in, max_in = np.zeros(2), np.array([w, h])
        min_out, max_out = np.zeros(2), np.ones(2)
        path_scaled = scale_path(
            curr_path, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out
        )
        # repeat it the num frames times
        path_scaled = np.repeat(path_scaled[None], end_idx - start_idx, axis=0)
        # adjust paths for length mismatch from naive repeating
        for i in range(end_idx - start_idx):
            paths.append(path_scaled[i][i:])
            try:
                quests.append(
                    str(
                        f_annotation["trajectory_labels"][split_idx - 1].decode(
                            "utf-8"
                        )[0]
                    )
                )
            except:
                # if "trajectory_labels" in f_annotation:
                #    print(f_annotation["trajectory_labels"])
                # else:
                #    print("No trajectory labels found")
                quests.append(None)
                # TODO: actually address this issue

    # HACK -> CoPilot generated
    # pad paths to max_path_len using last point -> RDP should remove redundant points
    max_path_len = max([len(p) for p in paths])
    for i, p in enumerate(paths):
        if len(p) < max_path_len:
            paths[i] = np.concatenate(
                [p, np.repeat(p[-1][None], max_path_len - len(p), axis=0)]
            )
        else:
            paths[i] = p[:max_path_len]

    subtask_path_2d = np.stack(paths, axis=0)
    # subtask_start_end_points

    # get full path
    path = f_annotation["gripper_positions"]
    w, h = f_annotation["masked_frames"].shape[-2:]
    min_in, max_in = np.zeros(2), np.array([w, h])
    min_out, max_out = np.zeros(2), np.ones(2)
    path_scaled = scale_path(
        path, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out
    )
    full_path_2d = np.repeat(path_scaled[None], len(observation["ee_pos"]), axis=0)

    images = observation["agentview_rgb"][()][:, ::-1]

    # get mask
    masks = []
    for i in range(len(images)):
        significant_points = f_annotation["significant_points"][i]
        stopped_points = f_annotation["stopped_points"][i]
        # movement_key = "movement_across_video" # "movement_across_subtrajectory"
        # movement_across_video = f_annotation[movement_key]
        mask_points = np.concatenate([significant_points, stopped_points], axis=0)
        unmasked_template = np.ones_like(images[i])
        mask = process_mask_obs(unmasked_template, mask_points)
        # mask = add_mask_2d_to_img(unmasked_template, mask_points)

        masks.append(mask)
    masks = np.stack(masks, axis=0)
    # for now, just return the masked_frames applied to the images
    # masks = f_annotation["masked_frames"][()]
    masked_imgs = []
    path_imgs = []
    masked_path_imgs = []

    for split_idx in range(1, len(traj_split_indices)):
        start_idx = traj_split_indices[split_idx - 1]
        end_idx = traj_split_indices[split_idx]
        if split_idx == len(traj_split_indices) - 1:
            end_idx += 1
        curr_path = np.array(subtask_path_2d[start_idx]).copy()
        masked_imgs.append(images[start_idx:end_idx].copy() * masks[start_idx:end_idx])
        masked_path_imgs.append(process_path_obs(masked_imgs[-1].copy(), curr_path))
        path_imgs.append(process_path_obs(images[start_idx:end_idx].copy(), curr_path))

    masked_imgs = np.concatenate(masked_imgs, axis=0)
    path_imgs = np.concatenate(path_imgs, axis=0)
    masked_path_imgs = np.concatenate(masked_path_imgs, axis=0)

    return masked_imgs, path_imgs, masked_path_imgs, quests
