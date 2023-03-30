import cv2
import numpy as np


def get_grid(w, h, homogenous=False) -> np.ndarray:
    """
    Get a grid of points representing the image coordinate
    Args:
        w: width of the image
        h: height of the image
        homogenous: if True, return a 3xN array, else return a 2xN array
    Returns:
        A 2xN array of points representing the image coordinate
    """
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x, y)
    if homogenous:
        return (
            np.stack([xv.flatten(), yv.flatten(), np.ones_like(xv).flatten()])
            .round()
            .astype(int)
        )
    else:
        return np.stack([xv.flatten(), yv.flatten()]).round().astype(int)


def apply_affine_transform_coords(
    coords: np.ndarray, A: np.ndarray, origin: tuple | None = None
) -> np.ndarray:
    """
    Apply affine transformation to a set of points
    Args:
        coords: A Nx2 array of points
        A: A 3x3 affine transformation matrix
        origin: The origin of the transformation
    Returns:
        A Nx2 array of points after transformation
    """
    if origin is not None:
        A = np.dot(
            np.array(
                [
                    [1, 0, origin[0]],
                    [0, 1, origin[1]],
                    [0, 0, 1],
                ]
            ),
            A,
        )
        A = np.dot(
            A,
            np.array(
                [
                    [1, 0, -origin[0]],
                    [0, 1, -origin[1]],
                    [0, 0, 1],
                ]
            ),
        )
    # Append a column of ones to the input coordinates
    ones_column = np.ones((coords.shape[0], 1))
    coords_homogeneous = np.hstack((coords, ones_column))
    # Apply the affine transformation to the coordinates
    transformed_coords = np.dot(A, coords_homogeneous.T).T
    # Divide the transformed coordinates by the last column to get
    # the non-homogeneous coordinates
    transformed_coords /= transformed_coords[:, -1][:, np.newaxis]
    # Extract the first two columns as the transformed points
    transformed_points = transformed_coords[:, :2]
    return transformed_points


def apply_affine_transform_image(
    img: np.ndarray,
    A: np.ndarray,
    center: bool = False,
    fill: np.ndarray | None = None,
    **kwargs
) -> np.ndarray:
    """
    Apply affine transformation to an image
    Args:
        img: The image to be transformed
        A: A 3x3 affine transformation matrix
        center: If True, the transformation is applied to the center of the
            image
        fill: The image to fill the empty space after transformation
    Returns:
        The transformed image
    """
    if center:
        # Move the origin to the center of the image
        A = np.dot(
            np.array(
                [
                    [1, 0, img.shape[1] / 2],
                    [0, 1, img.shape[0] / 2],
                    [0, 0, 1],
                ]
            ),
            A,
        )
        # Move the origin back to the top-left corner of the image
        A = np.dot(
            A,
            np.array(
                [
                    [1, 0, -img.shape[1] / 2],
                    [0, 1, -img.shape[0] / 2],
                    [0, 0, 1],
                ]
            ),
        )
    # Remove the last row of the affine transformation matrix for
    # cv2.warpAffine
    transformed = cv2.warpAffine(
        img, A[:2, :], (img.shape[1], img.shape[0]), borderValue=0, **kwargs
    )
    if fill is not None:
        transformed[transformed == 0] = fill[transformed == 0]

    return transformed
