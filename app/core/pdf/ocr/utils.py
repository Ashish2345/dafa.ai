"""
Utility functions for OCR processing.

These functions are adapted to be self-contained within the diu_new package.
"""

import re
from collections import defaultdict
from copy import deepcopy
from statistics import median
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from google.cloud.vision_v1.types import AnnotateImageResponse


def break_colons(text: str, box: tuple, space_type: int):
    """
    Divides the word into multiple words if there is a colon.

    This is required to prevent the `combine()` function from clubbing the colon with
    adjacent words.

    Args:
        text: str
            word text
        box: tuple of 4 numbers
            the overall bounding box of the word
        space_type: int, {0,1,2,3,4}
            original space type of the word (see Google OCR docs)

    Returns:
        words: list of (text, box, space_type) tuples
            list of words after separating colons
    """
    # Find colons preceded by non-digits
    matches = list(re.finditer(r"(?<=\D):|^:", text))
    if not matches:
        return [(text, box, space_type)]

    x0, y0, x2, y2 = box
    charwidth = (x2 - x0) / len(text) if len(text) > 0 else 0

    # Initialize a blank output list
    result = []
    for i, match in enumerate(matches):
        start, end = match.start(), match.end()

        # find end of token (eot)
        try:
            eot = matches[i + 1].start()
            space = 2
        except (IndexError, ValueError):
            eot = len(text) - 1
            space = space_type

        # If there is text before the first matching colon, add them to output list
        if i == 0 and start > 0:
            partial_text = text[:start]
            partial_x0 = x0
            partial_x2 = partial_x0 + charwidth * start
            result.append((partial_text, [partial_x0, y0, partial_x2, y2], 2))

        # Add the colon to the output list
        partial_text = ":"
        partial_x0 = x0 + charwidth * start
        partial_x2 = partial_x0 + charwidth
        result.append((partial_text, [partial_x0, y0, partial_x2, y2], 2))

        # check if there is text right of this colon, add it to output list
        if end < eot:
            partial_text = text[end : eot + 1]
            partial_x0 = partial_x2
            partial_x2 = partial_x0 + charwidth * len(partial_text)
            result.append((partial_text, [partial_x0, y0, partial_x2, y2], space))

    return result


def reset_lines(df_word_level: pd.DataFrame, factor: Optional[float] = 1.0) -> List[dict]:
    """
    Assigns the text lying in similar y-range to a same line.

    Args:
        df_word_level: pd.DataFrame
            Pandas dataframe with word-level OCR data
        factor: Optional[float]
            Setting high factor will cause the texts in larger y0's range
            to be assigned same y0 and vice-versa

    Returns:
        List[dict]: List of dictionaries with updated line assignments
    """
    df_word_level_lst = list(df_word_level.itertuples())
    if not df_word_level_lst:
        return []

    th = median([w.y2 - w.y0 for w in df_word_level_lst]) * factor
    rows_dict = defaultdict(list)
    row_key = 0

    for word in sorted(df_word_level_lst, key=lambda w: w.y0):
        if word.y0 - row_key >= 0.3 * th:
            row_key = word.y0
        rows_dict[row_key].append(word)

    rows_list = [sorted(r, key=lambda x: x.x0) for r in rows_dict.values()]

    # Since similar y range are in together, we reset the lines for those
    # The data in namedtuple cannot be modified, convert to dict first
    items_list = []
    for line, (sublist, r) in enumerate(zip(rows_list, rows_dict)):
        for item in sublist:
            item_dict = item._asdict()
            item_dict["line"] = line
            item_dict["y0"] = r
            items_list.append(item_dict)

    return items_list


def sort_df(df: pd.DataFrame, by: List[str] = None) -> pd.DataFrame:
    """
    Sort the DataFrame by given columns after resetting line assignments.

    Args:
        df: pd.DataFrame
            Pandas DataFrame with OCR data
        by: List[str]
            Columns to sort by (default: ["line", "y0", "x0"])

    Returns:
        pd.DataFrame: Sorted DataFrame with updated line and index_sort
    """
    if by is None:
        by = ["line", "y0", "x0"]

    if df is None or df.empty:
        return df

    df = df.copy()
    assert all(col in df.columns for col in by), f"by columns not in df: {by}"

    df = pd.DataFrame(reset_lines(df))
    df = df.sort_values(by=by, ignore_index=True)
    df["index_sort"] = df.index

    return df


def remove_no_break_space(text: str) -> str:
    """
    Remove non-breaking space characters from text.

    Args:
        text: Input text string

    Returns:
        Text with non-breaking spaces replaced with regular spaces
    """
    # Replace various unicode no-break space characters
    return text.replace("\u00a0", " ").replace("\u202f", " ").replace("\ufeff", "")


# =============================================================================
# Orientation Correction Utilities
# =============================================================================


def rotate(
    image: np.ndarray,
    angle: float,
    wrap: bool = True,
    center: Optional[Tuple[float, float]] = None,
    scale: float = 1.0,
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Returns `image` rotated by `angle` (in degrees) in CCW direction.

    Args:
        image: numpy array
            Original image
        angle: float
            Angle (in degrees, counter-clockwise direction) to rotate
        wrap: bool
            Makes sure the original image is not clipped. Adds background as necessary.
        center: tuple
            (center_x, center_y)
        scale: float
            Resize scale
        fill: tuple: (B, G, R)
            Color to fill the extra backgrounds (if `wrap` is True)

    Returns:
        rotated_image: numpy array
            Rotated image
    """
    original_height, original_width = image.shape[:2]
    if center is None:
        center = (original_width / 2, original_height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

    if wrap:
        # grab the sin and cos (ie, the rotation components of the matrix)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        # compute the new bounding dimensions of the image
        new_width = int((original_height * sin) + (original_width * cos))
        new_height = int((original_height * cos) + (original_width * sin))
        # adjust the rotation matrix to take into account translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
    else:
        new_height, new_width = original_height, original_width

    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), borderValue=fill)
    return rotated_image


def get_orientation_angle_from_response(response: "AnnotateImageResponse") -> float:
    """
    Determines the angle by which the image is rotated from Google Vision response.

    Args:
        response: google.cloud.vision_v1.types.AnnotateImageResponse
            Google Vision OCR response object

    Returns:
        angle: float
            The angle from horizontal line (counter-clockwise) in degrees.
                          ← = 90   ↑ = 0    → = -90   ↓ = 180
                          ↖ = 45  ↗ = -45   ↘ = -135   ↙ = 135
            If no text is detected, returns 0.0.
    """
    try:
        words = [
            word
            for page in response.full_text_annotation.pages
            for block in page.blocks
            for paragraph in block.paragraphs
            for word in paragraph.words
            if len(word.symbols) >= 5
        ]
    except AttributeError:
        return 0.0

    angles = []
    for word in words:
        try:
            bottomright, bottomleft = word.bounding_box.vertices[2:4]
            angle = np.rad2deg(np.arctan2(bottomright.y - bottomleft.y, bottomright.x - bottomleft.x))
            angles.append(angle)
        except (AttributeError, IndexError):
            continue

    if not angles:
        return 0.0

    return float(np.median(angles))


def get_orientation_angle_from_df(df: pd.DataFrame) -> float:
    """
    Determines the orientation angle from a DataFrame with 4-point coordinates.

    The DataFrame should have columns: point_x0, point_x1, point_x2, point_x3,
    point_y0, point_y1, point_y2, point_y3.

    The 4 points are arranged as:
        0----------1
        | TEXT     |
        3----------2

    Args:
        df: pd.DataFrame
            OCR DataFrame with 4-point coordinates

    Returns:
        angle: float
            The orientation angle in degrees
    """
    df = df.copy()
    df = df[df["Text"].str.len() > 3]
    angles = []

    required_cols = ["point_x0", "point_x1", "point_x2", "point_x3", "point_y0", "point_y1", "point_y2", "point_y3"]
    if not all(col in df.columns for col in required_cols):
        return 0.0

    for row in df.itertuples():
        try:
            x2, x3 = row.point_x2, row.point_x3
            y2, y3 = row.point_y2, row.point_y3
            # slope is the slope of bottom edge
            angle = np.rad2deg(np.arctan2(y2 - y3, x2 - x3))
            angles.append(angle)
        except AttributeError:
            continue

    if not angles:
        return 0.0
    return float(np.median(angles))


def get_rotation_matrix(angle: float) -> np.ndarray:
    """
    Compute the rotation matrix for a given angle.

    | cos(angle)  -sin(angle) |
    | sin(angle)   cos(angle) |

    Args:
        angle: float
            Angle in degrees (CCW)

    Returns:
        np.ndarray: 2x2 rotation matrix
    """
    angle_rad = np.deg2rad(angle)
    sin, cos = np.sin(angle_rad), np.cos(angle_rad)
    return np.array([[cos, -sin], [sin, cos]])


def get_rotated_coords(coords: np.ndarray, angle: float) -> np.ndarray:
    """
    Compute the coordinates after rotation.

    Args:
        coords: np.array [n x 2]
            list of (x,y) pairs
        angle: float
            Angle in degrees (CCW)

    Returns:
        rotated_coords: np.array [n x 2]
            Coords after rotation
    """
    coords = np.array(coords)
    rotation_matrix = get_rotation_matrix(angle)
    return coords @ rotation_matrix


def get_rotated_df(image: np.ndarray, df: pd.DataFrame, angle: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute OCR dataframe with all coordinates rotated using 2-point method.

    This method uses only top-left (x0, y0) and bottom-right (x2, y2) coordinates
    to transform the bounding boxes.

    Args:
        image: np.array
            Original (rotated) image
        df: pandas.DataFrame
            Original (rotated) OCR dataframe
        angle: float
            Angle in degrees (CCW) to rotate

    Returns:
        Tuple of (rotated_df, rotated_image)
    """
    df = deepcopy(df)
    image_rotated = rotate(image, angle)

    center_y, center_x = [(size / 2) for size in image.shape[:2]]
    center_y_rotated, center_x_rotated = [(size / 2) for size in image_rotated.shape[:2]]

    topleft = df[["x0", "y0"]].values - [center_x, center_y]
    bottomright = df[["x2", "y2"]].values - [center_x, center_y]

    topleft_rotated = get_rotated_coords(topleft, angle)
    topleft_rotated = topleft_rotated + [center_x_rotated, center_y_rotated]

    bottomright_rotated = get_rotated_coords(bottomright, angle)
    bottomright_rotated = bottomright_rotated + [center_x_rotated, center_y_rotated]

    coords_rotated = np.hstack([topleft_rotated, bottomright_rotated])
    df[["x0", "y0", "x2", "y2"]] = np.round(coords_rotated).astype(int)

    return df, image_rotated


def get_rotated_df_4point(image: np.ndarray, df: pd.DataFrame, angle: float) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute OCR dataframe with all coordinates rotated using 4-point method.

    This correction uses all 4 points of each bbox for more accurate rotation.
    The DataFrame should have columns: point_x0, point_x1, point_x2, point_x3,
    point_y0, point_y1, point_y2, point_y3.

    After rotation, (x0, y0, x2, y2) are computed from min/max of rotated points.

    Args:
        image: np.array
            Original (rotated) image
        df: pandas.DataFrame
            Original (rotated) OCR dataframe with 4-point coordinates
        angle: float
            Angle in degrees (CCW) to rotate

    Returns:
        Tuple of (rotated_df, rotated_image)
    """
    df = deepcopy(df)
    image_rotated = rotate(image, angle)

    center_y, center_x = [(size / 2) for size in image.shape[:2]]
    center_y_rotated, center_x_rotated = [(size / 2) for size in image_rotated.shape[:2]]

    # Get all 4 points, with centre of image shifted to origin
    points_0 = df[["point_x0", "point_y0"]].values - [center_x, center_y]
    points_1 = df[["point_x1", "point_y1"]].values - [center_x, center_y]
    points_2 = df[["point_x2", "point_y2"]].values - [center_x, center_y]
    points_3 = df[["point_x3", "point_y3"]].values - [center_x, center_y]

    # Perform rotation and then translate the origin
    points_0_rotated = get_rotated_coords(points_0, angle) + [center_x_rotated, center_y_rotated]
    points_1_rotated = get_rotated_coords(points_1, angle) + [center_x_rotated, center_y_rotated]
    points_2_rotated = get_rotated_coords(points_2, angle) + [center_x_rotated, center_y_rotated]
    points_3_rotated = get_rotated_coords(points_3, angle) + [center_x_rotated, center_y_rotated]

    # Update the 4-point columns
    df["point_x0"] = points_0_rotated[:, 0]
    df["point_x1"] = points_1_rotated[:, 0]
    df["point_x2"] = points_2_rotated[:, 0]
    df["point_x3"] = points_3_rotated[:, 0]

    df["point_y0"] = points_0_rotated[:, 1]
    df["point_y1"] = points_1_rotated[:, 1]
    df["point_y2"] = points_2_rotated[:, 1]
    df["point_y3"] = points_3_rotated[:, 1]

    # Compute (x0, y0, x2, y2) from min/max of all rotated points
    points_x_rot = list(
        zip(
            points_0_rotated[:, 0],
            points_1_rotated[:, 0],
            points_2_rotated[:, 0],
            points_3_rotated[:, 0],
        )
    )
    points_y_rot = list(
        zip(
            points_0_rotated[:, 1],
            points_1_rotated[:, 1],
            points_2_rotated[:, 1],
            points_3_rotated[:, 1],
        )
    )

    topleft_rotated = np.array([(min(xs), min(ys)) for xs, ys in zip(points_x_rot, points_y_rot)])
    bottomright_rotated = np.array([(max(xs), max(ys)) for xs, ys in zip(points_x_rot, points_y_rot)])

    coords_rotated = np.hstack([topleft_rotated, bottomright_rotated])
    df[["x0", "y0", "x2", "y2"]] = np.round(coords_rotated).astype(int)

    return df, image_rotated
