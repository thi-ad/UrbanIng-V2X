#    Urbaning
#    Copyright (C) 2025  Technische Hochschule Ingolstadt
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from urbaning.data.object_label import classes


def draw_dashed_polyline(img, points, color=(0, 0, 0), thickness=1, dash_length=10, gap_length=5, is_closed=False):
    pts = np.asarray(points, np.float32)
    if is_closed:
        pts = np.vstack([pts, pts[0]])

    # Compute segment vectors and lengths
    vecs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(vecs, axis=1) + 1e-3
    seg_dirs = vecs / seg_lengths[:, None]

    # Precompute total lengths
    for p1, direction, seg_len in zip(pts[:-1], seg_dirs, seg_lengths):
        # Vectorized step array for dash start positions
        step = dash_length + gap_length
        starts = np.arange(0, seg_len, step)
        ends = np.minimum(starts + dash_length, seg_len)

        # Compute all start/end points in one go
        start_pts = p1 + starts[:, None] * direction
        end_pts = p1 + ends[:, None] * direction

        # Convert once to int
        lines = np.hstack([start_pts, end_pts]).astype(int)

        # Batch draw with minimal cv2.line calls (no true batch API, but loop minimized)
        for x1, y1, x2, y2 in lines:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    return img


def lanelet_linestring_plot_types(linestring, thickness=1):
    attrs = linestring.attributes
    line_type = attrs["type"]
    subtype = attrs["subtype"] if "subtype" in attrs.keys() else None

    # Default values
    dashes = 3
    color = (0, 0, 0)
    out_thickness = thickness

    # Define colors and styles
    white = (255, 255, 255)
    gray = (125, 125, 125)
    off_white = (250, 250, 250)

    if line_type in (None, "curbstone", "road_border", "guard_rail"):
        color = off_white
    elif line_type == "line_thin":
        color = white
        if subtype == "dashed":
            dashes = 0
    elif line_type == "line_thick":
        color = white
        out_thickness *= 2
        if subtype == "dashed":
            dashes = 0
    elif line_type in ("pedestrian_marking", "bike_marking"):
        color = white
        dashes = 1
    elif line_type == "stop_line":
        color = white
        out_thickness *= 3
    elif line_type == "virtual":
        color = gray
        dashes = 2
    elif line_type == "traffic_sign":
        color = (0, 0, 0)

    return dashes, color, thickness


def generate_colors(index: int, bgr: bool = False) -> tuple:
    """
    Generate a unique color from an index, using a color wheel approach.

    Args:
        index (int): Index representing the number to get a unique color for.
        bgr (bool): Return color in BGR format if True, otherwise RGB.

    Returns:
        tuple: A tuple representing a color (R, G, B) or (B, G, R).
    """
    # Use a color wheel strategy, hue step to vary colors
    hue = 0.41 + index * 0.618033988749895  # Golden ratio conjugate
    hue = hue % 1.0  # Normalize between 0 and 1

    # Convert hue to RGB (HSV to RGB conversion)
    r = int(255 * (0.5 + 0.5 * math.cos(2.0 * math.pi * (hue + 0))))
    g = int(255 * (0.5 + 0.5 * math.cos(2.0 * math.pi * (hue + 1 / 3))))
    b = int(255 * (0.5 + 0.5 * math.cos(2.0 * math.pi * (hue + 2 / 3))))

    return (b, g, r) if bgr else (r, g, b)


object_type2color = {  # bgr
    "Car": (180, 119, 31),                  # Blue
    "Van": (44, 160, 44),                   # Green
    "Bus": (40, 39, 214),                   # Red
    "Truck": (75, 86, 140),                 # Brown
    "Trailer": (232, 199, 174),             # Light blue
    "OtherVehicle": (207, 190, 23),         # Cyan
    "Cyclist": (14, 127, 255),              # Orange
    "Motorcycle": (194, 119, 227),          # Pink
    "EScooter": (138, 223, 152),            # Light green
    "Pedestrian": (189, 103, 148),          # Purple
    "OtherPedestrians": (213, 176, 197),    # Light purple
    "Animal": (34, 189, 188),               # Olive
    "Other": (127, 127, 127),               # Gray
}


def get_color_function(color):
    if isinstance(color, str):
        if color == 'track_id':
            return lambda x: generate_colors(x.track_id)
        elif color == 'class_label':
            return lambda x: object_type2color[x.object_type]
        else:
            raise ValueError(f"Unknown color parameter: '{color}'")
    elif isinstance(color, tuple):
        return lambda x: color