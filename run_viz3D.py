#!/usr/bin/env python3
# =============================================================================
# COLMAP → Plotly Viewer (runner)
#
# Description
# ----------
# Minimal example to visualize a COLMAP reconstruction using viz3D.py
# (no external SfM deps). Reads a model with read_write_model.py, wraps
# it for viz3D, and draws cameras + 3D points. Optionally overlays the
# original images as textured billboards at each camera’s frustum plane.
#
# Usage
# -----
#   python run_viz.py
#
# Requirements
# ------------
# - read_write_model.py available on PYTHONPATH (from COLMAP repo)
# - viz3D.py (this project)
#
# Author: Hossein Javidnia
# Contact: hosseinjavidnia@gmail.com
# Copyright (c) 2025 Trinity College Dublin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import viz3D
from read_write_model import read_model

# -----------------------------------------------------------------------------
# User Config
# -----------------------------------------------------------------------------
# Path to the COLMAP model folder (contains cameras.*, images.*, points3D.*)
MODEL_DIR = "DISK_LightGlue"

# File extension for the model: ".bin" or ".txt"
EXT = ".bin"

# Viewer appearance / behavior
PROJECTION = "perspective"  # {"perspective", "orthographic"}
POINTS_RGB = True  # True: color points from model RGB; False: use POINT_COLOR
POINT_COLOR = "rgba(255,0,0,0.8)"  # Used when POINTS_RGB=False
TEMPLATE = "plotly_white"  # {"plotly_white", "plotly_dark", ...}
FIG_HEIGHT = 900  # Canvas height in pixels

# Image billboards (optional overlay of original images on frusta)
IMAGE_ROOT = "paris_tram_1/jpg"  # Folder containing the original image files
SHOW_IMAGE_BILLBOARDS = True  # Toggle image billboards on/off
# -----------------------------------------------------------------------------


def main() -> None:
    """
    Load a COLMAP model, build a Plotly 3D figure with cameras + points,
    optionally overlay image billboards, and show the interactive viewer.
    """
    # 1) Load COLMAP model (cameras, images, points3D)
    cameras, images, points3D = read_model(MODEL_DIR, ext=EXT)

    # 2) Wrap into viz3D's minimal adapter
    rec = viz3D.wrap_colmap_dicts(cameras, images, points3D)

    # 3) Create figure and plot reconstruction (points + camera frusta)
    fig = viz3D.init_figure(height=FIG_HEIGHT, projection=PROJECTION, template=TEMPLATE)
    viz3D.plot_reconstruction(
        fig,
        rec,
        color=POINT_COLOR,
        points_rgb=POINTS_RGB,
    )

    # 3b) (Optional) Overlay original images as textured billboards
    if SHOW_IMAGE_BILLBOARDS:
        viz3D.plot_image_billboards(
            fig,
            rec,
            image_root=IMAGE_ROOT,
            every_n=1,  # Draw 1 out of N images (increase N for speed on large scenes)
            grid=(
                48,
                48,
            ),  # Quad subdivision for billboard sharpness (higher = crisper, heavier)
            max_side=256,  # Downsample cap for source image (largest side in pixels)
            opacity=1.0,  # 1.0 = fully opaque
            size=1.0,  # Tied to frustum scale heuristic
        )

    # 3c) Add legend group toggling (click group names to show/hide layers)
    viz3D.add_layer_legend(fig)

    # 4) Add view buttons (Top/Bottom/Left/Right/Front/Back/Reset)
    viz3D.add_view_buttons(fig)

    # 4b) Add layer toggles row below the view buttons (Cams / Points / Images)
    viz3D.add_layer_buttons_below_views(fig)

    # 5) Show the interactive viewer
    fig.show()


if __name__ == "__main__":
    main()
