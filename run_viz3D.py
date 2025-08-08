#!/usr/bin/env python3
# =============================================================================
# COLMAP → Plotly Viewer (runner)
#
# Description:
#     Minimal example to visualize a COLMAP reconstruction using viz3D.py
#     (no external SfM deps). Reads a model with read_write_model.py,
#     wraps it for viz3D, and draws cameras + 3D points.
#
# Usage:
#     python run_viz.py
#
# Requirements:
#     - read_write_model.py available on PYTHONPATH (from COLMAP repo)
#     - viz3D.py
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

# --------------------------- User Config -------------------------------------
# Path to the COLMAP model folder (contains cameras.*, images.*, points3D.*)
MODEL_DIR = "DISK_LightGlue"

# File extension for the model: ".bin" or ".txt"
EXT = ".bin"

# Viewer settings
PROJECTION = "perspective"  # "perspective" or "orthographic"
POINTS_RGB = True  # True: color points from model RGB
POINT_COLOR = "rgba(255,0,0,0.8)"  # used when POINTS_RGB=False
TEMPLATE = "plotly_dark"  # "plotly_white" or "plotly_dark"
FIG_HEIGHT = 800  # pixels (optional tweak)
# -----------------------------------------------------------------------------


def main():
    # 1) Load COLMAP model (cameras, images, points3D)
    cameras, images, points3D = read_model(MODEL_DIR, ext=EXT)

    # 2) Wrap into viz3D's minimal adapter
    rec = viz3D.wrap_colmap_dicts(cameras, images, points3D)

    # 3) Create figure and plot reconstruction
    fig = viz3D.init_figure(height=FIG_HEIGHT, projection=PROJECTION, template=TEMPLATE)
    viz3D.plot_reconstruction(
        fig,
        rec,
        color=POINT_COLOR,
        points_rgb=POINTS_RGB,
    )

    # 4) Show the interactive viewer
    viz3D.add_view_buttons(fig)
    fig.show()


if __name__ == "__main__":
    main()
