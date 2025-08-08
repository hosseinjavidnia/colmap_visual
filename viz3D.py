# viz3D.py
#
# Portions adapted from:
#   https://github.com/cvg/Hierarchical-Localization/blob/master/hloc/utils/viz_3d.py
#   Copyright (c) the hloc authors
#
# =============================================================================
# Plot COLMAP reconstructions with Plotly (no external SfM deps)
#
# Description:
#     Lightweight 3D visualizer for COLMAP reconstructions (cameras + points).
#     Works directly with the dict-like objects returned by COLMAP’s
#     `read_write_model.py`.
#
# Features:
#     - Cameras: frusta drawn from intrinsics + world poses.
#     - Points: optional RGB coloring; robust outlier filtering.
#     - Plotly 3D figure with autorange and aspectmode="data".
#
# Usage:
#     from read_write_model import read_model
#     import viz3D
#
#     cams, imgs, pts3D = read_model("/path/to/model", ext=".bin")
#     rec = viz3D.wrap_colmap_dicts(cams, imgs, pts3D)
#
#     fig = viz3D.init_figure(height=800, projection="perspective", template="plotly_dark")
#     viz3D.plot_reconstruction(fig, rec, points_rgb=True)
#     fig.show()
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

from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go

# =============================================================================
# Math & Small Utilities
# =============================================================================


def to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Append a homogeneous 1 to each 2D/3D point: (N, d) -> (N, d+1)."""
    pad = np.ones((points.shape[:-1] + (1,)), dtype=points.dtype)
    return np.concatenate([points, pad], axis=-1)


def _qvec_to_Rcw(qvec):
    """Convert COLMAP quaternion [qw, qx, qy, qz] -> rotation matrix R_cw (world->cam)."""
    qw, qx, qy, qz = map(float, qvec)
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )


def _rgb_triplets_to_plotly(color: Any) -> Any:
    """
    Convert Nx3 RGB arrays to Plotly color strings.
    Accepts uint8 [0..255] or float [0..1]. Leaves non-(N,3) inputs as-is.
    """
    try:
        arr = np.asarray(color)
        if arr.ndim == 2 and arr.shape[1] == 3:
            mx = float(np.nanmax(arr)) if arr.size else 0.0
            if (
                arr.dtype.kind in "fc" and mx <= 1.0 + 1e-6
            ):  # auto-scale [0..1] -> [0..255]
                arr = (arr * 255.0).clip(0, 255)
            return [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in arr]
    except Exception:
        pass
    return color


# =============================================================================
# Minimal Adapter Layer (wraps read_write_model.py outputs)
# =============================================================================


class _Rotation:
    """Tiny wrapper for parity with the original API: exposes .matrix()."""

    def __init__(self, R):
        self._R = np.asarray(R, float)

    def matrix(self):
        return self._R


class _Rigid3d:
    """Rigid transform with rotation matrix and translation vector."""

    def __init__(self, R, t):
        self._R = np.asarray(R, float)
        self.translation = np.asarray(t, float)

    @property
    def rotation(self):
        return _Rotation(self._R)

    def inverse(self):
        """Return the inverse transform (world<-camera)."""
        R = self._R.T
        t = -R @ self.translation
        return _Rigid3d(R, t)


class _Camera:
    """Camera intrinsics adapter (pinhole K from common COLMAP models)."""

    def __init__(self, cam):
        if isinstance(cam, dict):
            self.model = cam.get("model")
            self.params = np.asarray(cam.get("params"), float)
        else:
            self.model = getattr(cam, "model")
            self.params = np.asarray(getattr(cam, "params"), float)

    def calibration_matrix(self):
        """Return 3x3 pinhole K; distortion intentionally ignored for frustum viz."""
        m = str(self.model).upper()
        p = self.params
        if m in (
            "SIMPLE_PINHOLE",
            "SIMPLE_RADIAL",
            "SIMPLE_RADIAL_FISHEYE",
            "FOV",
            "FOV_MODIFIED",
            "THIN_PRISM_FISHEYE",
        ):
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
        elif m in (
            "PINHOLE",
            "OPENCV",
            "OPENCV_FISHEYE",
            "RADIAL",
            "RADIAL_FISHEYE",
            "FULL_OPENCV",
        ):
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        else:
            f, cx, cy = p[0], p[1], p[2]
            fx = fy = f
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], float)


class _Image:
    """Image pose adapter."""

    def __init__(self, im):
        if isinstance(im, dict):
            self.image_id = im.get("image_id", im.get("id"))
            self.camera_id = im["camera_id"]
            self.name = im.get("name")
            self.qvec = im["qvec"]
            self.tvec = im["tvec"]
        else:
            # read_write_model.Image typically uses `id`
            self.image_id = getattr(im, "image_id", getattr(im, "id", None))
            self.camera_id = getattr(im, "camera_id")
            self.name = getattr(im, "name", None)
            self.qvec = getattr(im, "qvec")
            self.tvec = getattr(im, "tvec")
        self._im = im

    def cam_from_world(self):
        """Return camera-from-world rigid transform (R_cw, t_cw)."""
        Rcw = _qvec_to_Rcw(self.qvec)
        tcw = np.asarray(self.tvec, float)
        return _Rigid3d(Rcw, tcw)

    def __str__(self):
        return f"Image(id={self.image_id}, name={self.name})"


class _Track:
    """Track wrapper with the minimal API used by the viewer."""

    def __init__(self, elems):
        self._elems = elems or []

    def length(self):
        return len(self._elems)


class _Point3D:
    """Point3D adapter; supports both {xyz, rgb, error, track} and {image_ids} formats."""

    def __init__(self, p):
        if isinstance(p, dict):
            xyz = p.get("xyz")
            color = p.get("rgb", p.get("color", [255, 255, 255]))
            error = p.get("error", np.inf)
            if p.get("image_ids") is not None:
                track_elems = [0] * len(p["image_ids"])  # only need the length
            else:
                track_elems = p.get("track") or []
        else:
            xyz = getattr(p, "xyz")
            color = getattr(p, "rgb", getattr(p, "color", [255, 255, 255]))
            error = getattr(p, "error", np.inf)
            if hasattr(p, "image_ids") and getattr(p, "image_ids") is not None:
                track_elems = [0] * len(getattr(p, "image_ids"))
            else:
                track_elems = getattr(p, "track", []) or []

        self.xyz = np.asarray(xyz, float)
        self.color = np.asarray(color, float)
        self.error = float(error)
        self.track = _Track(track_elems)


class _BBox:
    """Axis-aligned bounding box with a contains_point() helper."""

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def contains_point(self, xyz):
        x = np.asarray(xyz, float)
        return bool(np.all(x >= self.lo) and np.all(x <= self.hi))


class Reconstruction:
    """
    Adapter that mimics the minimal API used by the original viz file.
    Wraps the outputs of read_write_model.read_model().
    """

    def __init__(
        self, cameras: Dict[int, Any], images: Dict[int, Any], points3D: Dict[int, Any]
    ):
        self.cameras = {cid: _Camera(cam) for cid, cam in cameras.items()}
        self.images = {iid: _Image(im) for iid, im in images.items()}
        self.points3D = {pid: _Point3D(p) for pid, p in points3D.items()}

    def compute_bounding_box(self, qmin: float, qmax: float) -> _BBox:
        """Robust bbox using quantiles (like the original viewer)."""
        if not self.points3D:
            return _BBox(np.array([-1, -1, -1], float), np.array([1, 1, 1], float))
        xyz = np.array([p.xyz for p in self.points3D.values()], float)
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
        lo = np.quantile(xyz, qmin, axis=0)
        hi = np.quantile(xyz, qmax, axis=0)
        return _BBox(lo, hi)


def wrap_colmap_dicts(
    cameras: Dict[int, Any], images: Dict[int, Any], points3D: Dict[int, Any]
) -> Reconstruction:
    """Helper to construct a Reconstruction from read_model() outputs."""
    return Reconstruction(cameras, images, points3D)


# =============================================================================
# Figure & Primitive Drawing
# =============================================================================


def init_figure(
    height: int = 800, projection: str = "perspective", template: str = "plotly_dark"
) -> go.Figure:
    """
    Create and return a Plotly 3D figure with the familiar defaults.
    Set projection to "orthographic" or "perspective".
    """
    fig = go.Figure()
    axes = dict(
        visible=False,
        showbackground=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        autorange=True,
    )
    fig.update_layout(
        template=template,
        height=height,
        scene_camera=dict(
            eye=dict(x=0.0, y=-0.1, z=-2),
            up=dict(x=0, y=-1.0, z=0),
            projection=dict(type=projection),
        ),
        scene=dict(
            xaxis=axes,
            yaxis=axes,
            zaxis=axes,
            aspectmode="data",
            dragmode="orbit",
        ),
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
    )
    return fig


def plot_points(
    fig: go.Figure,
    pts: np.ndarray,
    color: Any = "rgba(255, 0, 0, 1)",
    ps: int = 2,
    colorscale: Optional[str] = None,
    name: Optional[str] = None,
):
    """Add a 3D points trace to the figure."""
    x, y, z = pts.T
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            name=name,
            legendgroup=name,
            marker=dict(
                size=ps,
                color=_rgb_triplets_to_plotly(color),
                line_width=0.0,
                colorscale=colorscale,
            ),
        )
    )


def plot_camera(
    fig: go.Figure,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    legendgroup: Optional[str] = None,
    fill: bool = False,
    size: float = 1.0,
    text: Optional[str] = None,
):
    """
    Draw a camera frustum from pose (R, t) in world and intrinsics K.
    Frustum geometry mirrors the original implementation.
    """
    # Estimate image plane from principal point
    W, H = K[0, 2] * 2, K[1, 2] * 2
    corners = np.array([[0, 0], [W, 0], [W, H], [0, H], [0, 0]])

    # Scale frustum to roughly scene scale (same heuristic as the original)
    if size is not None:
        image_extent = max(size * W / 1024.0, size * H / 1024.0)
        world_extent = max(W, H) / (K[0, 0] + K[1, 1]) / 0.5
        scale = 0.5 * image_extent / world_extent
    else:
        scale = 1.0

    # Back-project image corners to camera rays, then to world
    corners = to_homogeneous(corners) @ np.linalg.inv(K).T
    corners = (corners / 2 * scale) @ R.T + t

    legendgroup = legendgroup if legendgroup is not None else name

    # Lines from camera center to image corners
    x, y, z = np.concatenate(([t], corners)).T
    i = [0, 0, 0, 0]
    j = [1, 2, 3, 4]
    k = [2, 3, 4, 1]

    if fill:
        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                color=color,
                i=i,
                j=j,
                k=k,
                legendgroup=legendgroup,
                name=name,
                showlegend=False,
                hovertemplate=text.replace("\n", "<br>") if text else None,
            )
        )

    # Wireframe edges
    triangles = np.vstack((i, j, k)).T
    vertices = np.concatenate(([t], corners))
    tri_points = np.array([vertices[idx] for idx in triangles.reshape(-1)])
    x, y, z = tri_points.T

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            legendgroup=legendgroup,
            name=name,
            line=dict(color=color, width=1),
            showlegend=False,
            hovertemplate=text.replace("\n", "<br>") if text else None,
        )
    )


# =============================================================================
# High-Level API (same call pattern as the original viewer)
# =============================================================================


def plot_camera_colmap(
    fig: go.Figure, cam_from_world: _Rigid3d, camera: _Camera, **kwargs
):
    """Plot a camera frustum from adapter objects (mirrors original API)."""
    world_t_camera = cam_from_world.inverse()
    plot_camera(
        fig,
        world_t_camera.rotation.matrix(),
        world_t_camera.translation,
        camera.calibration_matrix(),
        **kwargs,
    )


def plot_image_colmap(
    fig: go.Figure, image: _Image, camera: _Camera, name: Optional[str] = None, **kwargs
):
    """Plot a single image’s camera frustum with hover text."""
    plot_camera_colmap(
        fig,
        image.cam_from_world(),
        camera,
        name=name or str(image.image_id),
        text=str(image),
        **kwargs,
    )


def plot_cameras(fig: go.Figure, reconstruction: Reconstruction, **kwargs):
    """Plot all camera frusta in the reconstruction."""
    for _, image in reconstruction.images.items():
        plot_image_colmap(fig, image, reconstruction.cameras[image.camera_id], **kwargs)


def plot_reconstruction(
    fig: go.Figure,
    rec: Reconstruction,
    max_reproj_error: float = 6.0,
    color: str = "rgb(0, 0, 255)",
    name: Optional[str] = None,
    min_track_length: int = 1,
    points: bool = True,
    cameras: bool = True,
    points_rgb: bool = True,
    cs: float = 1.0,
):
    """
    Plot points and cameras with robust filtering (same spirit as the original):
        - BBox:   quantiles [0.001, 0.999]
        - Error:  keep points with reprojection error <= max_reproj_error
        - Tracks: keep points seen in >= min_track_length images
    """
    # Robust bbox from points
    bbs = rec.compute_bounding_box(0.001, 0.999)

    # Filter points
    p3Ds = [
        p3D
        for _, p3D in rec.points3D.items()
        if (
            np.all(np.isfinite(p3D.xyz))
            and bbs.contains_point(p3D.xyz)
            and p3D.error <= max_reproj_error
            and p3D.track.length() >= min_track_length
        )
    ]

    # Fallback: if filters removed everything, at least draw finite points
    if points and not p3Ds:
        p3Ds = [p3D for _, p3D in rec.points3D.items() if np.all(np.isfinite(p3D.xyz))]

    xyzs = [p3D.xyz for p3D in p3Ds]
    if points_rgb:
        pcolor = [p3D.color for p3D in p3Ds]
    else:
        pcolor = color

    if points and len(xyzs):
        plot_points(fig, np.array(xyzs), color=pcolor, ps=1, name=name)

    if cameras:
        plot_cameras(fig, rec, color=color, legendgroup=name, size=cs)


def add_view_buttons(
    fig: go.Figure,
    distance: float | None = None,
    include_projection_toggle: bool = False,
    keep_current_radius: bool = True,
):
    """
    Add Top/Bottom/Left/Right/Front/Back/Reset buttons.
    Keeps the same apparent scale by default (reuses current camera eye norm).
    """
    # current camera (for radius, up vector, and Reset)
    init_cam = (
        fig.layout.scene.camera.to_plotly_json()
        if fig.layout.scene and fig.layout.scene.camera
        else {}
    )

    def _eye_radius(cam_dict):
        e = cam_dict.get("eye", {}) if isinstance(cam_dict, dict) else {}
        try:
            return float(
                (e.get("x", 0.0) ** 2 + e.get("y", 0.0) ** 2 + e.get("z", 0.0) ** 2)
                ** 0.5
            )
        except Exception:
            return None

    # choose the distance to use when snapping views
    r0 = _eye_radius(init_cam)
    if distance is None:
        if keep_current_radius and r0 and r0 > 0:
            distance = r0
        else:
            # fallback: estimate from data extents
            xs, ys, zs = [], [], []
            for tr in fig.data:
                if getattr(tr, "x", None) is not None:
                    xs += [v for v in tr.x if v is not None]
                if getattr(tr, "y", None) is not None:
                    ys += [v for v in tr.y if v is not None]
                if getattr(tr, "z", None) is not None:
                    zs += [v for v in tr.z if v is not None]
            if xs and ys and zs:
                import numpy as _np

                rangemax = max(
                    (_np.nanmax(xs) - _np.nanmin(xs)),
                    (_np.nanmax(ys) - _np.nanmin(ys)),
                    (_np.nanmax(zs) - _np.nanmin(zs)),
                )
                distance = float(rangemax if rangemax > 0 else 2.5) * 1.4
            else:
                distance = 2.5

    # preserve your current "up" vector (you default to y = -1)
    up_current = init_cam.get("up", dict(x=0, y=-1, z=0))

    def _view(x=0, y=0, z=0, up=None):
        cam = dict(eye=dict(x=x, y=y, z=z))
        cam["up"] = up if up is not None else up_current
        return cam

    # canonical directions with the SAME radius
    D = float(distance)
    views = [
        ("Top", _view(0, -distance, 0)),
        ("Bottom", _view(0, +distance, 0)),
        ("Left", _view(-distance, 0, 0)),
        ("Right", _view(+distance, 0, 0)),
        ("Front", _view(0, 0, -distance)),
        ("Back", _view(0, 0, +distance)),
        ("Reset", init_cam),
    ]

    btn_row_1 = dict(
        type="buttons",
        direction="right",
        x=0.01,
        y=0.99,
        xanchor="left",
        yanchor="top",
        buttons=[
            dict(label=label, method="relayout", args=[{"scene.camera": cam}])
            for label, cam in views
        ],
        pad=dict(r=4, t=4),
        showactive=False,
    )

    updatemenus = [btn_row_1]

    if include_projection_toggle:
        proj_buttons = dict(
            type="buttons",
            direction="right",
            x=0.01,
            y=0.94,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="Perspective",
                    method="relayout",
                    args=[{"scene.camera.projection.type": "perspective"}],
                ),
                dict(
                    label="Orthographic",
                    method="relayout",
                    args=[{"scene.camera.projection.type": "orthographic"}],
                ),
            ],
            pad=dict(r=4, t=4),
            showactive=True,
        )
        updatemenus.append(proj_buttons)

    existing = list(fig.layout.updatemenus) if fig.layout.updatemenus else []
    fig.update_layout(updatemenus=existing + updatemenus)
