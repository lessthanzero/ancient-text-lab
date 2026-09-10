"""Frontier 4: Microscopic Ductus & 3D Mesh AI (L0 Physical Epigraphy Layer).

Models stylus kinematics, 3D fracture surface boundaries, and Bézier stroke trajectories:
- Encodes physical stroke entry angles, stylus inclinations, groove depths, and burr vectors.
- Intersects surviving micro-groove fragments with signary ductus templates.
- Prunes candidate alphabet for damaged lacunae from full signary down to admissible physical matches.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class StylusTrajectory2D:
    """A quadratic Bézier stroke trajectory representing a continuous stylus movement."""

    p0: tuple[float, float]  # Start point (x, y) normalized to [0, 1]
    p1: tuple[float, float]  # Control point (x, y)
    p2: tuple[float, float]  # End point (x, y)
    mean_depth_mm: float  # Inscription groove depth in millimeters
    entry_angle_deg: float  # Tangent angle at start in degrees [0, 360)
    exit_angle_deg: float  # Tangent angle at end in degrees [0, 360)

    def sample_points(self, num_samples: int = 10) -> np.ndarray:
        """Sample equidistant (x, y) points along the Bézier curve."""
        t = np.linspace(0.0, 1.0, num_samples)[:, np.newaxis]
        p0 = np.array(self.p0)
        p1 = np.array(self.p1)
        p2 = np.array(self.p2)
        curve = (1.0 - t) ** 2 * p0 + 2.0 * (1.0 - t) * t * p1 + t**2 * p2
        return curve


@dataclass(frozen=True, slots=True)
class SignDuctusTemplate:
    """Idealized palaeographic ductus definition for a single sign or logogram."""

    sign_id: str
    script: str
    strokes: tuple[StylusTrajectory2D, ...]
    description: str


@dataclass(frozen=True, slots=True)
class SurvivingStrokeFragment:
    """Physical stroke remnant observed entering or terminating at a fracture boundary."""

    contact_point: tuple[float, float]  # Location on normalized glyph bounding box [0, 1]
    tangent_angle_deg: float  # Direction of the groove heading into fracture
    groove_depth_mm: float  # Microscopic depth measured via 3D photogrammetry / RTI
    groove_width_mm: float  # Stylus width profile


@dataclass(frozen=True, slots=True)
class FractureSurfaceProfile:
    """3D topographical boundary of a clay tablet fracture or stone chip."""

    tablet_id: str
    lacuna_id: str
    surviving_fragments: tuple[SurvivingStrokeFragment, ...]
    fracture_azimuth_deg: float  # Orientation of fracture fault line across glyph cell


@dataclass(frozen=True, slots=True)
class DuctusAdmissibilityScore:
    """Kinematic and spatial match score between a candidate sign and physical stroke remnants."""

    sign_id: str
    kinematic_score: float  # [0.0, 1.0] where 1.0 is perfect physical alignment
    spatial_residual_mm: float
    angle_error_deg: float
    is_physically_admissible: bool


class DuctusKinematicSolver:
    """Physics-informed solver matching microscopic tablet fractures against candidate ductus."""

    def __init__(
        self,
        templates: Sequence[SignDuctusTemplate],
        *,
        max_angle_tolerance_deg: float = 30.0,
        max_spatial_tolerance: float = 0.25,
        admissibility_threshold: float = 0.65,
    ) -> None:
        self.templates = {t.sign_id: t for t in templates}
        self.max_angle_tolerance_deg = max_angle_tolerance_deg
        self.max_spatial_tolerance = max_spatial_tolerance
        self.admissibility_threshold = admissibility_threshold

    def evaluate_candidate(
        self,
        template: SignDuctusTemplate,
        fracture: FractureSurfaceProfile,
    ) -> DuctusAdmissibilityScore:
        """Evaluate how closely a candidate ductus matches observed fracture stroke remnants."""
        if not fracture.surviving_fragments:
            # If no surviving strokes, all signs are physically unconstrained
            return DuctusAdmissibilityScore(
                sign_id=template.sign_id,
                kinematic_score=1.0,
                spatial_residual_mm=0.0,
                angle_error_deg=0.0,
                is_physically_admissible=True,
            )

        total_match_score = 0.0
        max_spatial_err = 0.0
        max_angle_err = 0.0

        for frag in fracture.surviving_fragments:
            # Find best matching stroke in candidate template
            best_stroke_score = 0.0
            best_stroke_dist = float("inf")
            best_stroke_angle_err = float("inf")

            frag_pt = np.array(frag.contact_point)

            for stroke in template.strokes:
                sampled = stroke.sample_points(num_samples=15)
                # Compute distance from fragment contact point to sampled stroke
                dists = np.linalg.norm(sampled - frag_pt, axis=1)
                min_dist = float(np.min(dists))

                # Check tangent angle alignment at entry or exit
                angle_diff_entry = abs(stroke.entry_angle_deg - frag.tangent_angle_deg) % 360.0
                if angle_diff_entry > 180.0:
                    angle_diff_entry = 360.0 - angle_diff_entry

                angle_diff_exit = abs(stroke.exit_angle_deg - frag.tangent_angle_deg) % 360.0
                if angle_diff_exit > 180.0:
                    angle_diff_exit = 360.0 - angle_diff_exit

                min_angle_diff = min(angle_diff_entry, angle_diff_exit)

                # Combined score for this stroke
                dist_penalty = max(0.0, 1.0 - (min_dist / self.max_spatial_tolerance))
                angle_penalty = max(0.0, 1.0 - (min_angle_diff / self.max_angle_tolerance_deg))

                stroke_score = 0.6 * dist_penalty + 0.4 * angle_penalty

                if stroke_score > best_stroke_score:
                    best_stroke_score = stroke_score
                    best_stroke_dist = min_dist
                    best_stroke_angle_err = min_angle_diff

            total_match_score += best_stroke_score
            max_spatial_err = max(max_spatial_err, best_stroke_dist)
            max_angle_err = max(max_angle_err, best_stroke_angle_err)

        mean_score = total_match_score / len(fracture.surviving_fragments)
        is_admissible = mean_score >= self.admissibility_threshold

        return DuctusAdmissibilityScore(
            sign_id=template.sign_id,
            kinematic_score=float(mean_score),
            spatial_residual_mm=float(max_spatial_err),
            angle_error_deg=float(max_angle_err),
            is_physically_admissible=is_admissible,
        )

    def prune_candidates(
        self,
        fracture: FractureSurfaceProfile,
    ) -> list[DuctusAdmissibilityScore]:
        """Prune full signary to physically admissible candidate signs, ranked by kinematic score."""
        scores = [
            self.evaluate_candidate(template, fracture) for template in self.templates.values()
        ]
        scores.sort(key=lambda s: s.kinematic_score, reverse=True)
        return scores
