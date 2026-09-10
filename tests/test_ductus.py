"""Tests for Frontier 4: Microscopic Ductus & 3D Mesh AI."""

from __future__ import annotations

from ancient_text_lab.ductus import (
    DuctusKinematicSolver,
    FractureSurfaceProfile,
    SignDuctusTemplate,
    StylusTrajectory2D,
    SurvivingStrokeFragment,
)


def test_ductus_kinematic_matching_and_pruning() -> None:
    # Template 1: Double Axe (AB08) - vertical stem and crossbars
    stem_stroke = StylusTrajectory2D(
        p0=(0.5, 0.1),
        p1=(0.5, 0.5),
        p2=(0.5, 0.9),
        mean_depth_mm=0.8,
        entry_angle_deg=90.0,
        exit_angle_deg=90.0,
    )
    cross_stroke = StylusTrajectory2D(
        p0=(0.2, 0.5),
        p1=(0.5, 0.5),
        p2=(0.8, 0.5),
        mean_depth_mm=0.6,
        entry_angle_deg=0.0,
        exit_angle_deg=0.0,
    )
    template_axe = SignDuctusTemplate(
        sign_id="AB08_DOUBLE_AXE",
        script="linear_a",
        strokes=(stem_stroke, cross_stroke),
        description="Cruciform double axe with vertical stem",
    )

    # Template 2: Circle / Ring (AB77) - circular curve
    arc_stroke = StylusTrajectory2D(
        p0=(0.2, 0.2),
        p1=(0.8, 0.2),
        p2=(0.5, 0.8),
        mean_depth_mm=0.4,
        entry_angle_deg=45.0,
        exit_angle_deg=135.0,
    )
    template_circle = SignDuctusTemplate(
        sign_id="AB77_CIRCLE",
        script="linear_a",
        strokes=(arc_stroke,),
        description="Curvilinear circular ring",
    )

    solver = DuctusKinematicSolver([template_axe, template_circle])

    # Case A: Fracture showing a vertical stroke entering at top center (contact=(0.5, 0.15), angle=90 deg)
    frag = SurvivingStrokeFragment(
        contact_point=(0.5, 0.15),
        tangent_angle_deg=90.0,
        groove_depth_mm=0.75,
        groove_width_mm=0.3,
    )
    fracture = FractureSurfaceProfile(
        tablet_id="HT_115",
        lacuna_id="lacuna_01",
        surviving_fragments=(frag,),
        fracture_azimuth_deg=180.0,
    )

    scores = solver.prune_candidates(fracture)
    assert len(scores) == 2

    # Double Axe must rank higher than Circle
    assert scores[0].sign_id == "AB08_DOUBLE_AXE"
    assert scores[0].is_physically_admissible
    assert scores[0].kinematic_score > scores[1].kinematic_score
    assert scores[0].angle_error_deg < 5.0
