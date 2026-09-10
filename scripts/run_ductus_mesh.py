"""Execute Frontier 4: Microscopic Ductus & 3D Mesh AI CLI.

Demonstrates physical candidate sign pruning from 3D stylus fracture remnants:
- Tests simulated tablet edge fractures against full vector signary ductus templates.
- Calculates kinematic entry scores, tangent angle residuals, and prunes inadmissible signs.
"""

from __future__ import annotations

from ancient_text_lab.ductus import (
    DuctusKinematicSolver,
    FractureSurfaceProfile,
    SignDuctusTemplate,
    StylusTrajectory2D,
    SurvivingStrokeFragment,
)


def run_ductus_demo() -> None:
    print("=================================================================")
    print("   FRONTIER 4: MICROSCOPIC DUCTUS & 3D MESH AI (L0 KINEMATICS)  ")
    print("=================================================================")

    # 1. Sign templates
    t_axe = SignDuctusTemplate(
        sign_id="AB08_DOUBLE_AXE",
        script="linear_a",
        strokes=(
            StylusTrajectory2D((0.5, 0.1), (0.5, 0.5), (0.5, 0.9), 0.8, 90.0, 90.0),
            StylusTrajectory2D((0.2, 0.5), (0.5, 0.5), (0.8, 0.5), 0.6, 0.0, 0.0),
        ),
        description="Linear A/B Double Axe (vertical stem + horizontal crossbar)",
    )

    t_trident = SignDuctusTemplate(
        sign_id="AB28_TRIDENT",
        script="linear_a",
        strokes=(
            StylusTrajectory2D((0.5, 0.2), (0.5, 0.5), (0.5, 0.9), 0.8, 90.0, 90.0),
            StylusTrajectory2D((0.2, 0.2), (0.2, 0.5), (0.5, 0.6), 0.7, 90.0, 45.0),
            StylusTrajectory2D((0.8, 0.2), (0.8, 0.5), (0.5, 0.6), 0.7, 90.0, 135.0),
        ),
        description="Linear A/B Trident (3 vertical tines meeting at base)",
    )

    t_circle = SignDuctusTemplate(
        sign_id="AB77_CIRCLE",
        script="linear_a",
        strokes=(
            StylusTrajectory2D((0.2, 0.2), (0.8, 0.2), (0.5, 0.8), 0.5, 45.0, 135.0),
        ),
        description="Linear A Circle / Wheel",
    )

    solver = DuctusKinematicSolver([t_axe, t_trident, t_circle])

    # 2. Simulate damaged tablet fracture: surviving vertical stroke at (0.5, 0.15)
    print("\n[SCENARIO] Tablet HT 115 has a severe diagonal break across line 3.")
    print("           RTI 3D photogrammetry detects a surviving stylus channel:")
    print("           Contact Point: (0.50, 0.15), Entry Angle: 90.0° (Vertical), Depth: 0.82 mm")

    frag = SurvivingStrokeFragment(
        contact_point=(0.50, 0.15),
        tangent_angle_deg=90.0,
        groove_depth_mm=0.82,
        groove_width_mm=0.35,
    )
    fracture = FractureSurfaceProfile(
        tablet_id="HT_115",
        lacuna_id="lac_03",
        surviving_fragments=(frag,),
        fracture_azimuth_deg=135.0,
    )

    scores = solver.prune_candidates(fracture)

    print("\n[KINEMATIC PRUNING REPORT]")
    print("-----------------------------------------------------------------")
    for s in scores:
        status = "ADMISSIBLE" if s.is_physically_admissible else "PRUNED (Physically Impossible)"
        print(f"  Candidate: {s.sign_id:18} | Score: {s.kinematic_score:.3f} | Angle Err: {s.angle_error_deg:4.1f}° | {status}")

    print("\n[RESULT] Pruned candidate space before textual inference:")
    admissible = [s.sign_id for s in scores if s.is_physically_admissible]
    print(f"         Admissible candidate set: {admissible}")
    print("=================================================================")


if __name__ == "__main__":
    run_ductus_demo()
