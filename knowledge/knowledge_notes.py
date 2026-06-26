"""Text knowledge base (plan layers 7-8 support): modeling guidance and failure
patterns, retrieved by vector search to ground reports and diagnostics.

Unlike reference_solutions.py these entries carry NO formula -- they are plain
domain knowledge. The verification layer can surface relevant caveats; the
diagnostics agent retrieves matching failure patterns to ground its root-cause
analysis (classic retrieval-augmented reasoning). Each note is embedded by its
`text`; `type` allows filtering (guidance vs failure_pattern).

Unit system: SI-mm.
"""

NOTES = [
    {
        "id": "note_slender_beam",
        "type": "guidance",
        "title": "Beam theory needs a slender beam",
        "text": ("Euler-Bernoulli beam theory assumes a slender beam, length-to-height "
                 "ratio L/h much greater than 1 (in practice >= about 10). For a stubby "
                 "beam with a low L/h it underestimates deflection because it neglects "
                 "transverse shear deformation; use Timoshenko theory or a 3D benchmark."),
        "source": "Euler-Bernoulli vs Timoshenko beam theory",
    },
    {
        "id": "note_fixed_end_stress",
        "type": "guidance",
        "title": "Fixed-end stress concentration does not converge",
        "text": ("Stress at a fully clamped (fixed) end is a stress concentration whose "
                 "peak von Mises grows as the mesh is refined and does not converge to the "
                 "beam-theory section value Mc/I. Do not auto-verify peak stress there."),
        "source": "stress singularity at a re-entrant / clamped corner",
    },
    {
        "id": "note_point_load_singularity",
        "type": "guidance",
        "title": "A concentrated load is an artificial singularity",
        "text": ("A concentrated point load applied at a node creates an artificial "
                 "numerical singularity; the local peak stress there is mesh-dependent and "
                 "not physical. Judge stress away from the load point, or use a pressure."),
        "source": "point-load singularity in elasticity",
    },
    {
        "id": "note_shear_deformation",
        "type": "guidance",
        "title": "Shear deformation matters at low L/h",
        "text": ("Transverse shear deformation becomes significant for L/h below about 10, "
                 "so the 3D FEM deflection exceeds the Euler-Bernoulli value. A fully "
                 "clamped 3D end also adds local stiffness; the two effects partly cancel "
                 "near L/h = 10."),
        "source": "Timoshenko beam theory",
    },
    {
        "id": "fail_singular_stiffness",
        "type": "failure_pattern",
        "title": "Singular stiffness matrix",
        "text": ("A singular global stiffness matrix (rigid-body modes, zero-pivot, or a "
                 "solver that returns garbage / huge displacements) usually means missing "
                 "or insufficient constraints: the model is under-restrained. Check the "
                 "*BOUNDARY card and the FIXED node set."),
        "source": "linear static FEA, constraint diagnostics",
    },
    {
        "id": "fail_negative_jacobian",
        "type": "failure_pattern",
        "title": "Negative Jacobian / distorted elements",
        "text": ("A negative or near-zero Jacobian indicates inverted or badly distorted "
                 "elements. Reduce the element size, improve mesh quality, or fix the "
                 "geometry; check for slivers at small features."),
        "source": "mesh quality diagnostics",
    },
    {
        "id": "fail_disp_error_low_lh",
        "type": "failure_pattern",
        "title": "Displacement error with a low L/h",
        "text": ("A large displacement error versus beam theory together with a low "
                 "length-to-height ratio indicates the analytical reference is out of its "
                 "validity range (shear deformation neglected), not a solver or mesh bug. "
                 "Use a slender beam or a Timoshenko / 3D reference."),
        "source": "verification scope diagnostics",
    },
    {
        "id": "fail_nonconvergence",
        "type": "failure_pattern",
        "title": "Nonlinear non-convergence",
        "text": ("Solver non-convergence (maximum iterations reached, divergence) in a "
                 "nonlinear step usually means the load increment is too large or the mesh "
                 "too coarse. Reduce the increment or refine the mesh."),
        "source": "nonlinear FEA convergence",
    },
]

_BY_ID = {n["id"]: n for n in NOTES}


def get_note(note_id: str):
    return _BY_ID.get(note_id)
