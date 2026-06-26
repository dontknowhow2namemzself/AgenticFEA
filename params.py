"""Single source of truth for all simulation parameters (plan section 0.4).

In the full pipeline this file is the OUTPUT of the parameterization layer
(LLM fills slots -> Pydantic validates -> writes here). For the deterministic-
chain milestone it is hand-written with fixed cantilever-beam values; the
parameterization layer is added only after the chain is verified end to end.

Unit system: SI-mm throughout
    length mm | force N | stress / E in MPa (= N/mm^2)
    density tonne/mm^3 | displacement mm (output)
Any unit conversion anywhere in the pipeline must be commented explicitly.
"""

PARAMS = {
    "analysis": {
        "type": "static",        # static | fatigue
        "nonlinear": False,      # reserved for stage 2 (NLGEOM / plasticity)
    },
    "geometry": {
        "type": "cantilever_beam",       # classification tag; the extension point
        "cantilever_beam": {             # parameter subset bound to that tag
            # Axis convention, fixed across every layer of the pipeline:
            #   x -> beam axis (length L); the beam spans x in [0, L]
            #   y -> width  b  (cross-section centered: y in [-b/2, +b/2])
            #   z -> height h  (load / bending direction; z in [-h/2, +h/2])
            # This keeps I = b*h^3/12 about the y-axis consistent with a -z load.
            "L": 100.0,   # mm
            "h": 10.0,    # mm  (z, height / load direction)
            "b": 10.0,    # mm  (y, width)
        },
        # Future tags: "plate_with_hole": {...}, "free_form": {...}
    },
    "material": {
        "name": "STEEL",
        "E": 210000.0,    # MPa (N/mm^2)
        "nu": 0.3,        # Poisson ratio (dimensionless)
        "rho": 7.85e-9,   # tonne/mm^3 (steel ~ 7.85e-9 in SI-mm)
    },
    "bc": {                              # references neutral geometry face labels
        "fixed": ["face_x_min"],         # -> FIXED nset, mapped by boundary_conditions.py
        "loads": [
            # magnitude in N; dir is a direction vector (normalized downstream)
            {"face": "face_x_max", "dir": [0, 0, -1], "magnitude": 1000.0},
        ],
    },
    "mesh": {
        "elem_size": 2.0,    # mm, target edge length
        "elem_order": 2,     # 2 -> second order
        # elem_shape: meshing strategy.
        #   "hex" (MVP)  -> structured transfinite hexahedra -> CalculiX C3D20R;
        #                   best matches the analytical beam deflection on a box.
        #   "tet" (RESERVED) -> free tetrahedral meshing -> C3D10; needed later
        #                   for non-box geometries (plate_with_hole / free_form)
        #                   that cannot be structured-meshed. Not implemented in
        #                   the MVP; mesh.py raises if selected.
        "elem_shape": "hex",
    },
    "acceptance": {
        "tol_disp": 0.05,    # displacement-only auto-verification threshold (5%)
        # No stress threshold by design: stress exits auto-judgment in the MVP
        # (fixed-end stress concentration diverges under refinement + point-load
        # false singularity). See verification layer, plan section 7.
    },
}
