"""Single source of truth for all simulation parameters (plan section 0.4).

In the full pipeline this file is the OUTPUT of the parameterization layer
(LLM fills slots -> Pydantic validates -> writes here). Two hand-written cases
are prepared below -- the verified cantilever MVP and the plate-with-hole
extension (plate plan M1) -- and ACTIVE_CASE selects which one the pipeline
runs. schema.py validates every case in CASES, not just the active one.

Unit system: SI-mm throughout
    length mm | force N | stress / E in MPa (= N/mm^2)
    density tonne/mm^3 | displacement mm (output)
Any unit conversion anywhere in the pipeline must be commented explicitly.
"""

# Which prepared case the pipeline runs. Flip to "plate_with_hole" once the
# deterministic chain supports it (plate plan M2..M9 land module by module;
# until then mesh.py raises NotImplementedError for the plate).
ACTIVE_CASE = "cantilever_beam"   # "cantilever_beam" | "plate_with_hole"

_CANTILEVER_CASE = {
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
        #   "tet" (RESERVED) -> free tetrahedral meshing -> C3D10; for non-box
        #                   3D geometries. Not implemented; mesh.py raises.
        "elem_shape": "hex",
    },
    "acceptance": {
        "tol_disp": 0.05,    # displacement-only auto-verification threshold (5%)
        # No stress threshold by design: stress exits auto-judgment for the
        # cantilever (fixed-end stress concentration diverges under refinement +
        # point-load singularity). See verification layer, plan section 7.
    },
}

_PLATE_CASE = {
    "analysis": {
        "type": "static",
        "nonlinear": False,
    },
    "geometry": {
        "type": "plate_with_hole",
        "plate_with_hole": {
            # 2D plane stress in the z = 0 plane (plate plan 0.1):
            #   plate spans x in [0, a], y in [0, b]  (corner origin -- keeps the
            #   future CNN grid mapping trivial)
            #   hole center = (xc, yc) in the same corner-origin coordinates;
            #   xc = a/2, yc = b/2 is the centered Kirsch/Howland baseline.
            #   Hard gate (schema): all four ligaments positive --
            #   xc - r, a - xc - r, yc - r, b - yc - r.
            "a": 200.0,   # mm, length along x (load direction)
            "b": 100.0,   # mm, width along y
            "t": 1.0,     # mm, thickness (plane stress *SOLID SECTION)
            "r": 20.0,    # mm, hole radius  (2r/b = 0.4, inside the Howland fit range)
            "xc": 100.0,  # mm, hole-center x (= a/2 -> centered)
            "yc": 50.0,   # mm, hole-center y (= b/2 -> centered)
        },
    },
    "material": {
        "name": "STEEL",
        "E": 210000.0,    # MPa
        "nu": 0.3,
        "rho": 7.85e-9,   # tonne/mm^3
    },
    "bc": {
        # Roller + anchor (plate plan D2): u_x = 0 on the whole left edge plus
        # u_y = 0 at the single node nearest that edge's midpoint. No corner
        # singularity (clean, convergent stress field), and it reproduces the
        # half-symmetry of the double-pulled Howland strip. A plain string
        # ("edge_x_min") would instead clamp all DOFs -- supported but not used.
        "fixed": [
            {"face": "edge_x_min", "dofs": [1]},               # roller
            {"face": "edge_x_min", "dofs": [2], "at": "mid"},  # rigid-body anchor
        ],
        "loads": [
            # Total force in N, applied as a uniform traction over edge_x_max
            # (consistent nodal forces downstream, plate plan D5).
            # sigma_gross = F/(b*t) = 10 MPa; sigma_net = F/((b-2r)*t) ~ 16.7 MPa.
            {"face": "edge_x_max", "dir": [1, 0, 0], "magnitude": 1000.0},
        ],
    },
    "mesh": {
        "elem_size": 4.0,    # mm, far-field target; the hole is refined locally
        "elem_order": 2,     # second order (CPS8R / CPS6)
        # "quad" -> recombined quad-dominant 2D mesh -> CalculiX CPS8R (+ CPS6);
        # "tri"  -> all-triangle fallback -> CPS6 (plate plan D7).
        "elem_shape": "quad",
    },
    "acceptance": {
        "tol_disp": 0.05,    # kept for schema uniformity; plate displacement is
                             # recorded, not auto-judged (no closed form, plan D3)
        # Stress-concentration-factor tolerance vs the Howland/Peterson fit
        # (centered-hole configs only; eccentric holes get an honest N/A).
        # MEASURED across the whole batch sampling range at this mesh setting
        # (M7), rather than guessed:
        #     r  =  8   12    16    20    25    30   mm
        #     2r/b= 0.16 0.24 0.32  0.40  0.50  0.60
        #     err = 0.19 0.00 0.34  0.29  0.18  0.59  %
        # Worst case 0.59%, and the reference fit's own stated accuracy is ~1%,
        # so 3% keeps ~5x margin over observation while still catching real
        # modelling errors (a wrong support, thickness or load edge shifts Kt by
        # tens of percent). The initial 0.08 was a pre-data guess.
        "tol_scf": 0.03,
    },
}

CASES = {
    "cantilever_beam": _CANTILEVER_CASE,
    "plate_with_hole": _PLATE_CASE,
}

PARAMS = CASES[ACTIVE_CASE]
