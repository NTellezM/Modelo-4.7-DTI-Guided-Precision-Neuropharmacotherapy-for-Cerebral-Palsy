"""Validación de malla FEM: volúmenes, marcadores, calidad."""

import meshio
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MeshValidationReport:
    n_cells: int = 0
    n_vertices: int = 0
    n_subdomains: int = 0
    min_volume: float = 0.0
    max_aspect_ratio: float = 0.0
    mean_aspect_ratio: float = 0.0
    orphan_facets: int = 0
    passed: bool = False
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _tet_aspect_ratios(points, cells):
    """Calcula el aspect ratio de cada tetraedro.

    Definición: AR = h_max / (3 * r_in)
    donde h_max es la arista más larga y r_in es el radio de la esfera inscrita.
    r_in = 3V / A_total, con V = volumen y A_total = suma de áreas de las 4 caras.

    Un tetraedro regular tiene AR ≈ 2.45. Valores > 10 indican degeneración.
    """
    if len(cells) == 0:
        return np.array([])

    p0 = points[cells[:, 0]]
    p1 = points[cells[:, 1]]
    p2 = points[cells[:, 2]]
    p3 = points[cells[:, 3]]

    # 6 aristas por tetraedro
    edges = np.stack([
        np.linalg.norm(p1 - p0, axis=1),
        np.linalg.norm(p2 - p0, axis=1),
        np.linalg.norm(p3 - p0, axis=1),
        np.linalg.norm(p2 - p1, axis=1),
        np.linalg.norm(p3 - p1, axis=1),
        np.linalg.norm(p3 - p2, axis=1),
    ], axis=1)  # (n_cells, 6)
    h_max = edges.max(axis=1)

    # Volúmenes (vectorizado)
    v01 = p1 - p0
    v02 = p2 - p0
    v03 = p3 - p0
    vols = np.abs(np.einsum("ij,ij->i", v01, np.cross(v02, v03))) / 6.0

    # Áreas de las 4 caras (cada cara es un triángulo)
    def tri_area(a, b, c):
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    a_total = (tri_area(p0, p1, p2) + tri_area(p0, p1, p3)
               + tri_area(p0, p2, p3) + tri_area(p1, p2, p3))

    # Radio inscrito: r_in = 3V / A_total
    r_in = 3.0 * vols / np.maximum(a_total, 1e-30)

    # Aspect ratio: h_max / (3 * r_in)
    ar = h_max / np.maximum(3.0 * r_in, 1e-30)
    return ar


def validate_mesh(mesh_dir, expected_subdomains=None):
    mesh_dir = Path(mesh_dir)
    errors, warnings = [], []

    try:
        mesh = meshio.read(str(mesh_dir / "mesh.xdmf"))
    except Exception as e:
        return MeshValidationReport(errors=[f"No se puede leer mesh.xdmf: {e}"])

    points = mesh.points
    cells = mesh.cells[0].data if mesh.cells else np.empty((0, 4), dtype=np.int64)
    n_c, n_v = len(cells), len(points)

    if n_c == 0:
        errors.append("Malla vacía (0 celdas)")
        return MeshValidationReport(n_cells=0, n_vertices=n_v, errors=errors)

    # Volúmenes (vectorizado)
    v01 = points[cells[:, 1]] - points[cells[:, 0]]
    v02 = points[cells[:, 2]] - points[cells[:, 0]]
    v03 = points[cells[:, 3]] - points[cells[:, 0]]
    vols = np.abs(np.einsum("ij,ij->i", v01, np.cross(v02, v03))) / 6.0

    n_degen = int((vols < 1e-15).sum())
    if n_degen > 0:
        errors.append(f"Celdas degeneradas: {n_degen} con volumen < 1e-15")

    # Aspect ratios
    ar = _tet_aspect_ratios(points, cells)
    max_ar = float(ar.max()) if len(ar) > 0 else 0.0
    mean_ar = float(ar.mean()) if len(ar) > 0 else 0.0

    if max_ar > 100.0:
        errors.append(f"Aspect ratio extremo: max={max_ar:.1f} (>100)")
    elif max_ar > 20.0:
        warnings.append(f"Aspect ratio alto: max={max_ar:.1f} (>20, puede afectar convergencia)")

    # Subdominios
    n_sub = 0
    markers = None
    sub_path = mesh_dir / "subdomains.xdmf"
    if sub_path.exists():
        sub = meshio.read(str(sub_path))
        markers = sub.cell_data.get("markers", [None])[0]
        if markers is not None:
            unique = set(np.unique(markers).astype(int))
            n_sub = len(unique)
            if expected_subdomains and not unique.issubset(set(expected_subdomains)):
                extra = unique - set(expected_subdomains)
                errors.append(f"Marcadores inesperados: {extra}")

    passed = len(errors) == 0
    return MeshValidationReport(
        n_cells=n_c, n_vertices=n_v, n_subdomains=n_sub,
        min_volume=float(vols.min()),
        max_aspect_ratio=max_ar, mean_aspect_ratio=mean_ar,
        orphan_facets=0, passed=passed, errors=errors, warnings=warnings,
    )
