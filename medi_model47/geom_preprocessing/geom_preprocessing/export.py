"""Exportación de malla a formato XDMF compatible con FEniCS."""

import meshio
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def export_to_xdmf(builder, output_dir):
    """Exporta malla y marcadores a XDMF (3 archivos)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points, cells, cell_markers, facet_cells, facet_markers = builder.get_mesh_data()

    if points is None or cells is None or len(cells) == 0:
        raise ValueError("La malla está vacía — no se puede exportar")

    # mesh.xdmf
    mesh = meshio.Mesh(points.astype(np.float64), [("tetra", cells)])
    meshio.xdmf.write(str(output_dir / "mesh.xdmf"), mesh)
    logger.info("mesh.xdmf: %d vértices, %d tetraedros", len(points), len(cells))

    # subdomains.xdmf
    if cell_markers is None:
        cell_markers = np.ones(len(cells), dtype=np.int32)
    sub = meshio.Mesh(
        points.astype(np.float64),
        [("tetra", cells)],
        cell_data={"markers": [cell_markers.astype(np.int32)]},
    )
    meshio.xdmf.write(str(output_dir / "subdomains.xdmf"), sub)
    logger.info("subdomains.xdmf: valores %s", np.unique(cell_markers))

    # facets.xdmf (si existen)
    if facet_cells is not None and len(facet_cells) > 0:
        if facet_markers is None:
            facet_markers = np.zeros(len(facet_cells), dtype=np.int32)
        fac = meshio.Mesh(
            points.astype(np.float64),
            [("triangle", facet_cells)],
            cell_data={"markers": [facet_markers.astype(np.int32)]},
        )
        meshio.xdmf.write(str(output_dir / "facets.xdmf"), fac)
        logger.info("facets.xdmf: %d triángulos", len(facet_cells))
