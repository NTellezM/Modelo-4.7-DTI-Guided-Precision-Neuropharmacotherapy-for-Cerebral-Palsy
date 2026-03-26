#!/usr/bin/env python3
"""convert_mesh.py — Convierte malla meshio a formato compatible con FEniCS."""

from __future__ import annotations
import argparse
import logging
from pathlib import Path
import meshio
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("convert_mesh")


def convert_mesh(input_path, output_dir, cell_type="tetra"):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Leyendo malla desde %s", input_path)
    mesh = meshio.read(str(input_path))

    points = mesh.points
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])

    logger.info("  Puntos: %d, dimensión: %d", len(points), points.shape[1])

    cells_found = {}
    for cell_block in mesh.cells:
        cells_found[cell_block.type] = cell_block.data
        logger.info("  Tipo '%s': %d celdas", cell_block.type, len(cell_block.data))

    if cell_type not in cells_found:
        raise ValueError(f"Tipo '{cell_type}' no encontrado. Disponibles: {list(cells_found.keys())}")

    cells = cells_found[cell_type]

    # Malla volumétrica
    mesh_path = output_dir / "mesh.xdmf"
    vol_mesh = meshio.Mesh(points=points.astype(np.float64), cells=[(cell_type, cells)])
    meshio.xdmf.write(str(mesh_path), vol_mesh)
    logger.info("Malla volumétrica: %s", mesh_path)

    # Marcadores de subdominio
    subdomains_path = output_dir / "subdomains.xdmf"
    has_subdomains = False

    for key in ["subdomain", "gmsh:physical", "medit:ref", "cell_tags", "markers"]:
        if key in (mesh.cell_data or {}):
            for i, cb in enumerate(mesh.cells):
                if cb.type == cell_type:
                    markers = mesh.cell_data[key][i].astype(np.int32)
                    sub_mesh = meshio.Mesh(
                        points=points.astype(np.float64),
                        cells=[(cell_type, cells)],
                        cell_data={"markers": [markers]},
                    )
                    meshio.xdmf.write(str(subdomains_path), sub_mesh)
                    has_subdomains = True
                    logger.info("Marcadores: %s (valores: %s)", subdomains_path, np.unique(markers))
                    break
            break

    if not has_subdomains:
        # Buscar subdomains.xdmf hermano del archivo de entrada
        sibling = input_path.parent / "subdomains.xdmf"
        if sibling.exists():
            try:
                sub = meshio.read(str(sibling))
                m = sub.cell_data.get("markers", [None])[0]
                if m is not None:
                    markers = m.astype(np.int32)
                    sub_mesh = meshio.Mesh(
                        points=points.astype(np.float64),
                        cells=[(cell_type, cells)],
                        cell_data={"markers": [markers]},
                    )
                    meshio.xdmf.write(str(subdomains_path), sub_mesh)
                    logger.info("Marcadores leídos de %s: %s", sibling, np.unique(markers))
                    has_subdomains = True
            except Exception as e:
                logger.warning("No se pudo leer %s: %s", sibling, e)

    if not has_subdomains:
        markers = np.ones(len(cells), dtype=np.int32)
        sub_mesh = meshio.Mesh(
            points=points.astype(np.float64),
            cells=[(cell_type, cells)],
            cell_data={"markers": [markers]},
        )
        meshio.xdmf.write(str(subdomains_path), sub_mesh)
        logger.info("Sin marcadores — asignado dominio=1 a todas las celdas")

    logger.info("Conversión completa → %s", output_dir)
    return {"mesh": mesh_path, "subdomains": subdomains_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cell-type", default="tetra")
    args = parser.parse_args()
    convert_mesh(args.input, args.output_dir, cell_type=args.cell_type)
