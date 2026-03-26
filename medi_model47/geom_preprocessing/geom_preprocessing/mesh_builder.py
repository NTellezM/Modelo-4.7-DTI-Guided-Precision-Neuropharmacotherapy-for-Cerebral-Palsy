"""Construcción de malla tetraédrica a partir de máscaras binarias con Gmsh."""

import numpy as np
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MeshBuilder:
    """Construye una malla FEM 3D a partir de máscaras de segmentación.

    Workflow:
    1. build_from_masks() o build_synthetic(): define la geometría en Gmsh
    2. generate(): ejecuta el mallado tetraédrico
    3. get_mesh_data(): extrae puntos, celdas y marcadores
    """

    def __init__(self, refinement_interface=0.2, coarse_size=1.5, optimize_netgen=True):
        self.refinement = refinement_interface
        self.coarse_size = coarse_size
        self.optimize = optimize_netgen
        self._points = None
        self._cells = None
        self._cell_markers = None
        self._facet_cells = None
        self._facet_markers = None

    def build_from_masks(self, masks, affine):
        """Construye geometría a partir de máscaras binarias.

        Args:
            masks: dict {nombre: ndarray_binario} (ej. {"gel": ..., "tissue": ..., "nac": ...})
            affine: matriz afín 4×4 (vóxel → mm)
        """
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("brain")

        # Para cada máscara, extraer la isosuperficie y crear un volumen en Gmsh
        tag_map = {}
        for idx, (name, mask) in enumerate(masks.items(), start=1):
            try:
                from skimage import measure
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
            except Exception as e:
                logger.warning("marching_cubes falló para '%s': %s. Usando esfera aproximada.", name, e)
                continue

            # Transformar a coordenadas físicas
            ones = np.ones((len(verts), 1))
            verts_h = np.hstack([verts, ones])
            verts_phys = (affine @ verts_h.T).T[:, :3]

            # Escribir como STL temporal e importar en Gmsh
            stl_path = tempfile.mktemp(suffix=f"_{name}.stl")
            _write_stl(verts_phys, faces, stl_path)
            gmsh.merge(stl_path)

            tag_map[name] = idx
            logger.info("Superficie '%s': %d vértices, %d triángulos → tag %d",
                        name, len(verts_phys), len(faces), idx)

        # Si no hay superficies, crear geometría sintética por defecto
        if not tag_map:
            logger.info("Sin máscaras válidas — generando geometría sintética")
            self.build_synthetic()
            return

        # Operaciones booleanas y mallado
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.coarse_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.refinement)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay

        gmsh.model.mesh.generate(3)

        if self.optimize:
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                logger.warning("Optimización Netgen no disponible")

        self._extract_gmsh_data()
        gmsh.finalize()

    def build_synthetic(self, geometry="sphere_in_cube", params=None):
        """Construye geometría sintética con subdominios conformes.

        Usa gmsh.model.occ.fragment() para crear interfases conformes entre
        gel (tag=1), tejido (tag=2) y NAc (tag=3). Asigna PhysicalGroups
        para que _extract_gmsh_data pueda leer los marcadores de subdominio.
        """
        import gmsh

        if not gmsh.isInitialized():
            gmsh.initialize()
        gmsh.model.add("synthetic")
        gmsh.option.setNumber("General.Verbosity", 1)

        p = params or {}
        L = p.get("cube_size", 10.0)
        r_gel = p.get("gel_radius", 2.0)
        r_nac = p.get("nac_radius", 1.5)
        center = p.get("center", [L / 2, L / 2, L / 2])
        nac_center = p.get("nac_center", [L * 0.75, L * 0.75, L / 2])

        # Crear primitivas
        box = gmsh.model.occ.addBox(0, 0, 0, L, L, L)
        gel = gmsh.model.occ.addSphere(*center, r_gel)
        nac = gmsh.model.occ.addSphere(*nac_center, r_nac)

        # Fragment: corta todos los volúmenes en las intersecciones,
        # creando interfases conformes (las superficies de contacto son compartidas).
        # Retorna (object_dimtags, object_dimtags_map).
        out, out_map = gmsh.model.occ.fragment(
            [(3, box)],            # objeto a fragmentar
            [(3, gel), (3, nac)],   # herramientas de corte
        )
        gmsh.model.occ.synchronize()

        # Después de fragment, los tags originales pueden haber cambiado.
        # out contiene todos los volúmenes resultantes.
        # Identificar cuál es gel, NAc y tejido por posición del centroide.
        gel_tags, nac_tags, tissue_tags = [], [], []
        center_arr = np.array(center)
        nac_arr = np.array(nac_center)

        for dim, tag in out:
            if dim != 3:
                continue
            # Calcular centroide del volumen via bounding box
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(dim, tag)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            cz = (zmin + zmax) / 2
            centroid = np.array([cx, cy, cz])

            d_gel = np.linalg.norm(centroid - center_arr)
            d_nac = np.linalg.norm(centroid - nac_arr)

            if d_gel < r_gel * 0.9:
                gel_tags.append(tag)
            elif d_nac < r_nac * 0.9:
                nac_tags.append(tag)
            else:
                tissue_tags.append(tag)

        # Asignar grupos físicos (determinan los marcadores de subdominio)
        if gel_tags:
            gmsh.model.addPhysicalGroup(3, gel_tags, tag=1, name="gel")
        if tissue_tags:
            gmsh.model.addPhysicalGroup(3, tissue_tags, tag=2, name="tissue")
        if nac_tags:
            gmsh.model.addPhysicalGroup(3, nac_tags, tag=3, name="nac")

        logger.info("Geometría sintética: gel=%s, tissue=%s, nac=%s",
                    gel_tags, tissue_tags, nac_tags)

        # Refinar en interfaz entre subdominios
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.coarse_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.refinement)

        gmsh.model.mesh.generate(3)

        if self.optimize:
            try:
                gmsh.model.mesh.optimize("Netgen")
            except Exception:
                pass

        self._extract_gmsh_data()
        gmsh.finalize()

    def generate(self):
        """Placeholder: el mallado ya se ejecuta en build_from_masks/build_synthetic."""
        pass

    def get_mesh_data(self):
        """Retorna (points, cells, cell_markers, facet_data).

        facet_data = (facet_cells, facet_markers) o (None, None).
        """
        return (self._points, self._cells, self._cell_markers,
                self._facet_cells, self._facet_markers)

    def get_quality_stats(self):
        """Estadísticas de calidad de la malla (vectorizado)."""
        if self._points is None or self._cells is None or len(self._cells) == 0:
            return {"n_cells": 0, "n_vertices": 0}

        p = self._points
        c = self._cells
        v01 = p[c[:, 1]] - p[c[:, 0]]
        v02 = p[c[:, 2]] - p[c[:, 0]]
        v03 = p[c[:, 3]] - p[c[:, 0]]
        vols = np.abs(np.einsum("ij,ij->i", v01, np.cross(v02, v03))) / 6.0

        return {
            "n_cells": len(c),
            "n_vertices": len(p),
            "min_volume": float(vols.min()),
            "max_volume": float(vols.max()),
            "mean_volume": float(vols.mean()),
        }

    def _extract_gmsh_data(self):
        """Extrae datos de Gmsh incluyendo marcadores de PhysicalGroup.

        Itera sobre los grupos físicos 3D y asigna el tag del grupo
        a cada elemento que pertenece a él. Elementos sin grupo reciben
        marcador 0.
        """
        import gmsh

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        self._points = coords.reshape(-1, 3)

        tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

        # Recoger todos los tetraedros de todos los grupos físicos 3D
        phys_groups = gmsh.model.getPhysicalGroups(dim=3)
        all_elem_tags = []
        all_elem_nodes = []
        elem_tag_to_marker = {}

        if phys_groups:
            # Iterar sobre grupos físicos para asignar marcadores
            for dim, phys_tag in phys_groups:
                entities = gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag)
                for ent in entities:
                    et, tags, nts = gmsh.model.mesh.getElements(dim, ent)
                    if not et:
                        continue
                    for i, elem_type in enumerate(et):
                        if elem_type != 4:  # 4 = tetraedro de 4 nodos
                            continue
                        for t in tags[i]:
                            elem_tag_to_marker[int(t)] = phys_tag
                        all_elem_tags.extend(tags[i])
                        all_elem_nodes.extend(nts[i])
        else:
            # Sin grupos físicos: tomar todos los elementos 3D con marcador 1
            et, tags, nts = gmsh.model.mesh.getElements(dim=3)
            if et:
                all_elem_tags = list(tags[0])
                all_elem_nodes = list(nts[0])
                for t in tags[0]:
                    elem_tag_to_marker[int(t)] = 1

        if not all_elem_nodes:
            self._cells = np.empty((0, 4), dtype=np.int64)
            self._cell_markers = np.empty(0, dtype=np.int32)
            return

        # Deduplicar (un elemento puede aparecer en múltiples entidades)
        seen = set()
        unique_tags = []
        unique_nodes = []
        nodes_flat = list(all_elem_nodes)
        n_per = 4
        idx = 0
        for t in all_elem_tags:
            ti = int(t)
            if ti not in seen:
                seen.add(ti)
                unique_tags.append(ti)
                unique_nodes.extend(nodes_flat[idx:idx + n_per])
            idx += n_per

        n_elems = len(unique_tags)
        cells = np.array([tag_to_idx[int(n)] for n in unique_nodes]).reshape(n_elems, n_per)
        markers = np.array([elem_tag_to_marker.get(int(t), 0) for t in unique_tags], dtype=np.int32)

        self._cells = cells
        self._cell_markers = markers
        self._facet_cells = None
        self._facet_markers = None


def _write_stl(verts, faces, path):
    """Escribe una superficie como STL binario."""
    import struct

    with open(path, "wb") as f:
        f.write(b"\0" * 80)  # header
        f.write(struct.pack("<I", len(faces)))
        for face in faces:
            p0, p1, p2 = verts[face[0]], verts[face[1]], verts[face[2]]
            n = np.cross(p1 - p0, p2 - p0)
            norm = np.linalg.norm(n)
            n = n / norm if norm > 1e-30 else np.zeros(3)
            f.write(struct.pack("<fff", *n))
            f.write(struct.pack("<fff", *p0))
            f.write(struct.pack("<fff", *p1))
            f.write(struct.pack("<fff", *p2))
            f.write(struct.pack("<H", 0))
