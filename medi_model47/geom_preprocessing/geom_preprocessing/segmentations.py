"""Carga de segmentaciones NIfTI y extracción de máscaras binarias."""

import nibabel as nib
import numpy as np
from scipy import ndimage


class SegmentationLoader:
    """Carga una segmentación NIfTI y extrae máscaras por etiqueta."""

    def __init__(self, path, expected_labels=None):
        self.img = nib.load(str(path))
        self.data = np.asarray(self.img.dataobj)
        self.affine = np.asarray(self.img.affine, dtype=np.float64)
        self.voxel_size = tuple(self.img.header.get_zooms()[:3])
        self.expected_labels = expected_labels

        if expected_labels:
            present = set(np.unique(self.data).astype(int))
            expected = set(expected_labels.keys())
            missing = expected - present
            if missing:
                raise ValueError(f"Etiquetas faltantes en la segmentación: {missing}")

    @property
    def shape(self):
        return self.data.shape[:3]

    def get_mask(self, label, smooth=True, closing_r=2, opening_r=1):
        """Extrae máscara binaria para una etiqueta.

        Args:
            label: valor entero de la etiqueta
            smooth: aplicar cierre+apertura morfológica
            closing_r: iteraciones de cierre
            opening_r: iteraciones de apertura
        """
        mask = (self.data == label).astype(np.uint8)

        if smooth:
            struct = ndimage.generate_binary_structure(3, 1)
            if closing_r > 0:
                mask = ndimage.binary_closing(mask, structure=struct, iterations=closing_r).astype(np.uint8)
            if opening_r > 0:
                mask = ndimage.binary_opening(mask, structure=struct, iterations=opening_r).astype(np.uint8)

        return mask

    def extract_surface(self, label, smooth=True):
        """Extrae isosuperficie (marching cubes) de una máscara.

        Returns:
            verts: (N, 3) coordenadas en espacio físico (mm)
            faces: (M, 3) índices de triángulos
        """
        from skimage import measure

        mask = self.get_mask(label, smooth=smooth)
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)

        # Transformar de vóxel a espacio físico
        ones = np.ones((len(verts), 1))
        verts_h = np.hstack([verts, ones])
        verts_phys = (self.affine @ verts_h.T).T[:, :3]

        return verts_phys, faces
