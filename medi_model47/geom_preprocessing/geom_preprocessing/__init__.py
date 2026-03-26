"""Generación de malla anatómica a partir de segmentaciones NIfTI."""
__version__ = "0.1.0"

from geom_preprocessing.segmentations import SegmentationLoader
from geom_preprocessing.mesh_builder import MeshBuilder
from geom_preprocessing.export import export_to_xdmf
from geom_preprocessing.validate import validate_mesh, MeshValidationReport
