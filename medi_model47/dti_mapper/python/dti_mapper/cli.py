"""Command-line interface for dti_mapper.

Usage:
    dti-mapper map --dti data/dti.nii.gz --mesh mesh/mesh.xdmf --output tensors.h5
    dti-mapper validate --tensors tensors.h5 --mesh mesh/mesh.xdmf
    dti-mapper info --tensors tensors.h5
"""

from __future__ import annotations

import logging
import sys

import click


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """dti_mapper — Log-Euclidean DTI tensor interpolation onto FEM meshes."""


@main.command()
@click.option("--dti", required=True, type=click.Path(exists=True),
              help="Path to DTI NIfTI file (.nii or .nii.gz)")
@click.option("--mesh", required=True, type=click.Path(exists=True),
              help="Path to mesh XDMF file")
@click.option("--output", required=True, type=click.Path(),
              help="Output path for interpolated tensors")
@click.option("--epsilon", default=1e-12, type=float,
              help="Eigenvalue clamping threshold (default: 1e-12)")
@click.option("--format", "output_format", default="h5",
              type=click.Choice(["h5", "npy"]),
              help="Output format (default: h5)")
@click.option("--validate/--no-validate", default=True,
              help="Validate output tensors (default: enabled)")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def map(
    dti: str,
    mesh: str,
    output: str,
    epsilon: float,
    output_format: str,
    validate: bool,
    log_level: str,
) -> None:
    """Map DTI tensors to FEM mesh nodes."""
    _setup_logging(log_level)

    from dti_mapper.pipeline import map_tensors_to_mesh

    result = map_tensors_to_mesh(
        dti_path=dti,
        mesh_path=mesh,
        output_path=output,
        epsilon=epsilon,
        validate=validate,
        output_format=output_format,
    )

    if result.validation and not result.validation.get("all_valid", False):
        click.echo("WARNING: Tensor validation failed", err=True)
        sys.exit(1)

    click.echo(f"Done in {result.elapsed_total_s:.1f}s → {result.output_path}")


@main.command()
@click.option("--tensors", required=True, type=click.Path(exists=True),
              help="Path to tensor file (.h5 or .npy)")
@click.option("--mesh", required=True, type=click.Path(exists=True),
              help="Path to mesh XDMF file")
@click.option("--log-level", default="INFO",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def validate(tensors: str, mesh: str, log_level: str) -> None:
    """Validate interpolated tensors against the data contract."""
    _setup_logging(log_level)

    from dti_mapper.validate import validate_tensor_output

    report = validate_tensor_output(tensors, mesh)
    click.echo(report.summary())

    if not report.passed:
        sys.exit(1)


@main.command()
@click.option("--tensors", required=True, type=click.Path(exists=True),
              help="Path to tensor HDF5 file")
def info(tensors: str) -> None:
    """Display metadata from a tensor HDF5 file."""
    from dti_mapper.io import load_tensors_h5

    data, metadata = load_tensors_h5(tensors)
    click.echo(f"Shape: {data.shape}")
    click.echo(f"Dtype: {data.dtype}")
    click.echo(f"Size: {data.nbytes / 1e6:.2f} MB")
    click.echo("Metadata:")
    for k, v in sorted(metadata.items()):
        click.echo(f"  {k}: {v}")


if __name__ == "__main__":
    main()
