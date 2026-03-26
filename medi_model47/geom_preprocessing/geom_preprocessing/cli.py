#!/usr/bin/env python3
"""CLI para geom_preprocessing: build, validate, synthetic."""

import argparse
import sys
from pathlib import Path


def cmd_build(args):
    from geom_preprocessing.segmentations import SegmentationLoader
    from geom_preprocessing.mesh_builder import MeshBuilder
    from geom_preprocessing.export import export_to_xdmf

    seg = SegmentationLoader(args.gel)
    masks = {"gel": seg.get_mask(args.gel_label, smooth=args.smooth)}

    seg_t = SegmentationLoader(args.tissue)
    masks["tissue"] = seg_t.get_mask(args.tissue_label, smooth=args.smooth)

    if args.nac:
        seg_n = SegmentationLoader(args.nac)
        masks["nac"] = seg_n.get_mask(args.nac_label, smooth=args.smooth)

    builder = MeshBuilder(refinement_interface=args.refine, coarse_size=args.coarse)
    builder.build_from_masks(masks, seg.affine)

    output = Path(args.output)
    export_to_xdmf(builder, output)

    from geom_preprocessing.validate import validate_mesh
    report = validate_mesh(output)
    if not report.passed:
        print(f"Validación fallida: {report.errors}", file=sys.stderr)
        sys.exit(1)
    print(f"Malla generada: {report.n_vertices} vértices, {report.n_cells} celdas → {output}")


def cmd_synthetic(args):
    from geom_preprocessing.mesh_builder import MeshBuilder
    from geom_preprocessing.export import export_to_xdmf

    builder = MeshBuilder(refinement_interface=args.refine, coarse_size=args.coarse)
    builder.build_synthetic(geometry=args.geometry)
    export_to_xdmf(builder, Path(args.output))
    print(f"Malla sintética → {args.output}")


def cmd_validate(args):
    from geom_preprocessing.validate import validate_mesh
    report = validate_mesh(args.mesh_dir, expected_subdomains=set(args.expected))
    print(f"Vértices: {report.n_vertices}, Celdas: {report.n_cells}")
    print(f"Volumen mínimo: {report.min_volume:.3e}")
    if report.passed:
        print("PASS")
    else:
        for e in report.errors:
            print(f"  ERROR: {e}")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="geom_preprocessing")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build")
    b.add_argument("--gel", required=True)
    b.add_argument("--tissue", required=True)
    b.add_argument("--nac")
    b.add_argument("--gel-label", type=int, default=1)
    b.add_argument("--tissue-label", type=int, default=1)
    b.add_argument("--nac-label", type=int, default=1)
    b.add_argument("--refine", type=float, default=0.2)
    b.add_argument("--coarse", type=float, default=1.5)
    b.add_argument("--smooth", action="store_true")
    b.add_argument("--output", required=True)
    b.set_defaults(func=cmd_build)

    s = sub.add_parser("synthetic")
    s.add_argument("--geometry", default="sphere_in_cube")
    s.add_argument("--refine", type=float, default=0.2)
    s.add_argument("--coarse", type=float, default=1.5)
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_synthetic)

    v = sub.add_parser("validate")
    v.add_argument("--mesh-dir", required=True)
    v.add_argument("--expected", nargs="+", type=int, default=[1, 2, 3])
    v.set_defaults(func=cmd_validate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
