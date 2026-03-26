#!/usr/bin/env python3
"""generate_synthetic.py — Datos sintéticos con subdominios para PC.

Topología:
  tag=1  gel            Esfera central (hidrogel implantado)
  tag=2  lesion         Sustancia blanca periventricular
  tag=3  motor_cortex   Corteza motora M1
"""
from pathlib import Path
import numpy as np
import nibabel as nib
import meshio
from scipy.spatial import Delaunay

def sphere_mask(shape, center, radius):
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    return ((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 <= radius**2).astype(np.uint8)

def generate_nifti_data(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (32, 32, 32)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    center = np.array(shape) // 2
    gel = sphere_mask(shape, center, 5)
    nib.save(nib.Nifti1Image(gel, affine), str(output_dir / "gel_mask.nii.gz"))
    motor = sphere_mask(shape, (24, 24, 16), 4)
    nib.save(nib.Nifti1Image(motor, affine), str(output_dir / "motor_mask.nii.gz"))
    tissue = np.clip(1 - gel - motor, 0, 1).astype(np.uint8)
    nib.save(nib.Nifti1Image(tissue, affine), str(output_dir / "tissue_mask.nii.gz"))
    dti = np.zeros((*shape, 6), dtype=np.float64)
    lambda_ax, lambda_rad = 1.7e-3, 0.3e-3
    for iz in range(shape[2]):
        theta = 2 * np.pi * iz / shape[2]
        e1 = np.array([np.cos(theta), np.sin(theta), 0.0])
        e2 = np.array([-np.sin(theta), np.cos(theta), 0.0])
        e3 = np.array([0.0, 0.0, 1.0])
        D = lambda_ax*np.outer(e1,e1) + lambda_rad*np.outer(e2,e2) + lambda_rad*np.outer(e3,e3)
        dti[:,:,iz,0]=D[0,0]; dti[:,:,iz,1]=D[0,1]; dti[:,:,iz,2]=D[0,2]
        dti[:,:,iz,3]=D[1,1]; dti[:,:,iz,4]=D[1,2]; dti[:,:,iz,5]=D[2,2]
    nib.save(nib.Nifti1Image(dti, affine), str(output_dir / "dti.nii.gz"))
    print(f"NIfTI data: {output_dir} ({shape} volume)")

def generate_mesh_data(output_dir, n_nodes=300):
    output_dir = Path(output_dir)
    rng = np.random.default_rng(2024)
    pts_bulk = rng.uniform(0.5, 7.5, size=(n_nodes-60, 3))
    def sphere_pts(center, r, n):
        th=rng.uniform(0,2*np.pi,n); ph=rng.uniform(0,np.pi,n); rv=rng.uniform(0,r*0.9,n)
        return np.column_stack([center[0]+rv*np.sin(ph)*np.cos(th),
                                center[1]+rv*np.sin(ph)*np.sin(th),
                                center[2]+rv*np.cos(ph)])
    points = np.clip(np.vstack([pts_bulk, sphere_pts([4,4,4],0.9,30), sphere_pts([6,6,4],1.1,30)]), 0.1, 7.9)
    tri = Delaunay(points)
    cells = tri.simplices
    centroids = points[cells].mean(axis=1)
    d_gel   = np.linalg.norm(centroids - np.array([4.0,4.0,4.0]), axis=1)
    d_motor = np.linalg.norm(centroids - np.array([6.0,6.0,4.0]), axis=1)
    markers = np.full(len(cells), 2, dtype=np.int32)
    markers[d_motor < 1.2] = 3
    markers[d_gel   < 1.0] = 1
    print(f"  Subdominios: gel={int((markers==1).sum())} | lesion={int((markers==2).sum())} | motor={int((markers==3).sum())}")
    meshio.write(str(output_dir/"synthetic_mesh.xdmf"), meshio.Mesh(points=points, cells=[("tetra",cells)]))
    meshio.write(str(output_dir/"subdomains.xdmf"),
                 meshio.Mesh(points=points, cells=[("tetra",cells)], cell_data={"markers":[markers]}))
    print(f"Mesh: {output_dir}/synthetic_mesh.xdmf ({len(points)} nodes)")

if __name__ == "__main__":
    out = Path("data/synthetic")
    generate_nifti_data(out)
    generate_mesh_data(out)
    print("Datos sintéticos completos.")
