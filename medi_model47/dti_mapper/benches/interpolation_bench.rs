// benches/interpolation_bench.rs
//! Performance benchmarks for the dti_mapper pipeline.
//!
//! Run with: `cargo bench -- interpolation`

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nalgebra::Matrix3;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use dti_mapper::interpolation;
use dti_mapper::tensor_ops;

/// Generate a random SPD tensor: A*A^T + eps*I
fn random_spd(rng: &mut StdRng) -> Matrix3<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let a = Matrix3::from_fn(|_, _| normal.sample(rng));
    a * a.transpose() + Matrix3::identity() * 0.01
}

/// Benchmark precompute_logs for varying grid sizes.
fn bench_precompute_logs(c: &mut Criterion) {
    let mut group = c.benchmark_group("precompute_logs");

    for &n_voxels in &[1_000, 10_000, 100_000, 500_000] {
        let mut rng = StdRng::seed_from_u64(42);
        let mut flat = vec![0.0f64; n_voxels * 9];
        for i in 0..n_voxels {
            let t = random_spd(&mut rng);
            for r in 0..3 {
                for col in 0..3 {
                    flat[i * 9 + r * 3 + col] = t[(r, col)];
                }
            }
        }

        group.throughput(Throughput::Elements(n_voxels as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_voxels}_voxels")),
            &n_voxels,
            |b, &n| {
                b.iter(|| {
                    tensor_ops::precompute_logs_batch(&flat, n, tensor_ops::DEFAULT_EPSILON)
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark interpolation for varying node counts.
fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolate_fem_points");

    // Fixed grid: 64^3 = 262144 voxels
    let grid_side = 64;
    let grid_size = grid_side * grid_side * grid_side;

    let mut rng = StdRng::seed_from_u64(42);
    let log_tensors: Vec<Matrix3<f64>> = (0..grid_size)
        .map(|_| {
            let t = random_spd(&mut rng);
            tensor_ops::tensor_log(&t, tensor_ops::DEFAULT_EPSILON).unwrap()
        })
        .collect();

    for &n_nodes in &[1_000, 10_000, 100_000, 500_000] {
        // Generate random valid indices and weights
        let mut indices = vec![0usize; n_nodes * 8];
        let weights = vec![0.125f64; n_nodes * 8]; // Equal weights

        for i in 0..n_nodes {
            // Random base voxel, then its 8 neighbors
            let base_x = (i * 17) % (grid_side - 1);
            let base_y = (i * 31) % (grid_side - 1);
            let base_z = (i * 13) % (grid_side - 1);

            for (c, (dx, dy, dz)) in [
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            ]
            .iter()
            .enumerate()
            {
                let idx = (base_x + dx) * grid_side * grid_side
                    + (base_y + dy) * grid_side
                    + (base_z + dz);
                indices[i * 8 + c] = idx;
            }
        }

        group.throughput(Throughput::Elements(n_nodes as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_nodes}_nodes")),
            &n_nodes,
            |b, &n| {
                b.iter(|| {
                    interpolation::interpolate_fem_points(
                        &log_tensors,
                        &indices[..n * 8],
                        &weights[..n * 8],
                        n,
                        grid_size,
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark trilinear weight computation.
fn bench_trilinear_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("trilinear_weights");

    let inv_affine = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let grid_shape = (64, 64, 64);

    for &n_points in &[10_000, 100_000, 500_000] {
        let points: Vec<f64> = (0..n_points * 3).map(|i| (i % 63) as f64 + 0.5).collect();

        group.throughput(Throughput::Elements(n_points as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{n_points}_points")),
            &n_points,
            |b, &n| {
                b.iter(|| {
                    interpolation::compute_trilinear_weights(&points, n, &inv_affine, grid_shape)
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_precompute_logs,
    bench_interpolation,
    bench_trilinear_weights
);
criterion_main!(benches);
