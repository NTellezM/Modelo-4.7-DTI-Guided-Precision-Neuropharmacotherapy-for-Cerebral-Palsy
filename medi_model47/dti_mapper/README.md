# dti_mapper — Log-Euclidean DTI Tensor Interpolation

Repositorio completo en Rust + Python. Ver `dti_mapper.tar.gz` para el código fuente.

## Estado: ✅ Validado

- 22 tests Rust + 26 tests Python
- cargo fmt + clippy limpios
- Benchmarks: ~20ms/500k nodos

## Instalación

```bash
cd dti_mapper
python3 -m venv venv && source venv/bin/activate
pip install maturin
maturin develop --release
```

## Verificación

```bash
cargo test && pytest python/tests/ -v
cargo fmt --check && cargo clippy -- -D warnings
```
