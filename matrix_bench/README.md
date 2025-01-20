```markdown
# Rust Matrix Multiplication Benchmark

A parallel matrix multiplication benchmark using Rust, ndarray, and Rayon.

## Building and Running

### First Time Setup
```bash
# Clone and enter directory
git clone [repository-url]
cd matrix_bench

# Install dependencies and build
cargo build --release
```

### Build Options

#### Debug Build (with symbols, unoptimized)
```bash
cargo build              # build only
cargo run               # build and run
```

#### Release Build (optimized)
```bash
cargo build --release    # build only
cargo run --release     # build and run
```

### Running the Benchmark

#### Via Cargo (checks for changes, may recompile)
```bash
cargo run --release
```

#### Direct Execution (fastest, no compilation check)
```bash
./target/release/matrix_bench
```

### Cleaning
If you need to rebuild from scratch:
```bash
cargo clean             # removes all build artifacts
cargo build --release   # rebuild
```

## Current Benchmark Settings
- Matrix Size: 1000 x 1000
- Data Type: f64 (double precision)
- Iterations: 5
- Parallel execution using Rayon
- Includes correctness verification

## Performance Notes
- Debug builds are ~30x slower than release builds
- Release builds achieve ~2.7 GFLOPS on test hardware
- Uses parallel execution across available CPU cores
```

## Cleaning up

```cargo clean
```
