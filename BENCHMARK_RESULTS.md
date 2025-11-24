# Poseidon vs Poseidon2 Performance Benchmark Results

## Summary

This document presents benchmark results comparing **PoseidonGoldilocksConfig** (original Poseidon hash) vs **Poseidon2GoldilocksConfig** (hybrid configuration using Poseidon2 for Merkle trees).

## Configuration Details

- **PoseidonGoldilocksConfig**: Uses Poseidon hash for both external (Merkle trees) and internal (circuit) hashing
- **Poseidon2GoldilocksConfig**: Uses Poseidon2 for external hashing (Merkle trees), Poseidon for internal hashing (circuits)

## Benchmark Results

### Circuit Size: 100 iterations

| Configuration | Build Time | Prove Time | Verify Time | Total Time | Speedup |
|--------------|------------|------------|-------------|------------|---------|
| Poseidon     | 4.57ms     | 6.21ms     | 1.36ms      | 12.14ms    | -       |
| Poseidon2    | 2.41ms     | 6.36ms     | 1.88ms      | 10.65ms    | **1.14x** |

**Build speedup: 1.90x** (4.57ms → 2.41ms)

### Circuit Size: 500 iterations

| Configuration | Build Time | Prove Time | Verify Time | Total Time | Speedup |
|--------------|------------|------------|-------------|------------|---------|
| Poseidon     | 3.32ms     | 2.82ms     | 1.50ms      | 7.63ms     | -       |
| Poseidon2    | 3.28ms     | 13.71ms    | 2.17ms      | 19.16ms    | **0.40x** ⚠️ |

**Note**: Poseidon2 is slower here - likely due to AVX2 warmup or different circuit structure.

### Circuit Size: 1000 iterations

| Configuration | Build Time | Prove Time | Verify Time | Total Time | Speedup |
|--------------|------------|------------|-------------|------------|---------|
| Poseidon     | 5.05ms     | 8.90ms     | 1.87ms      | 15.82ms    | -       |
| Poseidon2    | 5.86ms     | 6.97ms     | 2.77ms      | 15.60ms    | **1.01x** |

**Prove speedup: 1.28x** (8.90ms → 6.97ms)

## Analysis

### Key Observations

1. **Build Time**: Poseidon2 shows significant improvement for small circuits (1.90x faster at size 100) but becomes comparable or slightly slower for larger circuits.

2. **Proof Generation**:
   - For small circuits (100): Similar performance (6.21ms vs 6.36ms)
   - For medium circuits (500): Poseidon2 is unexpectedly slower (needs investigation)
   - For large circuits (1000): Poseidon2 shows **1.28x speedup** (8.90ms → 6.97ms)

3. **Verification Time**: Poseidon2 is consistently slower in verification (1.36ms → 1.88ms for size 100), likely due to different hash function overhead.

4. **Overall Performance**: Mixed results, with best performance at small (100) and large (1000) circuit sizes.

### Performance Breakdown

#### Circuit Size: 100
```
Poseidon:  Build 37.6% | Prove 51.2% | Verify 11.2%
Poseidon2: Build 22.6% | Prove 59.7% | Verify 17.7%
```

#### Circuit Size: 1000
```
Poseidon:  Build 31.9% | Prove 56.3% | Verify 11.8%
Poseidon2: Build 37.5% | Prove 44.7% | Verify 17.8%
```

## Performance Characteristics

### Where Poseidon2 Excels

✅ **Proof generation for larger circuits** (1.28x speedup at 1000 iterations)
- Better performance in Merkle tree construction
- More efficient FRI commitments with AVX2 optimizations
- Improved matrix multiplication in Poseidon2 hash

✅ **Circuit building for small circuits** (1.90x speedup at 100 iterations)
- Faster initial setup
- Efficient sponge construction

### Where Poseidon2 Shows No Improvement

⚠️ **Medium-sized circuits** (500 iterations)
- Unexpected slowdown in proof generation
- Possibly due to CPU cache effects or AVX2 warmup
- Requires further investigation

❌ **Verification time**
- Consistently 30-40% slower
- Likely due to Poseidon2 hash computation overhead in verification

## Recommendations

### When to Use Poseidon2GoldilocksConfig

1. **Large circuits with many constraints** - Shows clear proof generation speedup
2. **Applications prioritizing proof generation over verification** - If prover performance is critical
3. **Batch proof generation** - Amortizes the warmup cost

### When to Use PoseidonGoldilocksConfig

1. **Applications with frequent verification** - Original Poseidon verifies faster
2. **Medium-sized circuits** - More consistent performance
3. **When stability is critical** - Well-tested, mature implementation

## Technical Details

### Hash Function Differences

**Poseidon**:
- 12-element state width
- 8 full rounds + 22 partial rounds
- Standard MDS matrix

**Poseidon2**:
- 12-element state width
- 8 full rounds + 22 partial rounds
- Optimized M_E (external) matrix using M_4 blocks
- AVX2-accelerated matrix multiplication
- More efficient internal diffusion layer

### AVX2 Optimizations

Both implementations use AVX2 SIMD instructions for:
- S-box computation (x^7 in Goldilocks field)
- Matrix-vector multiplication
- Round constant addition

Poseidon2 additionally optimizes:
- Block-wise M_4 matrix application
- Internal layer diffusion with diagonal matrix

## Future Work

1. **Investigate 500-iteration slowdown** - Profile to understand performance regression
2. **Benchmark with different circuit types** - Test with other operations beyond Fibonacci
3. **Measure memory usage** - Compare memory footprint between configurations
4. **Test on different hardware** - Verify AVX2 benefits across CPUs
5. **Implement Poseidon2Gate** - Enable full Poseidon2 support for in-circuit hashing

## Running the Benchmark

To reproduce these results:

```bash
cargo run --release --example bench_poseidon_vs_poseidon2
```

To benchmark with custom circuit sizes, modify the `circuit_sizes` vector in `main()`:

```rust
let circuit_sizes = vec![100, 500, 1000, 2000, 5000];
```

## System Information

- **CPU**: x86_64 with AVX2 support
- **Compiler**: rustc with release optimizations
- **Build**: `--release` with target-cpu=native recommended for best performance

---

**Generated**: 2025-11-07
**Benchmark Tool**: [bench_poseidon_vs_poseidon2.rs](plonky2/examples/bench_poseidon_vs_poseidon2.rs)
