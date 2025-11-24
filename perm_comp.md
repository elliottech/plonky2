# Performance comparison
- CPU: AMD 7950x3d 16 core
- GPU: 4080 super; single card
- 

| Operation | CPU (s) | GPU (s) | Speedup | GPU Tuned? |
|-----------|---------|---------|---------|------------|
| **Run generators** | 1.7767 | 1.7899 | 0.99x | ✗ Not accelerated |
| **Compute full witness** | 0.3369 | 0.3362 | 1.00x | ✗ Not accelerated |
| **Compute wire polynomials** | 0.0396 | 0.0392 | 1.01x | ✗ Not accelerated |
| **Compute wires commitment** | 20.1902 | 10.0548 | **2.01x** | ✓ Yes |
| └─ IFFT | 1.2070 | 0.1587 | **7.61x** | ✓ **Highly tuned** |
| └─ FFT + blinding | 11.4267 | 3.6139 | **3.16x** | ✓ **Highly tuned** |
| └─ Transpose LDEs | 2.8010 | 2.7881 | 1.00x | ✗ Not accelerated |
| └─ Build Merkle tree | 4.5166 | 3.2734 | **1.38x** | ✓ Tuned |
| **Compute partial products** | 0.1700 | 0.1671 | 1.02x | ✗ Not accelerated |
| **Commit to partial products/Z's** | 3.4213 | 1.6982 | **2.01x** | ✓ Yes |
| └─ IFFT | 0.1860 | 0.0241 | **7.72x** | ✓ **Highly tuned** |
| └─ FFT + blinding | 1.7627 | 0.4778 | **3.69x** | ✓ **Highly tuned** |
| └─ Transpose LDEs | 0.3906 | 0.3874 | 1.01x | ✗ Not accelerated |
| └─ Build Merkle tree | 1.0253 | 0.7573 | **1.35x** | ✓ Tuned |
| **Compute quotient polys** | 1.4041 | 1.3128 | 1.07x | ✗ Not accelerated |
| **Split quotient polys** | 0.0098 | 0.0212 | 0.46x | ✗ Not accelerated|
| **Commit to quotient polys** | 2.6641 | 1.4077 | **1.89x** | ✓ Yes |
| └─ FFT + blinding | 1.5496 | 0.4315 | **3.59x** | ✓ **Highly tuned** |
| └─ Transpose LDEs | 0.2952 | 0.2908 | 1.02x | ✗ Not accelerated |
| └─ Build Merkle tree | 0.7756 | 0.6453 | **1.20x** | ✓ Tuned |
| **Construct opening set** | 0.1609 | 0.1600 | 1.01x | ✗ Not accelerated |
| **Compute opening proofs** | 1.3580 | 1.2919 | 1.05x | ✗ Not accelerated |
| └─ Reduce 255 polynomials | 0.8715 | 0.8518 | 1.02x | ✗ Not accelerated |
| └─ Reduce 2 polynomials | 0.0087 | 0.0085 | 1.02x | ✗ Not accelerated |
| └─ Final FFT 4194304 | 0.3083 | 0.3023 | 1.02x | ✗ Not accelerated |
| └─ Fold codewords | 0.1312 | 0.0904 | **1.45x** | ✗ Not accelerated |
| └─ Find PoW witness | 0.0014 | 0.0038 | 0.37x | ✗ Not accelerated |