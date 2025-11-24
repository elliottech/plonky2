use alloc::vec::Vec;
use core::cmp::{max, min};

use plonky2_util::{log2_strict, reverse_index_bits_in_place};
use unroll::unroll_for_loops;

use crate::packable::Packable;
use crate::packed::PackedField;
use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::types::Field;

pub type FftRootTable<F> = Vec<Vec<F>>;

pub fn fft_root_table<F: Field>(n: usize) -> FftRootTable<F> {
    let lg_n = log2_strict(n);
    // bases[i] = g^2^i, for i = 0, ..., lg_n - 1
    let mut bases = Vec::with_capacity(lg_n);
    let mut base = F::primitive_root_of_unity(lg_n);
    bases.push(base);
    for _ in 1..lg_n {
        base = base.square(); // base = g^2^_
        bases.push(base);
    }

    let mut root_table = Vec::with_capacity(lg_n);
    for lg_m in 1..=lg_n {
        let half_m = 1 << (lg_m - 1);
        let base = bases[lg_n - lg_m];
        let root_row = base.powers().take(half_m.max(2)).collect();
        root_table.push(root_row);
    }
    root_table
}

#[cfg(feature = "cuda")]
fn fft_dispatch_gpu<F: Field>(
    input: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    // if F::CUDA_SUPPORT {
    //     use zeknox::ntt_batch;
    //     use zeknox::types::NTTConfig;

    //     let mut a = input.to_vec();
    //     let mut b = input.to_vec();

    //     ntt_batch(
    //         0,
    //         a.as_mut_ptr(),
    //         input.len().trailing_zeros() as usize,
    //         NTTConfig::default(),
    //     );

    //     fft_dispatch_cpu(&mut b, zero_factor, root_table);
    //     ark_std::println!("a: {:?}", a);
    //     ark_std::println!("b: {:?}", b);

    //     assert_eq!(
    //         a, b,
    //         "failed GPU FFT vs CPU FFT comparison\ngpu:{:?}\ncpu:{:?}\ninput:{:?}",
    //         a, b, input
    //     );

    //     input.copy_from_slice(&a);
    // }
    // return fft_dispatch_cpu(input, zero_factor, root_table);

    use zeknox::ntt_batch;
    use zeknox::types::NTTConfig;
    if F::CUDA_SUPPORT {
        return ntt_batch(
            0,
            input.as_mut_ptr(),
            input.len().trailing_zeros() as usize,
            NTTConfig::default(),
        );
    } else {
        return fft_dispatch_cpu(input, zero_factor, root_table);
    }
}

/// Batch FFT computation for multiple polynomials on GPU
#[cfg(feature = "cuda")]
fn fft_batch_dispatch_gpu<F: Field>(
    inputs: &mut [F],
    poly_size: usize,
    num_polys: usize,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    use zeknox::ntt_batch;
    use zeknox::types::NTTConfig;

    if F::CUDA_SUPPORT {
        let mut cfg = NTTConfig::default();
        cfg.batches = num_polys as u32;

        return ntt_batch(
            0,
            inputs.as_mut_ptr(),
            poly_size.trailing_zeros() as usize,
            cfg,
        );
    } else {
        // Fallback to CPU: process each polynomial separately
        for i in 0..num_polys {
            let start = i * poly_size;
            let end = start + poly_size;
            fft_dispatch_cpu(&mut inputs[start..end], zero_factor, root_table);
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn coset_fft_gpu<F: Field>(
    poly: PolynomialCoeffs<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialValues<F> {
    use zeknox::ntt_batch;
    use zeknox::types::NTTConfig;

    if !F::CUDA_SUPPORT {
        // Fallback to CPU if CUDA not supported for this field
        let modified_poly: PolynomialCoeffs<F> = F::coset_shift()
            .powers()
            .zip(&poly.coeffs)
            .map(|(r, &c)| r * c)
            .collect::<Vec<_>>()
            .into();
        return fft_with_options(modified_poly, zero_factor, root_table);
    }

    let PolynomialCoeffs { coeffs: mut buffer } = poly;
    let lg_n = buffer.len().trailing_zeros() as usize;

    // // Initialize coset on GPU
    // // For Goldilocks field, the coset generator is 7 (MULTIPLICATIVE_GROUP_GENERATOR)
    // // TODO: Make this generic for other fields if needed
    // let coset_gen_u64 = 7u64;
    // init_coset_rs(0, lg_n, coset_gen_u64);

    // Configure NTT for coset
    let mut cfg = NTTConfig::default();
    cfg.with_coset = true;
    cfg.ntt_type = zeknox::types::NTTType::Coset;

    // Perform coset NTT on GPU
    ntt_batch(0, buffer.as_mut_ptr(), lg_n, cfg);

    PolynomialValues::new(buffer)
}

/// Batch coset FFT computation for multiple polynomials on GPU
#[cfg(feature = "cuda")]
fn coset_fft_batch_gpu<F: Field>(
    polys: Vec<PolynomialCoeffs<F>>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> Vec<PolynomialValues<F>> {
    use zeknox::ntt_batch;
    use zeknox::types::NTTConfig;

    if polys.is_empty() {
        return Vec::new();
    }

    let num_polys = polys.len();
    let poly_size = polys[0].len();

    // Verify all polynomials have the same size
    assert!(
        polys.iter().all(|p| p.len() == poly_size),
        "All polynomials must have the same size for batch coset FFT"
    );

    if !F::CUDA_SUPPORT {
        // Fallback to CPU if CUDA not supported for this field
        return polys
            .into_iter()
            .map(|poly| {
                let modified_poly: PolynomialCoeffs<F> = F::coset_shift()
                    .powers()
                    .zip(&poly.coeffs)
                    .map(|(r, &c)| r * c)
                    .collect::<Vec<_>>()
                    .into();
                fft_with_options(modified_poly, zero_factor, root_table)
            })
            .collect();
    }

    // Flatten all polynomials into a single contiguous buffer
    let mut buffer: Vec<F> = Vec::with_capacity(num_polys * poly_size);
    for poly in polys {
        buffer.extend_from_slice(&poly.coeffs);
    }

    let lg_n = poly_size.trailing_zeros() as usize;

    // Configure NTT for batch coset
    let mut cfg = NTTConfig::default();
    cfg.batches = num_polys as u32;
    cfg.with_coset = true;
    cfg.ntt_type = zeknox::types::NTTType::Coset;

    // Perform batch coset NTT on GPU
    ntt_batch(0, buffer.as_mut_ptr(), lg_n, cfg);

    // Split the buffer back into separate polynomials
    buffer
        .chunks(poly_size)
        .map(|chunk| PolynomialValues::new(chunk.to_vec()))
        .collect()
}

/// Compute coset FFT for multiple polynomials in batch.
/// All polynomials must have the same size (power of 2).
/// Returns a vector of PolynomialValues in the same order as input.
pub fn coset_fft_batch<F: Field>(polys: Vec<PolynomialCoeffs<F>>) -> Vec<PolynomialValues<F>> {
    coset_fft_batch_with_options(polys, None, None)
}

/// Compute coset FFT for multiple polynomials in batch with options.
/// All polynomials must have the same size (power of 2).
/// Returns a vector of PolynomialValues in the same order as input.
pub fn coset_fft_batch_with_options<F: Field>(
    polys: Vec<PolynomialCoeffs<F>>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> Vec<PolynomialValues<F>> {
    // #[cfg(feature = "cuda")]
    // {
    //     let a = coset_fft_batch_gpu(polys.clone(), zero_factor, root_table);
    //     let b = polys
    //         .into_iter()
    //         .map(|poly| {
    //             let modified_poly: PolynomialCoeffs<F> = F::coset_shift()
    //                 .powers()
    //                 .zip(&poly.coeffs)
    //                 .map(|(r, &c)| r * c)
    //                 .collect::<Vec<_>>()
    //                 .into();
    //             fft_with_options(modified_poly, zero_factor, root_table)
    //         })
    //         .collect::<Vec<_>>();
    //     assert_eq!(a.len(), b.len());

    //     for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
    //         assert_eq!(val_a, val_b, "Mismatch at index {}", i);
    //     }

    //     return a;
    // }

    // #[cfg(not(feature = "cuda"))]
    // {
    // CPU fallback: process each polynomial separately
    polys
        .into_iter()
        .map(|poly| {
            let modified_poly: PolynomialCoeffs<F> = F::coset_shift()
                .powers()
                .zip(&poly.coeffs)
                .map(|(r, &c)| r * c)
                .collect::<Vec<_>>()
                .into();
            fft_with_options(modified_poly, zero_factor, root_table)
        })
        .collect()
    // }
}

pub(crate) fn fft_dispatch_cpu<F: Field>(
    input: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    if root_table.is_some() {
        return fft_classic(input, zero_factor.unwrap_or(0), root_table.unwrap());
    } else {
        // let pre_computed = F::pre_compute_fft_root_table(input.len());
        // if pre_computed.is_some() {
        //     return fft_classic(input, zero_factor.unwrap_or(0), pre_computed.unwrap());
        // } else {
        //     let computed = fft_root_table::<F>(input.len());

        //     return fft_classic(input, zero_factor.unwrap_or(0), computed.as_ref());
        // }
        let computed = fft_root_table::<F>(input.len());

        return fft_classic(input, zero_factor.unwrap_or(0), computed.as_ref());
    };
}

#[inline]
fn fft_dispatch<F: Field>(
    input: &mut [F],
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) {
    #[cfg(feature = "cuda")]
    {
        // ark_std::println!("Using GPU FFT dispatch");
        return fft_dispatch_gpu(input, zero_factor, root_table);
    }
    #[cfg(not(feature = "cuda"))]
    {
        // ark_std::println!("Using CPU FFT dispatch");
        return fft_dispatch_cpu(input, zero_factor, root_table);
    }
}

#[inline]
pub fn fft<F: Field>(poly: PolynomialCoeffs<F>) -> PolynomialValues<F> {
    fft_with_options(poly, None, None)
}

#[inline]
pub fn fft_with_options<F: Field>(
    poly: PolynomialCoeffs<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialValues<F> {
    let PolynomialCoeffs { coeffs: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table);
    PolynomialValues::new(buffer)
}

/// Compute FFT for multiple polynomials in batch.
/// All polynomials must have the same size (power of 2).
/// Returns a vector of PolynomialValues in the same order as input.
#[inline]
pub fn fft_batch<F: Field>(polys: Vec<PolynomialCoeffs<F>>) -> Vec<PolynomialValues<F>> {
    fft_batch_with_options(polys, None, None)
}

/// Compute FFT for multiple polynomials in batch with options.
/// All polynomials must have the same size (power of 2).
/// Returns a vector of PolynomialValues in the same order as input.
pub fn fft_batch_with_options<F: Field>(
    polys: Vec<PolynomialCoeffs<F>>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> Vec<PolynomialValues<F>> {
    if polys.is_empty() {
        return Vec::new();
    }

    let num_polys = polys.len();
    let poly_size = polys[0].len();

    // Verify all polynomials have the same size
    assert!(
        polys.iter().all(|p| p.len() == poly_size),
        "All polynomials must have the same size for batch FFT"
    );
    assert!(
        poly_size.is_power_of_two(),
        "Polynomial size must be a power of 2"
    );

    // Flatten all polynomials into a single contiguous buffer
    let mut buffer: Vec<F> = Vec::with_capacity(num_polys * poly_size);
    for poly in polys {
        buffer.extend_from_slice(&poly.coeffs);
    }

    // Dispatch to GPU or CPU batch processing
    #[cfg(feature = "cuda")]
    fft_batch_dispatch_gpu(&mut buffer, poly_size, num_polys, zero_factor, root_table);

    #[cfg(not(feature = "cuda"))]
    {
        // CPU fallback: process each polynomial separately
        for i in 0..num_polys {
            let start = i * poly_size;
            let end = start + poly_size;
            fft_dispatch_cpu(&mut buffer[start..end], zero_factor, root_table);
        }
    }

    // Split the buffer back into separate polynomials
    buffer
        .chunks(poly_size)
        .map(|chunk| PolynomialValues::new(chunk.to_vec()))
        .collect()
}

#[inline]
pub fn ifft<F: Field>(poly: PolynomialValues<F>) -> PolynomialCoeffs<F> {
    ifft_with_options(poly, None, None)
}

pub fn ifft_with_options<F: Field>(
    poly: PolynomialValues<F>,
    zero_factor: Option<usize>,
    root_table: Option<&FftRootTable<F>>,
) -> PolynomialCoeffs<F> {
    let n = poly.len();
    let lg_n = log2_strict(n);
    let n_inv = F::inverse_2exp(lg_n);

    let PolynomialValues { values: mut buffer } = poly;
    fft_dispatch(&mut buffer, zero_factor, root_table);

    // We reverse all values except the first, and divide each by n.
    buffer[0] *= n_inv;
    buffer[n / 2] *= n_inv;
    for i in 1..(n / 2) {
        let j = n - i;
        let coeffs_i = buffer[j] * n_inv;
        let coeffs_j = buffer[i] * n_inv;
        buffer[i] = coeffs_i;
        buffer[j] = coeffs_j;
    }
    PolynomialCoeffs { coeffs: buffer }
}

/// Generic FFT implementation that works with both scalar and packed inputs.
#[unroll_for_loops]
fn fft_classic_simd<P: PackedField>(
    values: &mut [P::Scalar],
    r: usize,
    lg_n: usize,
    root_table: &FftRootTable<P::Scalar>,
) {
    let lg_packed_width = log2_strict(P::WIDTH); // 0 when P is a scalar.
    let packed_values = P::pack_slice_mut(values);
    let packed_n = packed_values.len();
    debug_assert!(packed_n == 1 << (lg_n - lg_packed_width));

    // Want the below for loop to unroll, hence the need for a literal.
    // This loop will not run when P is a scalar.
    assert!(lg_packed_width <= 4);
    for lg_half_m in 0..4 {
        if (r..min(lg_n, lg_packed_width)).contains(&lg_half_m) {
            // Intuitively, we split values into m slices: subarr[0], ..., subarr[m - 1]. Each of
            // those slices is split into two halves: subarr[j].left, subarr[j].right. We do
            // (subarr[j].left[k], subarr[j].right[k])
            //   := f(subarr[j].left[k], subarr[j].right[k], omega[k]),
            // where f(u, v, omega) = (u + omega * v, u - omega * v).
            let half_m = 1 << lg_half_m;

            // Set omega to root_table[lg_half_m][0..half_m] but repeated.
            let mut omega = P::default();
            for (j, omega_j) in omega.as_slice_mut().iter_mut().enumerate() {
                *omega_j = root_table[lg_half_m][j % half_m];
            }

            for k in (0..packed_n).step_by(2) {
                // We have two vectors and want to do math on pairs of adjacent elements (or for
                // lg_half_m > 0, pairs of adjacent blocks of elements). .interleave does the
                // appropriate shuffling and is its own inverse.
                let (u, v) = packed_values[k].interleave(packed_values[k + 1], half_m);
                let t = omega * v;
                (packed_values[k], packed_values[k + 1]) = (u + t).interleave(u - t, half_m);
            }
        }
    }

    // We've already done the first lg_packed_width (if they were required) iterations.
    let s = max(r, lg_packed_width);

    for lg_half_m in s..lg_n {
        let lg_m = lg_half_m + 1;
        let m = 1 << lg_m; // Subarray size (in field elements).
        let packed_m = m >> lg_packed_width; // Subarray size (in vectors).
        let half_packed_m = packed_m / 2;
        debug_assert!(half_packed_m != 0);

        // omega values for this iteration, as slice of vectors
        let omega_table = P::pack_slice(&root_table[lg_half_m][..]);
        for k in (0..packed_n).step_by(packed_m) {
            for j in 0..half_packed_m {
                let omega = omega_table[j];
                let t = omega * packed_values[k + half_packed_m + j];
                let u = packed_values[k + j];
                packed_values[k + j] = u + t;
                packed_values[k + half_packed_m + j] = u - t;
            }
        }
    }
}

/// FFT implementation based on Section 32.3 of "Introduction to
/// Algorithms" by Cormen et al.
///
/// The parameter r signifies that the first 1/2^r of the entries of
/// input may be non-zero, but the last 1 - 1/2^r entries are
/// definitely zero.
pub(crate) fn fft_classic<F: Field>(values: &mut [F], r: usize, root_table: &FftRootTable<F>) {
    reverse_index_bits_in_place(values);

    let n = values.len();
    let lg_n = log2_strict(n);

    if root_table.len() != lg_n {
        panic!(
            "Expected root table of length {}, but it was {}.",
            lg_n,
            root_table.len()
        );
    }

    // After reverse_index_bits, the only non-zero elements of values
    // are at indices i*2^r for i = 0..n/2^r.  The loop below copies
    // the value at i*2^r to the positions [i*2^r + 1, i*2^r + 2, ...,
    // (i+1)*2^r - 1]; i.e. it replaces the 2^r - 1 zeros following
    // element i*2^r with the value at i*2^r.  This corresponds to the
    // first r rounds of the FFT when there are 2^r zeros at the end
    // of the original input.
    if r > 0 {
        // if r == 0 then this loop is a noop.
        let mask = !((1 << r) - 1);
        for i in 0..n {
            values[i] = values[i & mask];
        }
    }

    let lg_packed_width = log2_strict(<F as Packable>::Packing::WIDTH);
    if lg_n <= lg_packed_width {
        // Need the slice to be at least the width of two packed vectors for the vectorized version
        // to work. Do this tiny problem in scalar.
        fft_classic_simd::<F>(values, r, lg_n, root_table);
    } else {
        fft_classic_simd::<<F as Packable>::Packing>(values, r, lg_n, root_table);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use plonky2_util::{log2_ceil, log2_strict};
    #[cfg(feature = "cuda")]
    use zeknox::init_twiddle_factors_rs;

    #[cfg(feature = "cuda")]
    use crate::fft::{coset_fft_batch, fft_dispatch_cpu, fft_dispatch_gpu};
    use crate::fft::{fft, fft_batch, fft_with_options, ifft};
    use crate::goldilocks_field::GoldilocksField;
    use crate::polynomial::{PolynomialCoeffs, PolynomialValues};
    use crate::types::Field;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_kat() {
        init_twiddle_factors_rs(0, 4);

        let input = [
            16807u64,
            10376289027450995739,
            18446743787439915009,
            1905022641934172156,
            4730749933575995392,
            68841472,
            18428264577490855681,
            18445589101169082369,
            18446744069414567514,
            8070455041963588582,
            49,
            1625527855624486912,
            7,
            18446744069414555649,
            7696581392640,
            481036337152,
        ];
        let input_field: Vec<GoldilocksField> = input
            .iter()
            .map(|&x| GoldilocksField::from_canonical_u64(x))
            .collect();

        let res_cpu = [
            8241673866677297204,
            18443207692673526440,
            3336172192632445894,
            12915814655533318448,
            5977358399840934215,
            2796120128477098295,
            16099264885043452953,
            1114428869533774434,
            1182881845840683068,
            18442399148451944616,
            5639697009785877037,
            5534977815694745617,
            3521085621945067109,
            15650623939293352472,
            11342098386477995483,
            17336148097415430195,
        ];
        let res_cpu_field: Vec<GoldilocksField> = res_cpu
            .iter()
            .map(|&x| GoldilocksField::from_canonical_u64(x))
            .collect();

        let res_gpu = [
            8241673866677297204,
            18443207692673526440,
            3336172192632445894,
            12915814655533318448,
            5977358399840934215,
            2796120128477098295,
            16099264885043452953,
            1114428869533774434,
            1182881845840683068,
            18442399148451944616,
            5639697009785877037,
            5534977815694745617,
            3521085621945067109,
            15650623939293352472,
            11342098386477995483,
            17336148097415430195,
        ];
        let res_gpu_field: Vec<GoldilocksField> = res_gpu
            .iter()
            .map(|&x| GoldilocksField::from_canonical_u64(x))
            .collect();

        let mut input_cpu = input_field.clone();
        fft_dispatch_cpu(&mut input_cpu, None, None);
        assert_eq!(input_cpu, res_cpu_field);

        let mut input_gpu = input_field.clone();
        fft_dispatch_gpu(&mut input_gpu, None, None);
        assert_eq!(input_gpu, res_gpu_field);
    }

    #[test]
    fn fft_and_ifft() {
        type F = GoldilocksField;
        let degree = 200usize;
        let degree_padded = degree.next_power_of_two();

        #[cfg(feature = "cuda")]
        let log_degree = {
            zeknox::clear_cuda_errors_rs();
            let log_degree = degree_padded.trailing_zeros() as usize;
            init_twiddle_factors_rs(0, log_degree);
            log_degree
        };
        // Create a vector of coeffs; the first degree of them are
        // "random", the last degree_padded-degree of them are zero.
        let coeffs = (0..degree)
            .map(|i| F::from_canonical_usize(i * 1337 % 100))
            .chain(core::iter::repeat_n(F::ZERO, degree_padded - degree))
            .collect::<Vec<_>>();
        assert_eq!(coeffs.len(), degree_padded);
        let coefficients = PolynomialCoeffs { coeffs };

        let points = fft(coefficients.clone());
        assert_eq!(points, evaluate_naive(&coefficients));

        let interpolated_coefficients = ifft(points);
        for i in 0..degree {
            assert_eq!(interpolated_coefficients.coeffs[i], coefficients.coeffs[i]);
        }
        for i in degree..degree_padded {
            assert_eq!(interpolated_coefficients.coeffs[i], F::ZERO);
        }

        for r in 0..4 {
            #[cfg(feature = "cuda")]
            init_twiddle_factors_rs(0, log_degree + r);
            // expand coefficients by factor 2^r by filling with zeros
            let zero_tail = coefficients.lde(r);
            assert_eq!(
                fft(zero_tail.clone()),
                fft_with_options(zero_tail, Some(r), None)
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_fft_gpu_vs_cpu_single() {
        type F = GoldilocksField;

        // Test various polynomial sizes
        for log_size in [8, 10, 12, 14,16,18,20] {
            let size = 1 << log_size;
            zeknox::clear_cuda_errors_rs();
            init_twiddle_factors_rs(0, log_size);

            // Create a random polynomial
            let coeffs: Vec<F> = (0..size)
                .map(|i| F::from_canonical_usize(i * 7919 % 1000000))
                .collect();

            let poly = PolynomialCoeffs {
                coeffs: coeffs.clone(),
            };

            // Compute FFT using GPU (via fft function which dispatches to GPU)
            let gpu_result = fft(poly.clone());

            // Compute FFT using CPU (force CPU path)
            let mut cpu_buffer = coeffs.clone();
            super::fft_dispatch_cpu(&mut cpu_buffer, None, None);
            let cpu_result = PolynomialValues::new(cpu_buffer);

            // Compare results
            assert_eq!(
                gpu_result.len(),
                cpu_result.len(),
                "GPU and CPU results have different lengths for size {}",
                size
            );

            for i in 0..size {
                assert_eq!(
                    gpu_result.values[i], cpu_result.values[i],
                    "Mismatch at index {} for polynomial size {}",
                    i, size
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_fft_batch_gpu_vs_cpu() {
        type F = GoldilocksField;

        let poly_size: usize = 1 << 10; // 1024 elements
        let num_polys = 8;
        let log_size = poly_size.trailing_zeros() as usize;

        zeknox::clear_cuda_errors_rs();
        init_twiddle_factors_rs(0, log_size);

        // Create multiple random polynomials
        let polys: Vec<PolynomialCoeffs<F>> = (0..num_polys)
            .map(|batch_idx| {
                let coeffs: Vec<F> = (0..poly_size)
                    .map(|i| F::from_canonical_usize((i * 7919 + batch_idx * 12345) % 1000000))
                    .collect();
                PolynomialCoeffs { coeffs }
            })
            .collect();

        // Compute batch FFT using GPU
        let gpu_results = fft_batch(polys.clone());

        // Compute FFT for each polynomial using CPU
        let cpu_results: Vec<PolynomialValues<F>> = polys
            .into_iter()
            .map(|poly| {
                let mut buffer = poly.coeffs.clone();
                super::fft_dispatch_cpu(&mut buffer, None, None);
                PolynomialValues::new(buffer)
            })
            .collect();

        // Compare results
        assert_eq!(gpu_results.len(), cpu_results.len());
        for (batch_idx, (gpu_result, cpu_result)) in
            gpu_results.iter().zip(cpu_results.iter()).enumerate()
        {
            assert_eq!(gpu_result.len(), cpu_result.len());
            for i in 0..poly_size {
                assert_eq!(
                    gpu_result.values[i], cpu_result.values[i],
                    "Batch FFT mismatch at batch {} index {}",
                    batch_idx, i
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_coset_fft_gpu_vs_cpu_single() {
        use zeknox::init_coset_rs;

        use crate::types::PrimeField64;
        type F = GoldilocksField;

        for log_size in [8, 10, 12] {
            let size = 1 << log_size;
            zeknox::clear_cuda_errors_rs();
            init_twiddle_factors_rs(0, log_size);

            // Initialize coset for GPU
            let coset_gen_u64 = F::coset_shift().to_canonical_u64();
            init_coset_rs(0, log_size, coset_gen_u64);

            // Create a random polynomial
            let coeffs: Vec<F> = (0..size)
                .map(|i| F::from_canonical_usize(i * 8191 % 1000000))
                .collect();

            let poly = PolynomialCoeffs {
                coeffs: coeffs.clone(),
            };

            // Compute coset FFT using GPU
            let gpu_result = super::coset_fft_gpu(poly.clone(), None, None);

            // Compute coset FFT using CPU (apply coset shift then FFT)
            let modified_poly: PolynomialCoeffs<F> = F::coset_shift()
                .powers()
                .zip(&coeffs)
                .map(|(r, &c)| r * c)
                .collect::<Vec<_>>()
                .into();

            let mut cpu_buffer = modified_poly.coeffs;
            super::fft_dispatch_cpu(&mut cpu_buffer, None, None);
            let cpu_result = PolynomialValues::new(cpu_buffer);

            // Compare results
            assert_eq!(
                gpu_result.len(),
                cpu_result.len(),
                "GPU and CPU coset FFT results have different lengths for size {}",
                size
            );

            for i in 0..size {
                assert_eq!(
                    gpu_result.values[i], cpu_result.values[i],
                    "Coset FFT mismatch at index {} for polynomial size {}",
                    i, size
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_coset_fft_batch_gpu_vs_cpu() {
        use zeknox::init_coset_rs;

        use crate::types::PrimeField64;
        type F = GoldilocksField;

        let poly_size: usize = 1 << 10; // 1024 elements
        let num_polys = 8;
        let log_size = poly_size.trailing_zeros() as usize;

        zeknox::clear_cuda_errors_rs();
        init_twiddle_factors_rs(0, log_size);

        // Initialize coset for GPU
        let coset_gen_u64 = F::coset_shift().to_canonical_u64();
        init_coset_rs(0, log_size, coset_gen_u64);

        // Create multiple random polynomials
        let polys: Vec<PolynomialCoeffs<F>> = (0..num_polys)
            .map(|batch_idx| {
                let coeffs: Vec<F> = (0..poly_size)
                    .map(|i| F::from_canonical_usize((i * 8191 + batch_idx * 54321) % 1000000))
                    .collect();
                PolynomialCoeffs { coeffs }
            })
            .collect();

        // Compute batch coset FFT using GPU
        let gpu_results = coset_fft_batch(polys.clone());

        // Compute coset FFT for each polynomial using CPU
        let cpu_results: Vec<PolynomialValues<F>> = polys
            .into_iter()
            .map(|poly| {
                let modified_poly: PolynomialCoeffs<F> = F::coset_shift()
                    .powers()
                    .zip(&poly.coeffs)
                    .map(|(r, &c)| r * c)
                    .collect::<Vec<_>>()
                    .into();

                let mut buffer = modified_poly.coeffs;
                super::fft_dispatch_cpu(&mut buffer, None, None);
                PolynomialValues::new(buffer)
            })
            .collect();

        // Compare results
        assert_eq!(gpu_results.len(), cpu_results.len());
        for (batch_idx, (gpu_result, cpu_result)) in
            gpu_results.iter().zip(cpu_results.iter()).enumerate()
        {
            assert_eq!(gpu_result.len(), cpu_result.len());
            for i in 0..poly_size {
                assert_eq!(
                    gpu_result.values[i], cpu_result.values[i],
                    "Batch coset FFT mismatch at batch {} index {}",
                    batch_idx, i
                );
            }
        }
    }

    #[test]
    fn test_batch_fft_empty() {
        type F = GoldilocksField;
        let polys: Vec<PolynomialCoeffs<F>> = vec![];
        let results = fft_batch(polys);
        assert!(results.is_empty());
    }

    #[test]
    #[should_panic(expected = "All polynomials must have the same size")]
    fn test_batch_fft_different_sizes() {
        type F = GoldilocksField;
        let poly1 = PolynomialCoeffs {
            coeffs: vec![F::ONE; 256],
        };
        let poly2 = PolynomialCoeffs {
            coeffs: vec![F::ONE; 512],
        };
        let _ = fft_batch(vec![poly1, poly2]);
    }

    fn evaluate_naive<F: Field>(coefficients: &PolynomialCoeffs<F>) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_padded = 1 << log2_ceil(degree);

        let coefficients_padded = coefficients.padded(degree_padded);
        evaluate_naive_power_of_2(&coefficients_padded)
    }

    fn evaluate_naive_power_of_2<F: Field>(
        coefficients: &PolynomialCoeffs<F>,
    ) -> PolynomialValues<F> {
        let degree = coefficients.len();
        let degree_log = log2_strict(degree);

        let subgroup = F::two_adic_subgroup(degree_log);

        let values = subgroup
            .into_iter()
            .map(|x| evaluate_at_naive(coefficients, x))
            .collect();
        PolynomialValues::new(values)
    }

    fn evaluate_at_naive<F: Field>(coefficients: &PolynomialCoeffs<F>, point: F) -> F {
        let mut sum = F::ZERO;
        let mut point_power = F::ONE;
        for &c in &coefficients.coeffs {
            sum += c * point_power;
            point_power *= point;
        }
        sum
    }
}
