#[cfg(not(feature = "std"))]
use alloc::{format, vec::Vec};

use itertools::Itertools;
use plonky2_field::types::Field;
use plonky2_maybe_rayon::*;

use crate::field::extension::Extendable;
use crate::field::fft::FftRootTable;
use crate::field::packed::PackedField;
use crate::field::polynomial::{PolynomialCoeffs, PolynomialValues};
use crate::fri::proof::FriProof;
use crate::fri::prover::fri_proof;
use crate::fri::structure::{FriBatchInfo, FriInstanceInfo};
use crate::fri::FriParams;
use crate::hash::hash_types::RichField;
use crate::hash::merkle_tree::MerkleTree;
use crate::iop::challenger::Challenger;
use crate::plonk::config::GenericConfig;
use crate::timed;
use crate::util::reducing::ReducingFactor;
use crate::util::timing::TimingTree;
use crate::util::{log2_strict, reverse_bits, reverse_index_bits_in_place, transpose};

/// Four (~64 bit) field elements gives ~128 bit security.
pub const SALT_SIZE: usize = 4;

/// Represents a FRI oracle, i.e. a batch of polynomials which have been Merklized.
#[derive(Eq, PartialEq, Debug)]
pub struct PolynomialBatch<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
{
    pub polynomials: Vec<PolynomialCoeffs<F>>,
    pub merkle_tree: MerkleTree<F, C::Hasher>,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize> Default
    for PolynomialBatch<F, C, D>
{
    fn default() -> Self {
        PolynomialBatch {
            polynomials: Vec::new(),
            merkle_tree: MerkleTree::default(),
            degree_log: 0,
            rate_bits: 0,
            blinding: false,
        }
    }
}

impl<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>
    PolynomialBatch<F, C, D>
{
    // pub fn from_values_gpu(
    //     values: Vec<PolynomialValues<F>>,
    //     rate_bits: usize,
    //     blinding: bool,
    //     cap_height: usize,
    //     timing: &mut TimingTree,
    //     fft_root_table: Option<&FftRootTable<F>>,
    // ) -> Self {
    //     let coeffs = timed!(
    //         timing,
    //         "CPU IFFT",
    //         values
    //             .into_par_iter()
    //             .map(|v| v.ifft_cpu())
    //             .collect::<Vec<_>>()
    //     );
    // }

    /// Creates a list polynomial commitment for the polynomials interpolating the values in `values`.
    /// This function is called by the builder during preprocessing the circuit.
    /// This function always calls IFFT on CPU to avoid strange GPU issue.
    pub fn from_values(
        values: Vec<PolynomialValues<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let coeffs = timed!(
            timing,
            "CPU IFFT",
            values
                .into_par_iter()
                .map(|v| v.ifft_cpu())
                .collect::<Vec<_>>()
        );

        Self::from_coeffs_gpu(
            coeffs,
            rate_bits,
            blinding,
            cap_height,
            timing,
            fft_root_table,
        )
    }

    /// Creates a list polynomial commitment for the polynomials `polynomials`.
    pub fn from_coeffs(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();
        let lde_values = timed!(
            timing,
            "FFT + blinding",
            Self::lde_values(&polynomials, rate_bits, blinding, fft_root_table)
        );

        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new_from_2d(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    pub fn from_coeffs_gpu(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Self {
        let degree = polynomials[0].len();

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };
        println!(
            "lde_values: num_polys={}, degree={}, blinding={}, salt_size={}",
            polynomials.len(),
            degree,
            blinding,
            salt_size
        );

        #[cfg(feature = "cuda")]
        {
            if F::CUDA_SUPPORT {
                return Self::from_coeffs_gpu_optimized(
                    polynomials,
                    rate_bits,
                    blinding,
                    cap_height,
                    timing,
                    fft_root_table,
                    degree,
                    salt_size,
                );
            }
        }

        // Fallback to CPU path
        let lde_values = polynomials
            .iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect::<Vec<_>>();
        let mut leaves = timed!(timing, "transpose LDEs", transpose(&lde_values));
        reverse_index_bits_in_place(&mut leaves);
        let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new_from_2d(leaves, cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    #[cfg(feature = "cuda")]
    fn from_coeffs_gpu_optimized(
        polynomials: Vec<PolynomialCoeffs<F>>,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        timing: &mut TimingTree,
        fft_root_table: Option<&FftRootTable<F>>,
        degree: usize,
        salt_size: usize,
    ) -> Self {
        println!("Using GPU-accelerated LDE computation");

        use std::time::Instant;

        use zeknox::device::memory::HostOrDeviceSlice;
        use zeknox::types::{NTTConfig, TransposeConfig};
        use zeknox::{ntt_batch_ptr, transpose_rev_batch};

        let lde_size = degree << rate_bits;
        let num_polys = polynomials.len() + salt_size;

        // let lde_cpu = {
        //     // Fallback to CPU path
        //     let lde_values = polynomials
        //         .iter()
        //         .map(|p| {
        //             assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
        //             p.lde(rate_bits)
        //                 .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
        //                 .values
        //         })
        //         .collect::<Vec<_>>();

        //     // for v in &lde_values {
        //     //     println!("lde_values {:?}", v);
        //     // }

        //     lde_values
        // };

        // let salt_size = if blinding { SALT_SIZE } else { 0 };

        // Step 1: Compute coset FFT on GPU, keeping data on GPU
        let gpu_lde_values = timed!(timing, "GPU coset FFT", {
            // let mut all_lde_data = Vec::with_capacity(num_polys);

            // // Process each polynomial
            // for p in polynomials.iter() {
            //     assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");

            //     // Perform LDE (padding) and coset scaling on CPU
            //     let mut padded_coeffs = p.lde(rate_bits).coeffs;

            //     // Apply coset shift
            //     let shift = F::coset_shift();
            //     for (i, coeff) in padded_coeffs.iter_mut().enumerate() {
            //         *coeff *= shift.exp_u64(i as u64);
            //     }

            //     all_lde_data.push(padded_coeffs);
            // }

            // // Add salt
            // for _ in 0..salt_size {
            //     all_lde_data.push(F::rand_vec(lde_size));
            // }

            // Allocate GPU memory for all polynomials
            let total_elements = num_polys * lde_size;
            println!(
                "Allocating GPU memory for {} polynomials of size {} (total {} elements)",
                num_polys, lde_size, total_elements
            );
            let total_alloce_size = num_polys * lde_size;
            // let total_alloce_size = num_polys.next_power_of_two() * lde_size;

            let timer = Instant::now();
            let mut gpu_buffer = HostOrDeviceSlice::cuda_malloc(0, total_alloce_size)
                .expect("Failed to allocate GPU memory");
            println!("cuda alloc took: {:?}", timer.elapsed());
            println!(
                "lde size: {}, total_elements: {}, total allocated size: {}",
                lde_size, total_elements, total_alloce_size
            );

            let timer = Instant::now();
            // Copy all data to GPU in one go
            let mut flat_data = vec![F::ZERO; total_alloce_size];

            for i in 0..polynomials.len() {

                flat_data[i*lde_size.. i*lde_size +degree].copy_from_slice(polynomials[i].coeffs.as_ref())

            }

            // polynomials.par_iter().zip(flat_data.par_chunks_exact_mut(lde_size)).for_each(|(p, c)|
            //     c[..degree].copy_from_slice(p.coeffs.as_ref())
            // );

            // let flat_data: Vec<F> = polynomials
            //     .iter()
            //     .flat_map(|v| v.lde(rate_bits).coeffs) // pad each polynomial to lde_size
            //     // .into_iter()
            //     // .collect()
            //     // .iter()
            //     // .chain(
            //     //     vec![F::ZERO; total_alloce_size - total_elements]
            //     //         .iter()
            //     //         .copied(),
            //     // )
            //     .collect();

            println!("cpu prepare took: {:?}", timer.elapsed());
            // let flat_data =  vec![F::ZERO; lde_size - total_elements].iter()

            println!(
                "Copying {} elements to GPU (expected {})",
                flat_data.len(),
                total_elements
            );

            let timer = Instant::now();
            gpu_buffer
                .copy_from_host(&flat_data)
                .expect("Failed to copy data to GPU");
            println!("IO took: {:?}", timer.elapsed());


            // Perform batched NTT on GPU
            let log_domain_size = log2_strict(lde_size);
            let ntt_config = NTTConfig {
                batches: num_polys as u32,
                are_inputs_on_device: true,
                are_outputs_on_device: true,
                with_coset: true,
                ..Default::default()
            };

            let timer = Instant::now();
            ntt_batch_ptr(0, gpu_buffer.as_mut_ptr(), log_domain_size, ntt_config);

            println!("comput took: {:?}", timer.elapsed());

            gpu_buffer
        });

        println!("Completed GPU coset FFT for {} polynomials", num_polys);

        // let total_elements = num_polys * lde_size;
        // let mut gpu_lde_values_copied_to_cpu = vec![F::ZERO; total_elements];

        // gpu_lde_values
        //     .copy_to_host(&mut gpu_lde_values_copied_to_cpu, total_elements)
        //     .expect("Failed to copy data from GPU");

        // println!("lde value: {:?}", gpu_lde_values_copied_to_cpu);

        // let lde_cpu_1d = lde_cpu.clone().into_iter().flatten().collect::<Vec<_>>();
        // assert_eq!(lde_cpu_1d.len(), gpu_lde_values_copied_to_cpu.len(), "LDE size mismatch");
        // assert_eq!(lde_cpu_1d, gpu_lde_values_copied_to_cpu, "LDE values mismatch");

        // Step 2: Transpose on GPU using Zeknox
        let gpu_transposed = timed!(timing, "GPU transpose", {
            let total_alloce_size = num_polys * lde_size;
            // let total_alloce_size = num_polys.next_power_of_two() * lde_size;

            let mut gpu_output = HostOrDeviceSlice::cuda_malloc(0, total_alloce_size)
                .expect("Failed to allocate GPU memory for transpose");

            let log_n = log2_strict(lde_size);
            let transpose_config = TransposeConfig {
                batches: num_polys as u32,
                are_inputs_on_device: true,
                are_outputs_on_device: true,
            };

            transpose_rev_batch(
                0,
                gpu_output.as_mut_ptr(),
                gpu_lde_values.as_ptr(),
                log_n,
                transpose_config,
            );

            // gpu_lde_values will be automatically freed when it goes out of scope

            gpu_output
        });
        print!("Completed GPU transpose for {} polynomials", num_polys);

        // Step 3: Copy back to CPU
        let leaves = timed!(timing, "GPU to CPU transfer", {
            let total_elements = num_polys * lde_size;
            let mut cpu_data = vec![F::ZERO; total_elements];

            gpu_transposed
                .copy_to_host(&mut cpu_data, total_elements)
                .expect("Failed to copy data from GPU");

            // // Reshape into leaves: Vec<Vec<F>> where each inner vec has num_polys elements
            // cpu_data
            //     .chunks(num_polys)
            //     .map(|chunk| chunk.to_vec())
            //     .collect::<Vec<_>>()

            cpu_data
        });

        // let mut leaves_cpu = timed!(timing, "transpose LDEs", transpose(&lde_cpu));
        // println!("tatal leaves: {}", leaves.len());
        // println!("leaves[0]:     {:?}", leaves[0]);
        // println!("leaves_cpu[0]: {:?}", leaves_cpu[0]);

        // reverse_index_bits_in_place(&mut leaves_cpu);
        // for i in 0..leaves.len() {
        //     if leaves[i] != leaves_cpu[i] {
        //         println!("Mismatch at leaf {}: \n{:?}\n{:?}\n", i, leaves[i], leaves_cpu[i]);
        //     }
        // }

        // assert!(leaves == leaves_cpu, "Transposed LDE values mismatch");

        // let merkle_tree = timed!(
        //     timing,
        //     "build Merkle tree",
        //     MerkleTree::new_from_2d(leaves, cap_height)
        // );

       let merkle_tree = timed!(
            timing,
            "build Merkle tree",
            MerkleTree::new_from_1d(leaves,polynomials.len(), cap_height)
        );

        Self {
            polynomials,
            merkle_tree,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
        }
    }

    pub(crate) fn lde_values(
        polynomials: &[PolynomialCoeffs<F>],
        rate_bits: usize,
        blinding: bool,
        fft_root_table: Option<&FftRootTable<F>>,
    ) -> Vec<Vec<F>> {
        let degree = polynomials[0].len();

        // If blinding, salt with two random elements to each leaf vector.
        let salt_size = if blinding { SALT_SIZE } else { 0 };
        println!(
            "lde_values: num_polys={}, degree={}, blinding={}, salt_size={}",
            polynomials.len(),
            degree,
            blinding,
            salt_size
        );

        polynomials
            .iter()
            .map(|p| {
                assert_eq!(p.len(), degree, "Polynomial degrees inconsistent");
                p.lde(rate_bits)
                    .coset_fft_with_options(F::coset_shift(), Some(rate_bits), fft_root_table)
                    .values
            })
            .chain(
                (0..salt_size)
                    .into_iter()
                    .map(|_| F::rand_vec(degree << rate_bits)),
            )
            .collect()
    }

    /// Fetches LDE values at the `index * step`th point.
    pub fn get_lde_values(&self, index: usize, step: usize) -> &[F] {
        let index = index * step;
        let index = reverse_bits(index, self.degree_log + self.rate_bits);
        let slice = &self.merkle_tree.get(index);
        &slice[..slice.len() - if self.blinding { SALT_SIZE } else { 0 }]
    }

    /// Like `get_lde_values`, but fetches LDE values from a batch of `P::WIDTH` points, and returns
    /// packed values.
    pub fn get_lde_values_packed<P>(&self, index_start: usize, step: usize) -> Vec<P>
    where
        P: PackedField<Scalar = F>,
    {
        let row_wise = (0..P::WIDTH)
            .map(|i| self.get_lde_values(index_start + i, step))
            .collect_vec();

        // This is essentially a transpose, but we will not use the generic transpose method as we
        // want inner lists to be of type P, not Vecs which would involve allocation.
        let leaf_size = row_wise[0].len();
        (0..leaf_size)
            .map(|j| {
                let mut packed = P::ZEROS;
                packed
                    .as_slice_mut()
                    .iter_mut()
                    .zip(&row_wise)
                    .for_each(|(packed_i, row_i)| *packed_i = row_i[j]);
                packed
            })
            .collect_vec()
    }

    /// Produces a batch opening proof.
    pub fn prove_openings(
        instance: &FriInstanceInfo<F, D>,
        oracles: &[&Self],
        challenger: &mut Challenger<F, C::Hasher>,
        fri_params: &FriParams,
        final_poly_coeff_len: Option<usize>,
        max_num_query_steps: Option<usize>,
        timing: &mut TimingTree,
    ) -> FriProof<F, C::Hasher, D> {
        assert!(D > 1, "Not implemented for D=1.");
        let alpha = challenger.get_extension_challenge::<D>();
        let mut alpha = ReducingFactor::new(alpha);

        // Final low-degree polynomial that goes into FRI.
        let mut final_poly = PolynomialCoeffs::empty();

        // Each batch `i` consists of an opening point `z_i` and polynomials `{f_ij}_j` to be opened at that point.
        // For each batch, we compute the composition polynomial `F_i = sum alpha^j f_ij`,
        // where `alpha` is a random challenge in the extension field.
        // The final polynomial is then computed as `final_poly = sum_i alpha^(k_i) (F_i(X) - F_i(z_i))/(X-z_i)`
        // where the `k_i`s are chosen such that each power of `alpha` appears only once in the final sum.
        // There are usually two batches for the openings at `zeta` and `g * zeta`.
        // The oracles used in Plonky2 are given in `FRI_ORACLES` in `plonky2/src/plonk/plonk_common.rs`.
        for FriBatchInfo { point, polynomials } in &instance.batches {
            // Collect the coefficients of all the polynomials in `polynomials`.
            let polys_coeff = polynomials.iter().map(|fri_poly| {
                &oracles[fri_poly.oracle_index].polynomials[fri_poly.polynomial_index]
            });
            let composition_poly = timed!(
                timing,
                &format!("reduce batch of {} polynomials", polynomials.len()),
                alpha.reduce_polys_base(polys_coeff)
            );
            let mut quotient = composition_poly.divide_by_linear(*point);
            quotient.coeffs.push(F::Extension::ZERO); // pad back to power of two
            alpha.shift_poly(&mut final_poly);
            final_poly += quotient;
        }

        let lde_final_poly = final_poly.lde(fri_params.config.rate_bits);
        let lde_final_values = timed!(
            timing,
            &format!("perform final FFT {}", lde_final_poly.len()),
            lde_final_poly.coset_fft(F::coset_shift().into())
        );

        let fri_proof = fri_proof::<F, C, D>(
            &oracles
                .par_iter()
                .map(|c| &c.merkle_tree)
                .collect::<Vec<_>>(),
            lde_final_poly,
            lde_final_values,
            challenger,
            fri_params,
            final_poly_coeff_len,
            max_num_query_steps,
            timing,
        );

        fri_proof
    }
}
