use anyhow::{Ok, Result};
use log::Level;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;

/// An example of using Plonky2 to prove a statement of the form
/// "I know the 100th element of the Fibonacci sequence, starting with constants a and b."
/// When a == 0 and b == 1, this is proving knowledge of the 100th (standard) Fibonacci number.
fn main() -> Result<()> {
    // Initialize logger to see timing output
    env_logger::Builder::from_default_env()
        .format_timestamp(None)
        .filter_level(log::LevelFilter::Debug)
        .init();
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let config = CircuitConfig::standard_recursion_config();
    println!("Building circuit...");
    let mut builder = CircuitBuilder::<F, D>::new(config);
    println!("Building arithmetic circuit...");
    // The arithmetic circuit.
    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;
    for _ in 0..9999999 {
        let temp = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = temp;
    }
    println!("Circuit built.");

    #[cfg(feature = "cuda")]
    {
        zeknox::clear_cuda_errors_rs();
        println!("Initializing CUDA twiddle factors...");
        // Initialize twiddle factors for all dimensions that will be used
        // This test involves multiple polynomials and recursive verification,
        // so we initialize a wider range of dimensions to be safe
        // for i in 0..=19 {
        //     zeknox::init_twiddle_factors_rs(0, i);
        // }

        zeknox::init_twiddle_factors_rs(0, 19);
        zeknox::init_twiddle_factors_rs(0, 22);
        // Initialize coset on GPU
        // For Goldilocks field, the coset generator is 7 (MULTIPLICATIVE_GROUP_GENERATOR)
        // TODO: Make this generic for other fields if needed
        let coset_gen_u64 = 7u64;
        zeknox::init_coset_rs(0, 22, coset_gen_u64);
        // zeknox::init_coset_rs(0, 16, coset_gen_u64);
    }

    // Public inputs are the two initial values (provided below) and the result (which is generated).
    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);

    // Provide initial values.
    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::ZERO)?;
    pw.set_target(initial_b, F::ONE)?;

    let data = builder.build::<C>();

    #[cfg(feature = "timing")]
    {
        use log::Level;
        use plonky2::util::timing::TimingTree;
        let mut timing = TimingTree::new("prove", Level::Info);
        println!("Starting proof generation...");
        let proof =
            plonky2::plonk::prover::prove(&data.prover_only, &data.common, pw, &mut timing)?;

        println!(
            "100th Fibonacci number mod |F| (starting with {}, {}) is: {}",
            proof.public_inputs[0], proof.public_inputs[1], proof.public_inputs[2]
        );

        // Print first few elements of wires_cap for comparison
        println!("First wires_cap hash: {:?}", proof.proof.wires_cap.0[0]);
        println!(
            "First plonk_zs hash: {:?}",
            proof.proof.plonk_zs_partial_products_cap.0[0]
        );
        println!(
            "First quotient hash: {:?}",
            proof.proof.quotient_polys_cap.0[0]
        );

        timing.print();
        data.verify(proof)?;
    }

    #[cfg(not(feature = "timing"))]
    {
        let proof = data.prove(pw)?;
        println!(
            "100th Fibonacci number mod |F| (starting with {}, {}) is: {}",
            proof.public_inputs[0], proof.public_inputs[1], proof.public_inputs[2]
        );
        data.verify(proof)?;
    }

    println!("finished");
    Ok(())
}
