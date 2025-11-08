use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};

/// An example of using Plonky2 with Poseidon2 hash to prove a statement of the form
/// "I know the 100th element of the Fibonacci sequence, starting with constants a and b."
///
/// This example demonstrates the use of Poseidon2GoldilocksConfig, which uses:
/// - Poseidon2 hash for external commitments (Merkle trees)
/// - Poseidon hash for internal algebraic hashing (circuit constraints)
fn main() -> Result<()> {
    const D: usize = 2;
    type C = Poseidon2GoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // The arithmetic circuit.
    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;
    for _ in 0..99 {
        let temp = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = temp;
    }

    // Public inputs are the two initial values (provided below) and the result (which is generated).
    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);

    // Provide initial values.
    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::ZERO)?;
    pw.set_target(initial_b, F::ONE)?;

    println!("Building circuit with Poseidon2GoldilocksConfig...");
    let data = builder.build::<C>();

    println!("Generating proof...");
    let proof = data.prove(pw)?;

    println!(
        "100th Fibonacci number mod |F| (starting with {}, {}) is: {}",
        proof.public_inputs[0], proof.public_inputs[1], proof.public_inputs[2]
    );

    println!("Verifying proof...");
    data.verify(proof)?;

    println!("Proof verified successfully with Poseidon2!");
    Ok(())
}
