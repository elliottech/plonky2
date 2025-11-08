use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig, PoseidonGoldilocksConfig};
use std::time::Instant;

/// Benchmark comparing Poseidon vs Poseidon2 hash function performance in the prover.
/// This benchmark measures:
/// - Circuit building time
/// - Proof generation time
/// - Proof verification time
/// - Total time
///
/// The benchmark uses a Fibonacci circuit which exercises:
/// - Merkle tree construction (external hashing)
/// - Public input hashing (internal hashing)
/// - FRI commitments and openings

fn run_benchmark_poseidon(circuit_size: usize) -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    println!("\n============================================================");
    println!("Benchmarking with PoseidonGoldilocksConfig");
    println!("Circuit size: {} Fibonacci iterations", circuit_size);
    println!("============================================================");

    // Build the circuit
    let build_start = Instant::now();
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // Create a Fibonacci circuit
    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;

    for _ in 0..circuit_size {
        let temp = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = temp;
    }

    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);

    let data = builder.build::<C>();
    let build_time = build_start.elapsed();
    println!("Circuit building time:    {:>12.2?}", build_time);

    // Print circuit statistics
    println!("\nCircuit statistics:");
    println!("  - Degree:               {:>12}", data.common.degree());
    println!("  - Number of gates:      {:>12}", data.common.gates.len());
    println!("  - Quotient degree:      {:>12}", data.common.quotient_degree_factor);

    // Prepare witness
    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::ZERO)?;
    pw.set_target(initial_b, F::ONE)?;

    // Generate proof
    println!("\nProof generation:");
    let prove_start = Instant::now();
    let proof = data.prove(pw)?;
    let prove_time = prove_start.elapsed();
    println!("  Time:                   {:>12.2?}", prove_time);

    // Print proof statistics
    println!("\nProof statistics:");
    println!("  - Public inputs:        {:>12}", proof.public_inputs.len());
    println!(
        "  - Result (F[{}]):        {:>12}",
        circuit_size, proof.public_inputs[2]
    );

    // Verify proof
    println!("\nProof verification:");
    let verify_start = Instant::now();
    data.verify(proof)?;
    let verify_time = verify_start.elapsed();
    println!("  Time:                   {:>12.2?}", verify_time);

    // Total time
    let total_time = build_time + prove_time + verify_time;
    println!("\nTotal time:               {:>12.2?}", total_time);

    println!("\nBreakdown:");
    println!(
        "  - Build:     {:>6.2?} ({:>5.1}%)",
        build_time,
        (build_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - Prove:     {:>6.2?} ({:>5.1}%)",
        prove_time,
        (prove_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - Verify:    {:>6.2?} ({:>5.1}%)",
        verify_time,
        (verify_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );

    Ok(())
}

fn run_benchmark_poseidon2(circuit_size: usize) -> Result<()> {
    const D: usize = 2;
    type C = Poseidon2GoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    println!("\n============================================================");
    println!("Benchmarking with Poseidon2GoldilocksConfig");
    println!("Circuit size: {} Fibonacci iterations", circuit_size);
    println!("============================================================");

    // Build the circuit
    let build_start = Instant::now();
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // Create a Fibonacci circuit
    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;

    for _ in 0..circuit_size {
        let temp = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = temp;
    }

    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);

    let data = builder.build::<C>();
    let build_time = build_start.elapsed();
    println!("Circuit building time:    {:>12.2?}", build_time);

    // Print circuit statistics
    println!("\nCircuit statistics:");
    println!("  - Degree:               {:>12}", data.common.degree());
    println!("  - Number of gates:      {:>12}", data.common.gates.len());
    println!("  - Quotient degree:      {:>12}", data.common.quotient_degree_factor);

    // Prepare witness
    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::ZERO)?;
    pw.set_target(initial_b, F::ONE)?;

    // Generate proof
    println!("\nProof generation:");
    let prove_start = Instant::now();
    let proof = data.prove(pw)?;
    let prove_time = prove_start.elapsed();
    println!("  Time:                   {:>12.2?}", prove_time);

    // Print proof statistics
    println!("\nProof statistics:");
    println!("  - Public inputs:        {:>12}", proof.public_inputs.len());
    println!(
        "  - Result (F[{}]):        {:>12}",
        circuit_size, proof.public_inputs[2]
    );

    // Verify proof
    println!("\nProof verification:");
    let verify_start = Instant::now();
    data.verify(proof)?;
    let verify_time = verify_start.elapsed();
    println!("  Time:                   {:>12.2?}", verify_time);

    // Total time
    let total_time = build_time + prove_time + verify_time;
    println!("\nTotal time:               {:>12.2?}", total_time);

    println!("\nBreakdown:");
    println!(
        "  - Build:     {:>6.2?} ({:>5.1}%)",
        build_time,
        (build_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - Prove:     {:>6.2?} ({:>5.1}%)",
        prove_time,
        (prove_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );
    println!(
        "  - Verify:    {:>6.2?} ({:>5.1}%)",
        verify_time,
        (verify_time.as_secs_f64() / total_time.as_secs_f64()) * 100.0
    );

    Ok(())
}

fn main() -> Result<()> {
    println!("\n");
    println!("============================================================");
    println!("       POSEIDON vs POSEIDON2 BENCHMARK");
    println!("============================================================");
    println!("\nThis benchmark compares the performance of:");
    println!("  1. PoseidonGoldilocksConfig  - Uses Poseidon for all hashing");
    println!("  2. Poseidon2GoldilocksConfig - Uses Poseidon2 for Merkle trees,");
    println!("                                 Poseidon for circuit constraints");

    // You can adjust the circuit size here
    // Larger circuits will show more pronounced differences
    let circuit_sizes = vec![100, 500, 1000];

    for &size in &circuit_sizes {
        println!("\n");
        println!("************************************************************");
        println!("               Circuit Size: {}", size);
        println!("************************************************************");

        // Run with Poseidon
        run_benchmark_poseidon(size)?;

        // Run with Poseidon2
        run_benchmark_poseidon2(size)?;
    }

    println!("\n");
    println!("============================================================");
    println!("              BENCHMARK COMPLETE");
    println!("============================================================");
    println!("\nNote: Poseidon2 performance improvements are mainly in:");
    println!("  - Merkle tree construction (during proof generation)");
    println!("  - FRI commitments and openings");
    println!("  - Larger circuits show more pronounced differences");
    println!("\n");

    Ok(())
}
