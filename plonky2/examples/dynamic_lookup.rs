use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use plonky2::field::types::Field;
use plonky2::gadgets::lookup::hash_dynamic_lookup_table;
use plonky2::gates::lookup_table::LookupTable;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};

const D: usize = 2;
type C = Poseidon2GoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type H = <C as GenericConfig<D>>::InnerHasher;

const TABLE_SIZE: usize = 2_000;
const NUM_LOOKUPS: usize = 256;

fn table_output(input: u16) -> u16 {
    ((u32::from(input) * 3 + 17) % (u32::from(u16::MAX) + 1)) as u16
}

fn main() -> Result<()> {
    let total_start = Instant::now();

    let table: LookupTable = Arc::new(
        (0..TABLE_SIZE)
            .map(|i| {
                let input = i as u16;
                (input, table_output(input))
            })
            .collect(),
    );

    let config = CircuitConfig::standard_recursion_config();
    let witness_columns = config.num_wires;
    let routed_columns = config.num_routed_wires;
    let lookups_per_row = routed_columns / 2;
    let table_entries_per_row = routed_columns / 3;

    let build_start = Instant::now();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let table_index = builder.add_dynamic_lookup_table(TABLE_SIZE);
    let table_digest_target = builder.dynamic_lookup_table_digest(table_index);

    let mut lookup_targets = Vec::with_capacity(NUM_LOOKUPS);
    for _ in 0..NUM_LOOKUPS {
        let input = builder.add_virtual_target();
        let output = builder.add_virtual_target();
        builder.add_dynamic_lookup(input, output, table_index);
        lookup_targets.push((input, output));
    }

    let data = builder.build::<C>();
    let build_time = build_start.elapsed();

    let witness_start = Instant::now();
    let mut pw = PartialWitness::new();
    for (i, &(input_target, output_target)) in lookup_targets.iter().enumerate() {
        // 7 is coprime to 2000, so these are 256 distinct table indices.
        let input = ((i * 7) % TABLE_SIZE) as u16;
        pw.set_target(input_target, F::from_canonical_u16(input))?;
        pw.set_target(output_target, F::from_canonical_u16(table_output(input)))?;
    }
    let table_digest = hash_dynamic_lookup_table::<F, H>(table_index, &table);
    pw.set_hash_target(table_digest_target, table_digest)?;
    let witness_time = witness_start.elapsed();

    let prove_start = Instant::now();
    let proof = data.prove_with_dynamic_lookup_tables(pw, &[table])?;
    let prove_time = prove_start.elapsed();
    let proof_size = proof.to_bytes().len();

    let verify_start = Instant::now();
    data.verify(proof)?;
    let verify_time = verify_start.elapsed();

    let rows = data.common.degree();
    let degree_bits = data.common.degree_bits();
    let lookup_gate_rows = NUM_LOOKUPS.div_ceil(lookups_per_row);
    let table_gate_rows = TABLE_SIZE.div_ceil(table_entries_per_row);

    println!("Dynamic LogUp + Poseidon2 example");
    println!("  table entries             : {TABLE_SIZE}");
    println!("  lookup requests           : {NUM_LOOKUPS}");
    println!("  circuit rows              : {rows} (2^{degree_bits})");
    println!("  witness columns           : {witness_columns}");
    println!("  routed columns            : {routed_columns}");
    println!("  total witness cells       : {}", rows * witness_columns);
    println!("  lookups per lookup row    : {lookups_per_row}");
    println!("  lookup gate rows          : {lookup_gate_rows}");
    println!("  entries per table row     : {table_entries_per_row}");
    println!("  table gate rows           : {table_gate_rows}");
    println!(
        "  public inputs             : {}",
        data.common.num_public_inputs
    );
    println!(
        "  witness Merkle cap nodes  : {}",
        data.common.config.fri_config.num_cap_elements()
    );
    println!("  proof size                : {proof_size} bytes");
    println!("  table digest              : {:?}", table_digest.elements);
    println!("  build time                : {build_time:?}");
    println!("  witness setup time        : {witness_time:?}");
    println!("  prove time                : {prove_time:?}");
    println!("  verify time               : {verify_time:?}");
    println!("  total time                : {:?}", total_start.elapsed());

    Ok(())
}
