#[cfg(not(feature = "std"))]
use alloc::{sync::Arc, vec, vec::Vec};
#[cfg(feature = "std")]
use std::sync::{Arc, Once};

use itertools::Itertools;
use log::Level;

use crate::field::types::Field;
use crate::gadgets::lookup::{hash_dynamic_lookup_table, OTHER_TABLE, SMALLER_TABLE, TIP5_TABLE};
use crate::gates::lookup_table::LookupTable;
use crate::gates::noop::NoopGate;
use crate::iop::witness::{PartialWitness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
use crate::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};
use crate::plonk::prover::prove;
use crate::util::serialization::{DefaultGateSerializer, DefaultGeneratorSerializer};
use crate::util::timing::TimingTree;

const D: usize = 2;
type C = Poseidon2GoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type H = <C as GenericConfig<D>>::InnerHasher;

const LUT_SIZE: usize = u16::MAX as usize + 1;

#[cfg(feature = "std")]
static LOGGER_INITIALIZED: Once = Once::new();

#[test]
fn test_no_lookup() -> anyhow::Result<()> {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    builder.add_gate(NoopGate, vec![]);
    let pw = PartialWitness::new();

    let data = builder.build::<C>();
    let mut timing = TimingTree::new("prove first", Level::Debug);
    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    timing.print();
    data.verify(proof)?;

    Ok(())
}

#[should_panic]
#[test]
fn test_lookup_table_not_used() {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let tip5_table = TIP5_TABLE.to_vec();
    let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());
    builder.add_lookup_table_from_pairs(table);

    builder.build::<C>();
}

#[should_panic]
#[test]
fn test_lookup_without_table() {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let dummy = builder.add_virtual_target();
    builder.add_lookup_from_index(dummy, 0);

    builder.build::<C>();
}

// Tests two lookups in one lookup table.
#[test]
fn test_one_lookup() -> anyhow::Result<()> {
    init_logger();

    let tip5_table = TIP5_TABLE.to_vec();
    let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();

    let look_val_a = 1;
    let look_val_b = 2;

    let out_a = table[look_val_a].1;
    let out_b = table[look_val_b].1;
    let table_index = builder.add_lookup_table_from_pairs(table);
    let output_a = builder.add_lookup_from_index(initial_a, table_index);

    let output_b = builder.add_lookup_from_index(initial_b, table_index);

    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(output_a);
    builder.register_public_input(output_b);

    let mut pw = PartialWitness::new();

    pw.set_target(initial_a, F::from_canonical_usize(look_val_a))?;
    pw.set_target(initial_b, F::from_canonical_usize(look_val_b))?;

    let data = builder.build::<C>();
    let mut timing = TimingTree::new("prove one lookup", Level::Debug);
    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    timing.print();
    data.verify(proof.clone())?;

    assert!(
        proof.public_inputs[2] == F::from_canonical_u16(out_a),
        "First lookup, at index {} in the Tip5 table gives an incorrect output.",
        proof.public_inputs[0]
    );
    assert!(
        proof.public_inputs[3] == F::from_canonical_u16(out_b),
        "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
        proof.public_inputs[1]
    );

    Ok(())
}

#[test]
fn test_dynamic_lookup_table_digest_is_public() -> anyhow::Result<()> {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let input = builder.add_virtual_target();
    let output = builder.add_virtual_target();
    let table_index = builder.add_dynamic_lookup_table(8);
    let table_digest = builder.dynamic_lookup_table_digest(table_index);
    builder.add_dynamic_lookup(input, output, table_index);
    builder.register_public_input(input);
    builder.register_public_input(output);

    let data = builder.build::<C>();
    let table: LookupTable = Arc::new((0..8).map(|i| (10 + i, 20 + i)).collect());

    let prove_for = |input_value: u16, output_value: u16| {
        let mut pw = PartialWitness::new();
        pw.set_target(input, F::from_canonical_u16(input_value))?;
        pw.set_target(output, F::from_canonical_u16(output_value))?;
        pw.set_hash_target(
            table_digest,
            hash_dynamic_lookup_table::<F, H>(table_index, &table),
        )?;
        data.prove_with_dynamic_lookup_tables(pw, core::slice::from_ref(&table))
    };

    let proof_a = prove_for(12, 22)?;
    let proof_b = prove_for(15, 25)?;
    data.verify(proof_a.clone()).expect("proof A");
    data.verify(proof_b.clone()).expect("proof B");
    assert_eq!(&proof_a.public_inputs[..4], &proof_b.public_inputs[..4]);

    let changed_table: LookupTable = Arc::new(
        (0..8)
            .map(|i| {
                let pair = (10 + i, 20 + i);
                if i == 7 {
                    (pair.0, 99)
                } else {
                    pair
                }
            })
            .collect(),
    );
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u16(12))?;
    pw.set_target(output, F::from_canonical_u16(22))?;
    pw.set_hash_target(
        table_digest,
        hash_dynamic_lookup_table::<F, H>(table_index, &changed_table),
    )?;
    let proof_c = data.prove_with_dynamic_lookup_tables(pw, &[changed_table])?;
    data.verify(proof_c.clone()).expect("proof C");
    assert_ne!(&proof_a.public_inputs[..4], &proof_c.public_inputs[..4]);

    Ok(())
}

#[test]
fn test_static_table_does_not_alias_dynamic_placeholder() {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let dynamic_index = builder.add_dynamic_lookup_table(8);
    let identity_table: LookupTable = Arc::new((0..8).map(|i| (i, i)).collect());
    let static_index = builder.add_lookup_table_from_pairs(identity_table);

    assert_ne!(dynamic_index, static_index);
}

#[test]
fn test_dynamic_lookup_with_prover_circuit_data() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let input = builder.add_virtual_target();
    let output = builder.add_virtual_target();
    let table_index = builder.add_dynamic_lookup_table(2);
    let table_digest = builder.dynamic_lookup_table_digest(table_index);
    builder.add_dynamic_lookup(input, output, table_index);

    let data = builder.build::<C>();
    let verifier_data = data.verifier_data();
    let prover_data = data.prover_data();
    let table: LookupTable = Arc::new(vec![(10, 20), (11, 21)]);
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u16(11))?;
    pw.set_target(output, F::from_canonical_u16(21))?;
    pw.set_hash_target(
        table_digest,
        hash_dynamic_lookup_table::<F, H>(table_index, &table),
    )?;

    let proof = prover_data.prove_with_dynamic_lookup_tables(pw, &[table])?;
    verifier_data.verify(proof)
}

#[test]
fn test_dynamic_lookup_rejects_unlisted_pair() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let input = builder.add_virtual_target();
    let output = builder.add_virtual_target();
    let table_index = builder.add_dynamic_lookup_table(2);
    let table_digest = builder.dynamic_lookup_table_digest(table_index);
    builder.add_dynamic_lookup(input, output, table_index);

    let data = builder.build::<C>();
    let table: LookupTable = Arc::new(vec![(10, 20), (11, 21)]);
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u16(10))?;
    pw.set_target(output, F::from_canonical_u16(21))?;
    pw.set_hash_target(
        table_digest,
        hash_dynamic_lookup_table::<F, H>(table_index, &table),
    )?;

    assert!(data.prove_with_dynamic_lookup_tables(pw, &[table]).is_err());
    Ok(())
}

#[test]
fn test_dynamic_lookup_requires_output_witness() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let input = builder.add_virtual_target();
    let output = builder.add_virtual_target();
    let table_index = builder.add_dynamic_lookup_table(2);
    let table_digest = builder.dynamic_lookup_table_digest(table_index);
    builder.add_dynamic_lookup(input, output, table_index);

    let data = builder.build::<C>();
    let table: LookupTable = Arc::new(vec![(10, 20), (11, 21)]);
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u16(10))?;
    pw.set_hash_target(
        table_digest,
        hash_dynamic_lookup_table::<F, H>(table_index, &table),
    )?;

    let error = data
        .prove_with_dynamic_lookup_tables(pw, &[table])
        .unwrap_err();
    assert!(error.to_string().contains("lookup output is missing"));
    Ok(())
}

#[test]
fn test_dynamic_lookup_reported_double_table() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let input = builder.add_virtual_target();
    let output = builder.add_virtual_target();
    let table_index = builder.add_dynamic_lookup_table(8);
    let table_digest = builder.dynamic_lookup_table_digest(table_index);
    builder.add_dynamic_lookup(input, output, table_index);
    builder.register_public_input(input);
    builder.register_public_input(output);

    let data = builder.build::<C>();
    let table: LookupTable = Arc::new(vec![
        (0, 0),
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
        (5, 10),
        (6, 12),
        (7, 14),
    ]);
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u16(7))?;
    pw.set_target(output, F::from_canonical_u16(14))?;
    pw.set_hash_target(
        table_digest,
        hash_dynamic_lookup_table::<F, H>(table_index, &table),
    )?;

    let proof = data.prove_with_dynamic_lookup_tables(pw, &[table])?;
    data.verify(proof)
}

#[test]
fn test_dynamic_lookup_rejects_wrong_public_digest() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let input = builder.add_virtual_target();
    let output = builder.add_virtual_target();
    let table_index = builder.add_dynamic_lookup_table(2);
    let table_digest = builder.dynamic_lookup_table_digest(table_index);
    builder.add_dynamic_lookup(input, output, table_index);

    let data = builder.build::<C>();
    let table: LookupTable = Arc::new(vec![(10, 20), (11, 21)]);
    let mut digest = hash_dynamic_lookup_table::<F, H>(table_index, &table);
    digest.elements[0] += F::ONE;
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u16(10))?;
    pw.set_target(output, F::from_canonical_u16(20))?;
    pw.set_hash_target(table_digest, digest)?;

    assert!(data.prove_with_dynamic_lookup_tables(pw, &[table]).is_err());
    Ok(())
}

#[test]
fn test_dynamic_lookup_circuit_serialization() -> anyhow::Result<()> {
    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);
    let input = builder.add_virtual_target();
    let output = builder.add_virtual_target();
    let table_index = builder.add_dynamic_lookup_table(2);
    let table_digest = builder.dynamic_lookup_table_digest(table_index);
    builder.add_dynamic_lookup(input, output, table_index);

    let data = builder.build::<C>();
    let bytes = data
        .to_bytes(
            &DefaultGateSerializer,
            &DefaultGeneratorSerializer::<C, D>::default(),
        )
        .map_err(|_| anyhow::anyhow!("failed to serialize dynamic lookup circuit"))?;
    let data = CircuitData::<F, C, D>::from_bytes(
        &bytes,
        &DefaultGateSerializer,
        &DefaultGeneratorSerializer::<C, D>::default(),
    )
    .map_err(|_| anyhow::anyhow!("failed to deserialize dynamic lookup circuit"))?;

    let table: LookupTable = Arc::new(vec![(10, 20), (11, 21)]);
    let mut pw = PartialWitness::new();
    pw.set_target(input, F::from_canonical_u16(11))?;
    pw.set_target(output, F::from_canonical_u16(21))?;
    pw.set_hash_target(
        table_digest,
        hash_dynamic_lookup_table::<F, H>(table_index, &table),
    )?;
    let proof = data.prove_with_dynamic_lookup_tables(pw, &[table])?;
    data.verify(proof)
}

#[test]
fn test_dynamic_lookup_bitmap_changes_circuit_digest() {
    let identity_table: LookupTable = Arc::new((0..8).map(|i| (i, i)).collect());

    let config = CircuitConfig::standard_recursion_config();
    let mut fixed_dynamic = CircuitBuilder::<F, D>::new(config.clone());
    let fixed_input = fixed_dynamic.add_virtual_target();
    let fixed_index = fixed_dynamic.add_lookup_table_from_pairs(identity_table);
    fixed_dynamic.add_lookup_from_index(fixed_input, fixed_index);
    let dynamic_input = fixed_dynamic.add_virtual_target();
    let dynamic_output = fixed_dynamic.add_virtual_target();
    let dynamic_index = fixed_dynamic.add_dynamic_lookup_table(8);
    fixed_dynamic.add_dynamic_lookup(dynamic_input, dynamic_output, dynamic_index);
    let fixed_dynamic = fixed_dynamic.build::<C>();

    let mut dynamic_dynamic = CircuitBuilder::<F, D>::new(config);
    for _ in 0..2 {
        let input = dynamic_dynamic.add_virtual_target();
        let output = dynamic_dynamic.add_virtual_target();
        let table_index = dynamic_dynamic.add_dynamic_lookup_table(8);
        dynamic_dynamic.add_dynamic_lookup(input, output, table_index);
    }
    let dynamic_dynamic = dynamic_dynamic.build::<C>();

    assert_ne!(
        fixed_dynamic.verifier_only.circuit_digest,
        dynamic_dynamic.verifier_only.circuit_digest
    );
}

// Tests one lookup in two different lookup tables.
#[test]
fn test_two_luts() -> anyhow::Result<()> {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();

    let look_val_a = 1;
    let look_val_b = 2;

    let tip5_table = TIP5_TABLE.to_vec();

    let first_out = tip5_table[look_val_a];
    let second_out = tip5_table[look_val_b];

    let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

    let other_table = OTHER_TABLE.to_vec();

    let table_index = builder.add_lookup_table_from_pairs(table);
    let output_a = builder.add_lookup_from_index(initial_a, table_index);

    let output_b = builder.add_lookup_from_index(initial_b, table_index);
    let sum = builder.add(output_a, output_b);

    let s = first_out + second_out;
    let final_out = other_table[s as usize];

    let table2: LookupTable = Arc::new((0..256).zip_eq(other_table).collect());
    let table2_index = builder.add_lookup_table_from_pairs(table2);

    let output_final = builder.add_lookup_from_index(sum, table2_index);

    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(sum);
    builder.register_public_input(output_a);
    builder.register_public_input(output_b);
    builder.register_public_input(output_final);

    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::from_canonical_usize(look_val_a))?;
    pw.set_target(initial_b, F::from_canonical_usize(look_val_b))?;
    let data = builder.build::<C>();
    let mut timing = TimingTree::new("prove two_luts", Level::Debug);
    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    data.verify(proof.clone())?;
    timing.print();

    assert!(
        proof.public_inputs[3] == F::from_canonical_u16(first_out),
        "First lookup, at index {} in the Tip5 table gives an incorrect output.",
        proof.public_inputs[0]
    );
    assert!(
        proof.public_inputs[4] == F::from_canonical_u16(second_out),
        "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
        proof.public_inputs[1]
    );
    assert!(
        proof.public_inputs[2] == F::from_canonical_u16(s),
        "Sum between the first two LUT outputs is incorrect."
    );
    assert!(
        proof.public_inputs[5] == F::from_canonical_u16(final_out),
        "Output of the second LUT at index {} is incorrect.",
        s
    );

    Ok(())
}

#[test]
fn test_different_inputs() -> anyhow::Result<()> {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();

    let init_a = 1;
    let init_b = 2;

    let tab: Vec<u16> = SMALLER_TABLE.to_vec();
    let table: LookupTable = Arc::new((2..10).zip_eq(tab).collect());

    let other_table = OTHER_TABLE.to_vec();

    let table2: LookupTable = Arc::new((0..256).zip_eq(other_table).collect());
    let small_index = builder.add_lookup_table_from_pairs(table.clone());
    let output_a = builder.add_lookup_from_index(initial_a, small_index);

    let output_b = builder.add_lookup_from_index(initial_b, small_index);
    let sum = builder.add(output_a, output_b);

    let other_index = builder.add_lookup_table_from_pairs(table2.clone());
    let output_final = builder.add_lookup_from_index(sum, other_index);

    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(sum);
    builder.register_public_input(output_a);
    builder.register_public_input(output_b);
    builder.register_public_input(output_final);

    let mut pw = PartialWitness::new();

    let look_val_a = table[init_a].0;
    let look_val_b = table[init_b].0;
    pw.set_target(initial_a, F::from_canonical_u16(look_val_a))?;
    pw.set_target(initial_b, F::from_canonical_u16(look_val_b))?;

    let data = builder.build::<C>();
    let mut timing = TimingTree::new("prove different lookups", Level::Debug);
    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    data.verify(proof.clone())?;
    timing.print();

    let out_a = table[init_a].1;
    let out_b = table[init_b].1;
    let s = out_a + out_b;
    let out_final = table2[s as usize].1;

    assert!(
        proof.public_inputs[3] == F::from_canonical_u16(out_a),
        "First lookup, at index {} in the smaller LUT gives an incorrect output.",
        proof.public_inputs[0]
    );
    assert!(
        proof.public_inputs[4] == F::from_canonical_u16(out_b),
        "Second lookup, at index {} in the smaller LUT gives an incorrect output.",
        proof.public_inputs[1]
    );
    assert!(
        proof.public_inputs[2] == F::from_canonical_u16(s),
        "Sum between the first two LUT outputs is incorrect."
    );
    assert!(
        proof.public_inputs[5] == F::from_canonical_u16(out_final),
        "Output of the second LUT at index {} is incorrect.",
        s
    );

    Ok(())
}

// This test looks up over 514 values for one LookupTableGate, which means that several LookupGates are created.
#[test]
fn test_many_lookups() -> anyhow::Result<()> {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();

    let look_val_a = 1;
    let look_val_b = 2;

    let tip5_table = TIP5_TABLE.to_vec();
    let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

    let out_a = table[look_val_a].1;
    let out_b = table[look_val_b].1;

    let tip5_index = builder.add_lookup_table_from_pairs(table);
    let output_a = builder.add_lookup_from_index(initial_a, tip5_index);

    let output_b = builder.add_lookup_from_index(initial_b, tip5_index);
    let sum = builder.add(output_a, output_b);

    for _ in 0..514 {
        builder.add_lookup_from_index(initial_a, tip5_index);
    }

    let other_table = OTHER_TABLE.to_vec();

    let table2: LookupTable = Arc::new((0..256).zip_eq(other_table).collect());

    let s = out_a + out_b;
    let out_final = table2[s as usize].1;

    let other_index = builder.add_lookup_table_from_pairs(table2);
    let output_final = builder.add_lookup_from_index(sum, other_index);

    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(sum);
    builder.register_public_input(output_a);
    builder.register_public_input(output_b);
    builder.register_public_input(output_final);

    let mut pw = PartialWitness::new();

    pw.set_target(initial_a, F::from_canonical_usize(look_val_a))?;
    pw.set_target(initial_b, F::from_canonical_usize(look_val_b))?;

    let data = builder.build::<C>();
    let mut timing = TimingTree::new("prove different lookups", Level::Debug);
    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;

    data.verify(proof.clone())?;
    timing.print();

    assert!(
        proof.public_inputs[3] == F::from_canonical_u16(out_a),
        "First lookup, at index {} in the Tip5 table gives an incorrect output.",
        proof.public_inputs[0]
    );
    assert!(
        proof.public_inputs[4] == F::from_canonical_u16(out_b),
        "Second lookup, at index {} in the Tip5 table gives an incorrect output.",
        proof.public_inputs[1]
    );
    assert!(
        proof.public_inputs[2] == F::from_canonical_u16(s),
        "Sum between the first two LUT outputs is incorrect."
    );
    assert!(
        proof.public_inputs[5] == F::from_canonical_u16(out_final),
        "Output of the second LUT at index {} is incorrect.",
        s
    );

    Ok(())
}

// Tests whether, when adding the same LUT to the circuit, the circuit only adds one copy, with the same index.
#[test]
fn test_same_luts() -> anyhow::Result<()> {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();

    let look_val_a = 1;
    let look_val_b = 2;

    let tip5_table = TIP5_TABLE.to_vec();
    let table: LookupTable = Arc::new((0..256).zip_eq(tip5_table).collect());

    let table_index = builder.add_lookup_table_from_pairs(table.clone());
    let output_a = builder.add_lookup_from_index(initial_a, table_index);

    let output_b = builder.add_lookup_from_index(initial_b, table_index);
    let sum = builder.add(output_a, output_b);

    let table2_index = builder.add_lookup_table_from_pairs(table);

    let output_final = builder.add_lookup_from_index(sum, table2_index);

    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(sum);
    builder.register_public_input(output_a);
    builder.register_public_input(output_b);
    builder.register_public_input(output_final);

    let luts_length = builder.get_luts_length();

    assert!(
        luts_length == 1,
        "There are {} LUTs when there should be only one",
        luts_length
    );

    let mut pw = PartialWitness::new();

    pw.set_target(initial_a, F::from_canonical_usize(look_val_a))?;
    pw.set_target(initial_b, F::from_canonical_usize(look_val_b))?;

    let data = builder.build::<C>();
    let mut timing = TimingTree::new("prove two_luts", Level::Debug);
    let proof = prove(&data.prover_only, &data.common, pw, &mut timing)?;
    data.verify(proof)?;
    timing.print();

    Ok(())
}

#[test]
fn test_big_lut() -> anyhow::Result<()> {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let inputs: [u16; LUT_SIZE] = core::array::from_fn(|i| i as u16);
    let lut_fn = |inp: u16| inp / 10;
    let lut_index = builder.add_lookup_table_from_fn(lut_fn, &inputs);

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();

    let look_val_a = 51;
    let look_val_b = 2;

    let output_a = builder.add_lookup_from_index(initial_a, lut_index);
    let output_b = builder.add_lookup_from_index(initial_b, lut_index);

    builder.register_public_input(output_a);
    builder.register_public_input(output_b);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::new();

    pw.set_target(initial_a, F::from_canonical_u16(look_val_a))?;
    pw.set_target(initial_b, F::from_canonical_u16(look_val_b))?;

    let proof = data.prove(pw)?;
    assert_eq!(
        proof.public_inputs[0],
        F::from_canonical_u16(lut_fn(look_val_a))
    );
    assert_eq!(
        proof.public_inputs[1],
        F::from_canonical_u16(lut_fn(look_val_b))
    );

    data.verify(proof)
}

#[test]
fn test_many_lookups_on_big_lut() -> anyhow::Result<()> {
    init_logger();

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let inputs: [u16; LUT_SIZE] = core::array::from_fn(|i| i as u16);
    let lut_fn = |inp: u16| inp / 10;
    let lut_index = builder.add_lookup_table_from_fn(lut_fn, &inputs);

    let inputs = (0..LUT_SIZE)
        .map(|_| {
            let input_target = builder.add_virtual_target();
            _ = builder.add_lookup_from_index(input_target, lut_index);
            input_target
        })
        .collect::<Vec<_>>();

    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();

    let look_val_a = 51;
    let look_val_b = 2;

    let output_a = builder.add_lookup_from_index(initial_a, lut_index);
    let output_b = builder.add_lookup_from_index(initial_b, lut_index);
    let sum = builder.add(output_a, output_b);

    builder.register_public_input(sum);

    let data = builder.build::<C>();

    let mut pw = PartialWitness::new();

    for (i, t) in inputs.into_iter().enumerate() {
        pw.set_target(t, F::from_canonical_usize(i))?
    }
    pw.set_target(initial_a, F::from_canonical_u16(look_val_a))?;
    pw.set_target(initial_b, F::from_canonical_u16(look_val_b))?;

    let proof = data.prove(pw)?;
    assert_eq!(
        proof.public_inputs[0],
        F::from_canonical_u16(lut_fn(look_val_a) + lut_fn(look_val_b))
    );

    data.verify(proof)
}

fn init_logger() {
    #[cfg(feature = "std")]
    {
        LOGGER_INITIALIZED.call_once(|| {
            let mut builder = env_logger::Builder::from_default_env();
            builder.format_timestamp(None);
            builder.filter_level(log::LevelFilter::Debug);

            builder.try_init().unwrap();
        });
    }
}
