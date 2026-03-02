// Copyright (c) Elliot Technologies, Inc.
// SPDX-License-Identifier: BUSL-1.1

use core::marker::PhantomData;

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::{HashOutTarget, RichField};
use crate::hash::poseidon2::config::{
    EXTERNAL_CONSTANTS_8, INTERNAL_CONSTANTS_8, MATRIX_DIAG_8_U64, OUT_8, ROUNDS_F_8,
    ROUNDS_F_HALF_8, ROUNDS_P_8, WIDTH_8,
};
use crate::hash::poseidon2::hash::Poseidon2;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::{BoolTarget, Target};
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

#[derive(Debug, Default)]
pub struct Poseidon2Gate8<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Poseidon2Gate8<F, D> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    pub fn wire_input(i: usize) -> usize {
        i
    }

    pub fn wire_output(i: usize) -> usize {
        WIDTH_8 + i
    }

    pub const WIRE_SWAP: usize = 2 * WIDTH_8;
    const START_DELTA: usize = 2 * WIDTH_8 + 1;

    fn wire_delta(i: usize) -> usize {
        debug_assert!(i < 4);
        Self::START_DELTA + i
    }

    const START_ROUND_F_BEGIN: usize = Self::START_DELTA + 4;

    fn wire_full_sbox_0(round: usize, i: usize) -> usize {
        debug_assert!(round != 0);
        debug_assert!(round < ROUNDS_F_HALF_8);
        debug_assert!(i < WIDTH_8);
        Self::START_ROUND_F_BEGIN + WIDTH_8 * (round - 1) + i
    }

    const START_PARTIAL: usize = Self::START_ROUND_F_BEGIN + WIDTH_8 * (ROUNDS_F_HALF_8 - 1);

    const fn wire_partial_sbox(round: usize) -> usize {
        debug_assert!(round < ROUNDS_P_8);
        Self::START_PARTIAL + round
    }

    const START_ROUND_F_END: usize = Self::START_PARTIAL + ROUNDS_P_8;

    const fn wire_full_sbox_1(round: usize, i: usize) -> usize {
        debug_assert!(round < ROUNDS_F_HALF_8);
        debug_assert!(i < WIDTH_8);
        Self::START_ROUND_F_END + WIDTH_8 * round + i
    }

    const fn end() -> usize {
        Self::START_ROUND_F_END + WIDTH_8 * ROUNDS_F_HALF_8
    }
}

fn sbox_p<P: Field>(x: P) -> P {
    let x2 = x.square();
    let x4 = x2.square();
    x * x2 * x4
}

fn sbox_8<P: Field>(state: &mut [P; WIDTH_8]) {
    for v in state.iter_mut() {
        *v = sbox_p(*v);
    }
}

fn add_rc_8<P: Field>(state: &mut [P; WIDTH_8], round: usize) {
    for i in 0..WIDTH_8 {
        state[i] += P::from_canonical_u64(EXTERNAL_CONSTANTS_8[round][i]);
    }
}

fn external_linear_layer_hl_8<P: Field>(state: &mut [P; WIDTH_8]) {
    for i in (0..WIDTH_8).step_by(4) {
        let t0 = state[i] + state[i + 1];
        let t1 = state[i + 2] + state[i + 3];
        let t2 = state[i + 1] + state[i + 1] + t1;
        let t3 = state[i + 3] + state[i + 3] + t0;
        let t4 = t1 + t1 + t1 + t1 + t3;
        let t5 = t0 + t0 + t0 + t0 + t2;
        let t6 = t3 + t5;
        let t7 = t2 + t4;

        state[i] = t6;
        state[i + 1] = t5;
        state[i + 2] = t7;
        state[i + 3] = t4;
    }

    let sums = [
        state[0] + state[4],
        state[1] + state[5],
        state[2] + state[6],
        state[3] + state[7],
    ];
    for i in 0..WIDTH_8 {
        state[i] += sums[i % 4];
    }
}

fn external_linear_layer_hl_8_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    state: &mut [ExtensionTarget<D>; WIDTH_8],
) {
    for i in (0..WIDTH_8).step_by(4) {
        let t0 = builder.add_extension(state[i], state[i + 1]);
        let t1 = builder.add_extension(state[i + 2], state[i + 3]);

        let s1_twice = builder.add_extension(state[i + 1], state[i + 1]);
        let t2 = builder.add_extension(s1_twice, t1);

        let s3_twice = builder.add_extension(state[i + 3], state[i + 3]);
        let t3 = builder.add_extension(s3_twice, t0);

        let t1_2 = builder.add_extension(t1, t1);
        let t1_4 = builder.add_extension(t1_2, t1_2);
        let t4 = builder.add_extension(t1_4, t3);

        let t0_2 = builder.add_extension(t0, t0);
        let t0_4 = builder.add_extension(t0_2, t0_2);
        let t5 = builder.add_extension(t0_4, t2);

        let t6 = builder.add_extension(t3, t5);
        let t7 = builder.add_extension(t2, t4);

        state[i] = t6;
        state[i + 1] = t5;
        state[i + 2] = t7;
        state[i + 3] = t4;
    }

    let sums = [
        builder.add_extension(state[0], state[4]),
        builder.add_extension(state[1], state[5]),
        builder.add_extension(state[2], state[6]),
        builder.add_extension(state[3], state[7]),
    ];

    for i in 0..WIDTH_8 {
        state[i] = builder.add_extension(state[i], sums[i % 4]);
    }
}

fn internal_linear_layer_8<P: Field>(state: &mut [P; WIDTH_8]) {
    let sum = state.iter().copied().sum::<P>();
    for i in 0..WIDTH_8 {
        state[i] = sum.multiply_accumulate(state[i], P::from_canonical_u64(MATRIX_DIAG_8_U64[i]));
    }
}

fn internal_linear_layer_8_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    state: &mut [ExtensionTarget<D>; WIDTH_8],
) {
    let sum = state
        .iter()
        .copied()
        .reduce(|acc, t| builder.add_extension(acc, t))
        .unwrap();
    for i in 0..WIDTH_8 {
        state[i] = builder.mul_const_add_extension(
            F::from_canonical_u64(MATRIX_DIAG_8_U64[i]),
            state[i],
            sum,
        );
    }
}

fn sbox_p_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    x: ExtensionTarget<D>,
) -> ExtensionTarget<D> {
    let x2 = builder.mul_extension(x, x);
    let x4 = builder.mul_extension(x2, x2);
    let x3 = builder.mul_extension(x2, x);
    builder.mul_extension(x3, x4)
}

fn sbox_8_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    state: &mut [ExtensionTarget<D>; WIDTH_8],
) {
    for v in state.iter_mut() {
        *v = sbox_p_circuit(builder, *v);
    }
}

fn add_rc_8_circuit<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    state: &mut [ExtensionTarget<D>; WIDTH_8],
    round: usize,
) {
    for i in 0..WIDTH_8 {
        let rc = builder.constant_extension(F::Extension::from_canonical_u64(
            EXTERNAL_CONSTANTS_8[round][i],
        ));
        state[i] = builder.add_extension(state[i], rc);
    }
}

pub fn poseidon2_compress_8_to_4_swapped<
    F: RichField + Extendable<D> + Poseidon2,
    const D: usize,
>(
    builder: &mut CircuitBuilder<F, D>,
    left: HashOutTarget,
    right: HashOutTarget,
    swap: BoolTarget,
) -> HashOutTarget {
    let gate = builder.add_gate(Poseidon2Gate8::<F, D>::new(), vec![]);
    let swap_wire = Target::wire(gate, Poseidon2Gate8::<F, D>::WIRE_SWAP);
    builder.connect(swap.target, swap_wire);

    for i in 0..OUT_8 {
        builder.connect(
            left.elements[i],
            Target::wire(gate, Poseidon2Gate8::<F, D>::wire_input(i)),
        );
        builder.connect(
            right.elements[i],
            Target::wire(gate, Poseidon2Gate8::<F, D>::wire_input(i + OUT_8)),
        );
    }

    HashOutTarget {
        elements: core::array::from_fn(|i| {
            Target::wire(gate, Poseidon2Gate8::<F, D>::wire_output(i))
        }),
    }
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> Gate<F, D> for Poseidon2Gate8<F, D> {
    fn id(&self) -> String {
        format!("{:?}<WIDTH={}>", self, WIDTH_8)
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Self::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(swap * (swap - F::Extension::ONE));

        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            constraints.push(swap * (input_rhs - input_lhs) - delta_i);
        }

        let mut state = [F::Extension::ZERO; WIDTH_8];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }

        external_linear_layer_hl_8(&mut state);

        for r in 0..ROUNDS_F_HALF_8 {
            add_rc_8(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH_8 {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            sbox_8(&mut state);
            external_linear_layer_hl_8(&mut state);
        }

        for (r, &rc) in INTERNAL_CONSTANTS_8.iter().enumerate() {
            state[0] += F::Extension::from_canonical_u64(rc);
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = sbox_p(state[0]);
            internal_linear_layer_8(&mut state);
        }

        for r in ROUNDS_F_HALF_8..ROUNDS_F_8 {
            add_rc_8(&mut state, r);
            for i in 0..WIDTH_8 {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF_8, i)];
                constraints.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            sbox_8(&mut state);
            external_linear_layer_hl_8(&mut state);
        }

        for i in 0..WIDTH_8 {
            constraints.push(state[i] - vars.local_wires[Self::wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * swap.sub_one());

        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        let mut state = [F::ZERO; WIDTH_8];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }

        external_linear_layer_hl_8(&mut state);

        for r in 0..ROUNDS_F_HALF_8 {
            add_rc_8(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH_8 {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }
            sbox_8(&mut state);
            external_linear_layer_hl_8(&mut state);
        }

        for (r, &rc) in INTERNAL_CONSTANTS_8.iter().enumerate() {
            state[0] += F::from_canonical_u64(rc);
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = sbox_in;
            state[0] = sbox_p(state[0]);
            internal_linear_layer_8(&mut state);
        }

        for r in ROUNDS_F_HALF_8..ROUNDS_F_8 {
            add_rc_8(&mut state, r);
            for i in 0..WIDTH_8 {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF_8, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }
            sbox_8(&mut state);
            external_linear_layer_hl_8(&mut state);
        }

        for i in 0..WIDTH_8 {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(builder.mul_sub_extension(swap, swap, swap));

        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let diff = builder.sub_extension(input_rhs, input_lhs);
            constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
        }

        let mut state = [builder.zero_extension(); WIDTH_8];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = builder.add_extension(input_lhs, delta_i);
            state[i + 4] = builder.sub_extension(input_rhs, delta_i);
        }

        external_linear_layer_hl_8_circuit(builder, &mut state);

        for r in 0..ROUNDS_F_HALF_8 {
            add_rc_8_circuit(builder, &mut state, r);
            if r != 0 {
                for i in 0..WIDTH_8 {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(builder.sub_extension(state[i], sbox_in));
                    state[i] = sbox_in;
                }
            }
            sbox_8_circuit(builder, &mut state);
            external_linear_layer_hl_8_circuit(builder, &mut state);
        }

        for (r, &rc) in INTERNAL_CONSTANTS_8.iter().enumerate() {
            let round_constant = builder.constant_extension(F::Extension::from_canonical_u64(rc));
            state[0] = builder.add_extension(state[0], round_constant);

            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r)];
            constraints.push(builder.sub_extension(state[0], sbox_in));
            state[0] = sbox_in;
            state[0] = sbox_p_circuit(builder, state[0]);
            internal_linear_layer_8_circuit(builder, &mut state);
        }

        for r in ROUNDS_F_HALF_8..ROUNDS_F_8 {
            add_rc_8_circuit(builder, &mut state, r);
            for i in 0..WIDTH_8 {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r - ROUNDS_F_HALF_8, i)];
                constraints.push(builder.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }
            sbox_8_circuit(builder, &mut state);
            external_linear_layer_hl_8_circuit(builder, &mut state);
        }

        for i in 0..WIDTH_8 {
            constraints
                .push(builder.sub_extension(state[i], vars.local_wires[Self::wire_output(i)]));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        vec![WitnessGeneratorRef::new(
            Poseidon2Generator8::<F, D> {
                row,
                _phantom: PhantomData,
            }
            .adapter(),
        )]
    }

    fn num_wires(&self) -> usize {
        Self::end()
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        7
    }

    fn num_constraints(&self) -> usize {
        WIDTH_8 * (ROUNDS_F_8 - 1) + ROUNDS_P_8 + WIDTH_8 + 1 + 4
    }
}

#[derive(Debug, Default)]
pub struct Poseidon2Generator8<F: RichField + Extendable<D> + Poseidon2, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> SimpleGenerator<F, D>
    for Poseidon2Generator8<F, D>
{
    fn id(&self) -> String {
        "Poseidon2Generator8".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..WIDTH_8)
            .map(|i| Poseidon2Gate8::<F, D>::wire_input(i))
            .chain(Some(Poseidon2Gate8::<F, D>::WIRE_SWAP))
            .map(|column| Target::wire(self.row, column))
            .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let local_wire = |column| Wire {
            row: self.row,
            column,
        };

        let mut state = (0..WIDTH_8)
            .map(|i| witness.get_wire(local_wire(Poseidon2Gate8::<F, D>::wire_input(i))))
            .collect::<Vec<_>>();

        let swap_value = witness.get_wire(local_wire(Poseidon2Gate8::<F, D>::WIRE_SWAP));
        debug_assert!(swap_value == F::ZERO || swap_value == F::ONE);

        for i in 0..4 {
            let delta_i = swap_value * (state[i + 4] - state[i]);
            out_buffer.set_wire(local_wire(Poseidon2Gate8::<F, D>::wire_delta(i)), delta_i)?;
            state[i] += delta_i;
            state[i + 4] -= delta_i;
        }

        let mut state: [F; WIDTH_8] = state.try_into().unwrap();
        external_linear_layer_hl_8(&mut state);

        for r in 0..ROUNDS_F_HALF_8 {
            add_rc_8(&mut state, r);
            if r != 0 {
                for i in 0..WIDTH_8 {
                    out_buffer.set_wire(
                        local_wire(Poseidon2Gate8::<F, D>::wire_full_sbox_0(r, i)),
                        state[i],
                    )?;
                }
            }
            sbox_8(&mut state);
            external_linear_layer_hl_8(&mut state);
        }

        for (r, &rc) in INTERNAL_CONSTANTS_8.iter().enumerate() {
            state[0] += F::from_canonical_u64(rc);
            out_buffer.set_wire(
                local_wire(Poseidon2Gate8::<F, D>::wire_partial_sbox(r)),
                state[0],
            )?;
            state[0] = sbox_p(state[0]);
            internal_linear_layer_8(&mut state);
        }

        for r in ROUNDS_F_HALF_8..ROUNDS_F_8 {
            add_rc_8(&mut state, r);
            for i in 0..WIDTH_8 {
                out_buffer.set_wire(
                    local_wire(Poseidon2Gate8::<F, D>::wire_full_sbox_1(
                        r - ROUNDS_F_HALF_8,
                        i,
                    )),
                    state[i],
                )?;
            }
            sbox_8(&mut state);
            external_linear_layer_hl_8(&mut state);
        }

        for i in 0..WIDTH_8 {
            out_buffer.set_wire(local_wire(Poseidon2Gate8::<F, D>::wire_output(i)), state[i])?;
        }

        Ok(())
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        Ok(Self {
            row,
            _phantom: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;

    use super::*;
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::hash::poseidon2::hash::Poseidon2Hash;
    use crate::iop::generator::generate_partial_witness;
    use crate::iop::witness::PartialWitness;
    use crate::plonk::circuit_data::CircuitConfig;
    use crate::plonk::config::{GenericConfig, Poseidon2GoldilocksConfig};

    #[test]
    fn generated_output_matches_hash_8_to_4() {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig {
            num_wires: 128,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::new(config);
        type Gate = Poseidon2Gate8<F, D>;
        let gate = Gate::new();
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build_prover::<C>();

        let permutation_inputs = (0..WIDTH_8)
            .map(F::from_canonical_usize)
            .collect::<Vec<_>>();

        let mut inputs = PartialWitness::new();
        inputs
            .set_wire(
                Wire {
                    row,
                    column: Gate::WIRE_SWAP,
                },
                F::ZERO,
            )
            .unwrap();
        for i in 0..WIDTH_8 {
            inputs
                .set_wire(
                    Wire {
                        row,
                        column: Gate::wire_input(i),
                    },
                    permutation_inputs[i],
                )
                .unwrap();
        }

        let witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();

        let expected_outputs = Poseidon2Hash::hash_8_to_4(permutation_inputs.try_into().unwrap());
        for i in 0..OUT_8 {
            let out = witness.get_wire(Wire {
                row,
                column: Gate::wire_output(i),
            });
            assert_eq!(out, expected_outputs.elements[i]);
        }
    }

    #[test]
    fn generated_output_matches_hash_8_to_4_swap_true() {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig {
            num_wires: 128,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::new(config);
        type Gate = Poseidon2Gate8<F, D>;
        let gate = Gate::new();
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build_prover::<C>();

        let permutation_inputs = (0..WIDTH_8)
            .map(|x| F::from_canonical_usize(100 + x))
            .collect::<Vec<_>>();

        let mut inputs = PartialWitness::new();
        inputs
            .set_wire(
                Wire {
                    row,
                    column: Gate::WIRE_SWAP,
                },
                F::ONE,
            )
            .unwrap();
        for i in 0..WIDTH_8 {
            inputs
                .set_wire(
                    Wire {
                        row,
                        column: Gate::wire_input(i),
                    },
                    permutation_inputs[i],
                )
                .unwrap();
        }

        let witness =
            generate_partial_witness(inputs, &circuit.prover_only, &circuit.common).unwrap();

        let swapped = [
            permutation_inputs[4],
            permutation_inputs[5],
            permutation_inputs[6],
            permutation_inputs[7],
            permutation_inputs[0],
            permutation_inputs[1],
            permutation_inputs[2],
            permutation_inputs[3],
        ];
        let expected_outputs = Poseidon2Hash::hash_8_to_4(swapped);
        for i in 0..OUT_8 {
            let out = witness.get_wire(Wire {
                row,
                column: Gate::wire_output(i),
            });
            assert_eq!(out, expected_outputs.elements[i]);
        }
    }

    #[test]
    fn randomized_matches_hash_8_to_4() {
        use rand::{thread_rng, Rng};

        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig {
            num_wires: 128,
            ..CircuitConfig::standard_recursion_config()
        };
        let mut builder = CircuitBuilder::new(config);
        type Gate = Poseidon2Gate8<F, D>;
        let gate = Gate::new();
        let row = builder.add_gate(gate, vec![]);
        let circuit = builder.build_prover::<C>();

        let mut rng = thread_rng();
        for _ in 0..200 {
            let input = core::array::from_fn(|_| F::from_noncanonical_u64(rng.gen()));

            for swap in [F::ZERO, F::ONE] {
                let mut pw = PartialWitness::new();
                pw.set_wire(
                    Wire {
                        row,
                        column: Gate::WIRE_SWAP,
                    },
                    swap,
                )
                .unwrap();
                for (i, v) in input.iter().enumerate() {
                    pw.set_wire(
                        Wire {
                            row,
                            column: Gate::wire_input(i),
                        },
                        *v,
                    )
                    .unwrap();
                }

                let witness =
                    generate_partial_witness(pw, &circuit.prover_only, &circuit.common).unwrap();

                let expected_in = if swap == F::ONE {
                    [
                        input[4], input[5], input[6], input[7], input[0], input[1], input[2],
                        input[3],
                    ]
                } else {
                    input
                };
                let expected = Poseidon2Hash::hash_8_to_4(expected_in);
                for i in 0..OUT_8 {
                    let out = witness.get_wire(Wire {
                        row,
                        column: Gate::wire_output(i),
                    });
                    assert_eq!(out, expected.elements[i]);
                }
            }
        }
    }

    #[test]
    fn low_degree() {
        type F = GoldilocksField;
        let gate = Poseidon2Gate8::<F, 4>::new();
        test_low_degree(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = Poseidon2GoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = Poseidon2Gate8::<F, D>::new();
        test_eval_fns::<F, C, _, D>(gate)
    }
}
