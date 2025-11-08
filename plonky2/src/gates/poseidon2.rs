#[cfg(not(feature = "std"))]
use alloc::{
    format,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use core::marker::PhantomData;

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::types::Field;
use crate::gates::gate::Gate;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::hash::poseidon2::{self, P2Permutation, Poseidon2, SPONGE_WIDTH};
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::wire::Wire;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::CommonCircuitData;
use crate::plonk::vars::{EvaluationTargets, EvaluationVars, EvaluationVarsBase};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// Evaluates a full Poseidon2 permutation with 12 state elements.
///
/// This also has some extra features to make it suitable for efficiently verifying Merkle proofs.
/// It has a flag which can be used to swap the first four inputs with the next four, for ordering
/// sibling digests.
#[derive(Debug, Default)]
pub struct Poseidon2Gate<F: RichField + Extendable<D>, const D: usize>(PhantomData<F>);

impl<F: RichField + Extendable<D>, const D: usize> Poseidon2Gate<F, D> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }

    /// The wire index for the `i`th input to the permutation.
    pub(crate) const fn wire_input(i: usize) -> usize {
        i
    }

    /// The wire index for the `i`th output to the permutation.
    pub(crate) const fn wire_output(i: usize) -> usize {
        SPONGE_WIDTH + i
    }

    /// If this is set to 1, the first four inputs will be swapped with the next four inputs. This
    /// is useful for ordering hashes in Merkle proofs. Otherwise, this should be set to 0.
    pub(crate) const WIRE_SWAP: usize = 2 * SPONGE_WIDTH;

    const START_DELTA: usize = 2 * SPONGE_WIDTH + 1;

    /// A wire which stores `swap * (input[i + 4] - input[i])`; used to compute the swapped inputs.
    const fn wire_delta(i: usize) -> usize {
        assert!(i < 4);
        Self::START_DELTA + i
    }

    const START_FULL_0: usize = Self::START_DELTA + 4;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the first set
    /// of full rounds.
    const fn wire_full_sbox_0(round: usize, i: usize) -> usize {
        debug_assert!(
            round != 0,
            "First round S-box inputs are not stored as wires"
        );
        debug_assert!(round < poseidon2::HALF_N_ROUNDS_F);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_0 + SPONGE_WIDTH * (round - 1) + i
    }

    const START_PARTIAL: usize =
        Self::START_FULL_0 + SPONGE_WIDTH * (poseidon2::HALF_N_ROUNDS_F - 1);

    /// A wire which stores the input of the S-box of the `round`-th round of the partial rounds.
    const fn wire_partial_sbox(round: usize) -> usize {
        debug_assert!(round < poseidon2::ROUNDS_P);
        Self::START_PARTIAL + round
    }

    const START_FULL_1: usize = Self::START_PARTIAL + poseidon2::ROUNDS_P;

    /// A wire which stores the input of the `i`-th S-box of the `round`-th round of the second set
    /// of full rounds.
    const fn wire_full_sbox_1(round: usize, i: usize) -> usize {
        debug_assert!(round < poseidon2::HALF_N_ROUNDS_F);
        debug_assert!(i < SPONGE_WIDTH);
        Self::START_FULL_1 + SPONGE_WIDTH * round + i
    }

    /// End of wire indices, exclusive.
    const fn end() -> usize {
        Self::START_FULL_1 + SPONGE_WIDTH * poseidon2::HALF_N_ROUNDS_F
    }

    /// Apply M_4 matrix to a 4-element slice using circuit builder
    fn apply_m_4_circuit(builder: &mut CircuitBuilder<F, D>, x: &mut [ExtensionTarget<D>]) {
        assert!(x.len() == 4);
        let t0 = builder.add_extension(x[0], x[1]);
        let t1 = builder.add_extension(x[2], x[3]);
        let t2_tmp = builder.add_extension(x[1], x[1]);
        let t2 = builder.add_extension(t2_tmp, t1);
        let t3_tmp = builder.add_extension(x[3], x[3]);
        let t3 = builder.add_extension(t3_tmp, t0);
        let t4_tmp1 = builder.add_extension(t1, t1);
        let t4_tmp2 = builder.add_extension(t4_tmp1, t4_tmp1);
        let t4 = builder.add_extension(t4_tmp2, t3);
        let t5_tmp1 = builder.add_extension(t0, t0);
        let t5_tmp2 = builder.add_extension(t5_tmp1, t5_tmp1);
        let t5 = builder.add_extension(t5_tmp2, t2);
        let t6 = builder.add_extension(t3, t5);
        let t7 = builder.add_extension(t2, t4);
        x[0] = t6;
        x[1] = t5;
        x[2] = t7;
        x[3] = t4;
    }

    /// Apply internal matrix multiplication using circuit builder
    fn matmul_internal_circuit(builder: &mut CircuitBuilder<F, D>, state: &mut [ExtensionTarget<D>; SPONGE_WIDTH]) {
        let mut sum = builder.zero_extension();
        for i in 0..SPONGE_WIDTH {
            sum = builder.add_extension(sum, state[i]);
        }

        for i in 0..SPONGE_WIDTH {
            let diag = F::Extension::from_canonical_u64(poseidon2::MATRIX_DIAG_12_GOLDILOCKS[i]);
            let diag_target = builder.constant_extension(diag);
            state[i] = builder.mul_extension(state[i], diag_target);
            state[i] = builder.add_extension(state[i], sum);
        }
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for Poseidon2Gate<F, D> {
    fn id(&self) -> String {
        format!("{self:?}<WIDTH={SPONGE_WIDTH}>")
    }

    fn serialize(
        &self,
        _dst: &mut Vec<u8>,
        _common_data: &CommonCircuitData<F, D>,
    ) -> IoResult<()> {
        Ok(())
    }

    fn deserialize(_src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        Ok(Poseidon2Gate::new())
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(swap * (swap - F::Extension::ONE));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            constraints.push(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::Extension::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // Apply the initial linear layer
        poseidon2::apply_m_4(&mut state[0..4]);
        poseidon2::apply_m_4(&mut state[4..8]);
        poseidon2::apply_m_4(&mut state[8..12]);

        let sums: [F::Extension; 4] = core::array::from_fn(|k| {
            (0..SPONGE_WIDTH)
                .step_by(4)
                .map(|j| state[j + k].clone())
                .sum::<F::Extension>()
        });

        for i in 0..SPONGE_WIDTH {
            state[i] += sums[i % 4].clone();
        }

        let mut round_ctr = 0;
        let rounds_f_beginning = poseidon2::HALF_N_ROUNDS_F;

        // First set of full rounds.
        for r in 0..rounds_f_beginning {
            // Add round constants
            for i in 0..SPONGE_WIDTH {
                state[i] += F::Extension::from_canonical_u64(poseidon2::RC12[round_ctr][i]);
            }

            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }

            // S-box layer
            for i in 0..SPONGE_WIDTH {
                let x2 = state[i] * state[i];
                let x4 = x2 * x2;
                let x3 = x2 * state[i];
                state[i] = x3 * x4;
            }

            // Linear layer
            poseidon2::apply_m_4(&mut state[0..4]);
            poseidon2::apply_m_4(&mut state[4..8]);
            poseidon2::apply_m_4(&mut state[8..12]);

            let sums: [F::Extension; 4] = core::array::from_fn(|k| {
                (0..SPONGE_WIDTH)
                    .step_by(4)
                    .map(|j| state[j + k].clone())
                    .sum::<F::Extension>()
            });

            for i in 0..SPONGE_WIDTH {
                state[i] += sums[i % 4].clone();
            }

            round_ctr += 1;
        }

        // Partial rounds.
        let p_end = rounds_f_beginning + poseidon2::ROUNDS_P;
        for r in rounds_f_beginning..p_end {
            state[0] += F::Extension::from_canonical_u64(poseidon2::RC12[round_ctr][0]);
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r - rounds_f_beginning)];
            constraints.push(state[0] - sbox_in);
            state[0] = sbox_in;

            let x2 = state[0] * state[0];
            let x4 = x2 * x2;
            let x3 = x2 * state[0];
            state[0] = x3 * x4;

            poseidon2::matmul_internal(&mut state, poseidon2::MATRIX_DIAG_12_GOLDILOCKS);
            round_ctr += 1;
        }

        // Second set of full rounds.
        for r in 0..poseidon2::HALF_N_ROUNDS_F {
            // Add round constants
            for i in 0..SPONGE_WIDTH {
                state[i] += F::Extension::from_canonical_u64(poseidon2::RC12[round_ctr][i]);
            }

            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(state[i] - sbox_in);
                state[i] = sbox_in;
            }

            // S-box layer
            for i in 0..SPONGE_WIDTH {
                let x2 = state[i] * state[i];
                let x4 = x2 * x2;
                let x3 = x2 * state[i];
                state[i] = x3 * x4;
            }

            // Linear layer
            poseidon2::apply_m_4(&mut state[0..4]);
            poseidon2::apply_m_4(&mut state[4..8]);
            poseidon2::apply_m_4(&mut state[8..12]);

            let sums: [F::Extension; 4] = core::array::from_fn(|k| {
                (0..SPONGE_WIDTH)
                    .step_by(4)
                    .map(|j| state[j + k].clone())
                    .sum::<F::Extension>()
            });

            for i in 0..SPONGE_WIDTH {
                state[i] += sums[i % 4].clone();
            }

            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            constraints.push(state[i] - vars.local_wires[Self::wire_output(i)]);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        vars: EvaluationVarsBase<F>,
        mut yield_constr: StridedConstraintConsumer<F>,
    ) {
        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        yield_constr.one(swap * swap.sub_one());

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            yield_constr.one(swap * (input_rhs - input_lhs) - delta_i);
        }

        // Compute the possibly-swapped input layer.
        let mut state = [F::ZERO; SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = Self::wire_input(i);
            let input_rhs = Self::wire_input(i + 4);
            state[i] = vars.local_wires[input_lhs] + delta_i;
            state[i + 4] = vars.local_wires[input_rhs] - delta_i;
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // Apply the initial linear layer
        poseidon2::apply_m_4(&mut state[0..4]);
        poseidon2::apply_m_4(&mut state[4..8]);
        poseidon2::apply_m_4(&mut state[8..12]);

        let sums: [F; 4] = core::array::from_fn(|k| {
            (0..SPONGE_WIDTH)
                .step_by(4)
                .map(|j| state[j + k])
                .sum::<F>()
        });

        for i in 0..SPONGE_WIDTH {
            state[i] += sums[i % 4];
        }

        let mut round_ctr = 0;
        let rounds_f_beginning = poseidon2::HALF_N_ROUNDS_F;

        // First set of full rounds.
        for r in 0..rounds_f_beginning {
            // Add round constants
            for i in 0..SPONGE_WIDTH {
                state[i] += F::from_canonical_u64(poseidon2::RC12[round_ctr][i]);
            }

            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    yield_constr.one(state[i] - sbox_in);
                    state[i] = sbox_in;
                }
            }

            // S-box layer
            for i in 0..SPONGE_WIDTH {
                let x2 = state[i] * state[i];
                let x4 = x2 * x2;
                let x3 = x2 * state[i];
                state[i] = x3 * x4;
            }

            // Linear layer
            poseidon2::apply_m_4(&mut state[0..4]);
            poseidon2::apply_m_4(&mut state[4..8]);
            poseidon2::apply_m_4(&mut state[8..12]);

            let sums: [F; 4] = core::array::from_fn(|k| {
                (0..SPONGE_WIDTH)
                    .step_by(4)
                    .map(|j| state[j + k])
                    .sum::<F>()
            });

            for i in 0..SPONGE_WIDTH {
                state[i] += sums[i % 4];
            }

            round_ctr += 1;
        }

        // Partial rounds.
        let p_end = rounds_f_beginning + poseidon2::ROUNDS_P;
        for r in rounds_f_beginning..p_end {
            state[0] += F::from_canonical_u64(poseidon2::RC12[round_ctr][0]);
            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r - rounds_f_beginning)];
            yield_constr.one(state[0] - sbox_in);
            state[0] = sbox_in;

            let x2 = state[0] * state[0];
            let x4 = x2 * x2;
            let x3 = x2 * state[0];
            state[0] = x3 * x4;

            poseidon2::matmul_internal(&mut state, poseidon2::MATRIX_DIAG_12_GOLDILOCKS);
            round_ctr += 1;
        }

        // Second set of full rounds.
        for r in 0..poseidon2::HALF_N_ROUNDS_F {
            // Add round constants
            for i in 0..SPONGE_WIDTH {
                state[i] += F::from_canonical_u64(poseidon2::RC12[round_ctr][i]);
            }

            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                yield_constr.one(state[i] - sbox_in);
                state[i] = sbox_in;
            }

            // S-box layer
            for i in 0..SPONGE_WIDTH {
                let x2 = state[i] * state[i];
                let x4 = x2 * x2;
                let x3 = x2 * state[i];
                state[i] = x3 * x4;
            }

            // Linear layer
            poseidon2::apply_m_4(&mut state[0..4]);
            poseidon2::apply_m_4(&mut state[4..8]);
            poseidon2::apply_m_4(&mut state[8..12]);

            let sums: [F; 4] = core::array::from_fn(|k| {
                (0..SPONGE_WIDTH)
                    .step_by(4)
                    .map(|j| state[j + k])
                    .sum::<F>()
            });

            for i in 0..SPONGE_WIDTH {
                state[i] += sums[i % 4];
            }

            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            yield_constr.one(state[i] - vars.local_wires[Self::wire_output(i)]);
        }
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let mut constraints = Vec::with_capacity(self.num_constraints());

        // Assert that `swap` is binary.
        let swap = vars.local_wires[Self::WIRE_SWAP];
        constraints.push(builder.mul_sub_extension(swap, swap, swap));

        // Assert that each delta wire is set properly: `delta_i = swap * (rhs - lhs)`.
        for i in 0..4 {
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let diff = builder.sub_extension(input_rhs, input_lhs);
            constraints.push(builder.mul_sub_extension(swap, diff, delta_i));
        }

        // Compute the possibly-swapped input layer.
        let mut state = [builder.zero_extension(); SPONGE_WIDTH];
        for i in 0..4 {
            let delta_i = vars.local_wires[Self::wire_delta(i)];
            let input_lhs = vars.local_wires[Self::wire_input(i)];
            let input_rhs = vars.local_wires[Self::wire_input(i + 4)];
            state[i] = builder.add_extension(input_lhs, delta_i);
            state[i + 4] = builder.sub_extension(input_rhs, delta_i);
        }
        for i in 8..SPONGE_WIDTH {
            state[i] = vars.local_wires[Self::wire_input(i)];
        }

        // Apply initial linear layer (M_E matrix)
        Self::apply_m_4_circuit(builder, &mut state[0..4]);
        Self::apply_m_4_circuit(builder, &mut state[4..8]);
        Self::apply_m_4_circuit(builder, &mut state[8..12]);

        let sums: [ExtensionTarget<D>; 4] = core::array::from_fn(|k| {
            let mut sum = builder.zero_extension();
            for j in (0..SPONGE_WIDTH).step_by(4) {
                sum = builder.add_extension(sum, state[j + k]);
            }
            sum
        });

        for i in 0..SPONGE_WIDTH {
            state[i] = builder.add_extension(state[i], sums[i % 4]);
        }

        let mut round_ctr = 0;
        let rounds_f_beginning = poseidon2::HALF_N_ROUNDS_F;

        // First set of full rounds
        for r in 0..rounds_f_beginning {
            // Add round constants
            for i in 0..SPONGE_WIDTH {
                let rc = F::Extension::from_canonical_u64(poseidon2::RC12[round_ctr][i]);
                let rc_target = builder.constant_extension(rc);
                state[i] = builder.add_extension(state[i], rc_target);
            }

            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    let sbox_in = vars.local_wires[Self::wire_full_sbox_0(r, i)];
                    constraints.push(builder.sub_extension(state[i], sbox_in));
                    state[i] = sbox_in;
                }
            }

            // S-box layer (x^7)
            for i in 0..SPONGE_WIDTH {
                let x2 = builder.mul_extension(state[i], state[i]);
                let x4 = builder.mul_extension(x2, x2);
                let x3 = builder.mul_extension(x2, state[i]);
                state[i] = builder.mul_extension(x3, x4);
            }

            // Linear layer
            Self::apply_m_4_circuit(builder, &mut state[0..4]);
            Self::apply_m_4_circuit(builder, &mut state[4..8]);
            Self::apply_m_4_circuit(builder, &mut state[8..12]);

            let sums: [ExtensionTarget<D>; 4] = core::array::from_fn(|k| {
                let mut sum = builder.zero_extension();
                for j in (0..SPONGE_WIDTH).step_by(4) {
                    sum = builder.add_extension(sum, state[j + k]);
                }
                sum
            });

            for i in 0..SPONGE_WIDTH {
                state[i] = builder.add_extension(state[i], sums[i % 4]);
            }

            round_ctr += 1;
        }

        // Partial rounds
        let p_end = rounds_f_beginning + poseidon2::ROUNDS_P;
        for r in rounds_f_beginning..p_end {
            let rc = F::Extension::from_canonical_u64(poseidon2::RC12[round_ctr][0]);
            let rc_target = builder.constant_extension(rc);
            state[0] = builder.add_extension(state[0], rc_target);

            let sbox_in = vars.local_wires[Self::wire_partial_sbox(r - rounds_f_beginning)];
            constraints.push(builder.sub_extension(state[0], sbox_in));
            state[0] = sbox_in;

            // S-box on first element
            let x2 = builder.mul_extension(state[0], state[0]);
            let x4 = builder.mul_extension(x2, x2);
            let x3 = builder.mul_extension(x2, state[0]);
            state[0] = builder.mul_extension(x3, x4);

            // Internal matrix multiplication
            Self::matmul_internal_circuit(builder, &mut state);
            round_ctr += 1;
        }

        // Second set of full rounds
        for r in 0..poseidon2::HALF_N_ROUNDS_F {
            // Add round constants
            for i in 0..SPONGE_WIDTH {
                let rc = F::Extension::from_canonical_u64(poseidon2::RC12[round_ctr][i]);
                let rc_target = builder.constant_extension(rc);
                state[i] = builder.add_extension(state[i], rc_target);
            }

            for i in 0..SPONGE_WIDTH {
                let sbox_in = vars.local_wires[Self::wire_full_sbox_1(r, i)];
                constraints.push(builder.sub_extension(state[i], sbox_in));
                state[i] = sbox_in;
            }

            // S-box layer
            for i in 0..SPONGE_WIDTH {
                let x2 = builder.mul_extension(state[i], state[i]);
                let x4 = builder.mul_extension(x2, x2);
                let x3 = builder.mul_extension(x2, state[i]);
                state[i] = builder.mul_extension(x3, x4);
            }

            // Linear layer
            Self::apply_m_4_circuit(builder, &mut state[0..4]);
            Self::apply_m_4_circuit(builder, &mut state[4..8]);
            Self::apply_m_4_circuit(builder, &mut state[8..12]);

            let sums: [ExtensionTarget<D>; 4] = core::array::from_fn(|k| {
                let mut sum = builder.zero_extension();
                for j in (0..SPONGE_WIDTH).step_by(4) {
                    sum = builder.add_extension(sum, state[j + k]);
                }
                sum
            });

            for i in 0..SPONGE_WIDTH {
                state[i] = builder.add_extension(state[i], sums[i % 4]);
            }

            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            constraints.push(builder.sub_extension(state[i], vars.local_wires[Self::wire_output(i)]));
        }

        constraints
    }

    fn generators(&self, row: usize, _local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        let gen = Poseidon2Generator::<F, D> {
            row,
            _phantom: PhantomData,
        };
        vec![WitnessGeneratorRef::new(gen.adapter())]
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
        let rounds_f_total = poseidon2::ROUNDS_F;
        SPONGE_WIDTH * (rounds_f_total - 1)
            + poseidon2::ROUNDS_P
            + SPONGE_WIDTH
            + 1
            + 4
    }
}

#[derive(Debug, Default)]
pub struct Poseidon2Generator<F: RichField + Extendable<D> + Poseidon2, const D: usize> {
    row: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D> + Poseidon2, const D: usize> SimpleGenerator<F, D>
    for Poseidon2Generator<F, D>
{
    fn id(&self) -> String {
        "Poseidon2Generator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        (0..SPONGE_WIDTH)
            .map(|i| Poseidon2Gate::<F, D>::wire_input(i))
            .chain(Some(Poseidon2Gate::<F, D>::WIRE_SWAP))
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

        let mut state = (0..SPONGE_WIDTH)
            .map(|i| witness.get_wire(local_wire(Poseidon2Gate::<F, D>::wire_input(i))))
            .collect::<Vec<_>>();

        let swap_value = witness.get_wire(local_wire(Poseidon2Gate::<F, D>::WIRE_SWAP));
        debug_assert!(swap_value == F::ZERO || swap_value == F::ONE);

        for i in 0..4 {
            let delta_i = swap_value * (state[i + 4] - state[i]);
            out_buffer.set_wire(local_wire(Poseidon2Gate::<F, D>::wire_delta(i)), delta_i)?;
        }

        if swap_value == F::ONE {
            for i in 0..4 {
                state.swap(i, 4 + i);
            }
        }

        let mut state: [F; SPONGE_WIDTH] = state.try_into().unwrap();

        // Apply the initial linear layer
        poseidon2::Poseidon2MEMatrix.permute_mut(&mut state);

        let rounds_f_beginning = poseidon2::HALF_N_ROUNDS_F;
        let mut round_ctr = 0;

        // First set of full rounds
        for r in 0..rounds_f_beginning {
            <F as Poseidon2>::add_rc(&mut state, &poseidon2::RC12[round_ctr]);
            if r != 0 {
                for i in 0..SPONGE_WIDTH {
                    out_buffer.set_wire(
                        local_wire(Poseidon2Gate::<F, D>::wire_full_sbox_0(r, i)),
                        state[i],
                    )?;
                }
            }
            <F as Poseidon2>::sbox(&mut state);
            poseidon2::Poseidon2MEMatrix.permute_mut(&mut state);
            round_ctr += 1;
        }

        // Partial rounds
        let p_end = rounds_f_beginning + poseidon2::ROUNDS_P;
        for r in rounds_f_beginning..p_end {
            state[0] += F::from_canonical_u64(poseidon2::RC12[round_ctr][0]);
            out_buffer.set_wire(
                local_wire(Poseidon2Gate::<F, D>::wire_partial_sbox(r - rounds_f_beginning)),
                state[0],
            )?;
            state[0] = <F as Poseidon2>::sbox_p(&state[0]);
            poseidon2::matmul_internal(&mut state, poseidon2::MATRIX_DIAG_12_GOLDILOCKS);
            round_ctr += 1;
        }

        // Second set of full rounds
        for r in 0..poseidon2::HALF_N_ROUNDS_F {
            <F as Poseidon2>::add_rc(&mut state, &poseidon2::RC12[round_ctr]);
            for i in 0..SPONGE_WIDTH {
                out_buffer.set_wire(
                    local_wire(Poseidon2Gate::<F, D>::wire_full_sbox_1(r, i)),
                    state[i],
                )?;
            }
            <F as Poseidon2>::sbox(&mut state);
            poseidon2::Poseidon2MEMatrix.permute_mut(&mut state);
            round_ctr += 1;
        }

        for i in 0..SPONGE_WIDTH {
            out_buffer.set_wire(local_wire(Poseidon2Gate::<F, D>::wire_output(i)), state[i])?
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
