#![allow(clippy::all)]

use plonky2_field::goldilocks_field::GoldilocksField;

use super::config::*;
/// This is a test implementation of the Poseidon2 permutation without any gate or generator optimizations.
use crate::field::types::Field;
use crate::iop::target::Target;
use crate::plonk::circuit_builder::CircuitBuilder;

type F = GoldilocksField;
type Builder = CircuitBuilder<F, 2>;

pub fn permute_swapped(builder: &mut Builder, inputs: &[Target; WIDTH]) -> [Target; WIDTH] {
    let mut state: [Target; 12] = inputs.clone();
    external_permute_mut(builder, &mut state);

    // The first half of the external rounds.
    for r in 0..ROUNDS_F_HALF {
        let external_constants = EXTERNAL_CONSTANTS[r];
        let external_constants: [F; WIDTH] = external_constants
            .iter()
            .map(|&x| F::from_canonical_u64(x))
            .collect::<Vec<F>>()
            .try_into()
            .unwrap();
        let external_constants: [Target; WIDTH] =
            builder.constants(&external_constants).try_into().unwrap();
        add_rc(builder, &mut state, &external_constants);
        sbox(builder, &mut state);
        external_permute_mut(builder, &mut state);
    }

    // The internal rounds.
    for r in 0..ROUNDS_P {
        let internal_constant = INTERNAL_CONSTANTS[r];
        let internal_constant = F::from_canonical_u64(internal_constant);
        let internal_constant = builder.constant(internal_constant);
        state[0] = builder.add(state[0], internal_constant);
        state[0] = sbox_p(builder, state[0]);
        internal_permute_mut(builder, &mut state);
    }

    // The second half of the external rounds.
    for r in ROUNDS_F_HALF..ROUNDS_F {
        let external_constants = EXTERNAL_CONSTANTS[r];
        let external_constants: [F; WIDTH] = external_constants
            .iter()
            .map(|&x| F::from_canonical_u64(x))
            .collect::<Vec<F>>()
            .try_into()
            .unwrap();
        let external_constants: [Target; WIDTH] =
            builder.constants(&external_constants).try_into().unwrap();
        add_rc(builder, &mut state, &external_constants);
        sbox(builder, &mut state);
        external_permute_mut(builder, &mut state);
    }

    state
}

fn external_permute_mut(builder: &mut Builder, state: &mut [Target; WIDTH]) {
    // First, we apply M_4 to each consecutive four elements of the state.
    // In Appendix B's terminology, this replaces each x_i with x_i'.
    for i in (0..WIDTH).step_by(4) {
        // Would be nice to find a better way to do this.
        let mut state_4 = [state[i], state[i + 1], state[i + 2], state[i + 3]];
        apply_mat4(builder, &mut state_4);
        state[i..i + 4].clone_from_slice(&state_4);
    }
    // Now, we apply the outer circulant matrix (to compute the y_i values).

    // We first precompute the four sums of every four elements.
    let sums: [Target; 4] = core::array::from_fn(|k| {
        (0..WIDTH)
            .step_by(4)
            .map(|j| state[j + k])
            .reduce(|acc, t| builder.add(acc, t))
            .unwrap()
    });

    // The formula for each y_i involves 2x_i' term and x_j' terms for each j that equals i mod 4.
    // In other words, we can add a single copy of x_i' to the appropriate one of our precomputed sums
    for i in 0..WIDTH {
        state[i] = builder.add(state[i], sums[i % 4]);
    }
}

fn add_rc(builder: &mut Builder, state: &mut [Target; WIDTH], external_constant: &[Target; WIDTH]) {
    state
        .iter_mut()
        .zip(external_constant)
        .for_each(|(a, b)| *a = builder.add(*a, *b));
}

fn sbox(builder: &mut Builder, state: &mut [Target; WIDTH]) {
    state.iter_mut().for_each(|a| *a = sbox_p(builder, *a));
}

fn sbox_p(builder: &mut Builder, a: Target) -> Target {
    let a2 = builder.mul(a, a);
    let a3 = builder.mul(a2, a);
    let a6 = builder.mul(a3, a3);

    builder.mul(a6, a)
}

// Multiply a 4-element vector x by:
// [ 2 3 1 1 ]
// [ 1 2 3 1 ]
// [ 1 1 2 3 ]
// [ 3 1 1 2 ].
// This is more efficient than the previous matrix.
fn apply_mat4(builder: &mut Builder, x: &mut [Target; 4]) {
    let two = builder.constant(F::from_canonical_u64(2));

    let t01 = builder.add(x[0], x[1]);
    let t23 = builder.add(x[2], x[3]);
    let t0123 = builder.add(t01, t23);
    let t01123 = builder.add(t0123, x[1]);
    let t01233 = builder.add(t0123, x[3]);
    // The order here is important. Need to overwrite x[0] and x[2] after x[1] and x[3].
    let dx0 = builder.mul(x[0], two);
    let dx2 = builder.mul(x[2], two);
    x[3] = builder.add(t01233, dx0); // 3*x[0] + x[1] + x[2] + 2*x[3]
    x[1] = builder.add(t01123, dx2); // x[0] + 2*x[1] + 3*x[2] + x[3]
    x[0] = builder.add(t01123, t01); // 2*x[0] + 3*x[1] + x[2] + x[3]
    x[2] = builder.add(t01233, t23); // x[0] + x[1] + 2*x[2] + 3*x[3]
}

/// Given a vector v compute the matrix vector product (1 + diag(v))state with 1 denoting the constant matrix of ones.
pub fn internal_permute_mut(builder: &mut Builder, state: &mut [Target; WIDTH]) {
    let sum = builder.add_many(state.iter().cloned());
    for i in 0..WIDTH {
        let constant = MATRIX_DIAG_12_U64[i];
        let constant = F::from_canonical_u64(constant);
        let constant = builder.constant(constant);
        state[i] = builder.mul(state[i], constant);
        state[i] = builder.add(state[i], sum);
    }
}