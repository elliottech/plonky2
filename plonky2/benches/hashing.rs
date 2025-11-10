mod allocator;

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use plonky2::field::goldilocks_field::GoldilocksField;
use plonky2::field::types::Sample;
use plonky2::hash::hash_types::{BytesHash, HashOut, RichField};
use plonky2::hash::keccak::KeccakHash;
use plonky2::hash::poseidon::{Poseidon, SPONGE_WIDTH};
use plonky2::hash::poseidon2::{Poseidon2Hash, POSEIDON2_WIDTH};
use plonky2::plonk::config::Hasher;
use tynm::type_name;

pub(crate) fn bench_keccak<F: RichField>(c: &mut Criterion) {
    c.bench_function("keccak256", |b| {
        b.iter_batched(
            || (BytesHash::<32>::rand(), BytesHash::<32>::rand()),
            |(left, right)| <KeccakHash<32> as Hasher<F>>::two_to_one(left, right),
            BatchSize::SmallInput,
        )
    });
}

pub(crate) fn bench_poseidon<F: Poseidon>(c: &mut Criterion) {
    c.bench_function(
        &format!("poseidon<{}, {SPONGE_WIDTH}>", type_name::<F>()),
        |b| {
            b.iter_batched(
                || F::rand_array::<SPONGE_WIDTH>(),
                |state| F::poseidon(state),
                BatchSize::SmallInput,
            )
        },
    );
}

pub(crate) fn bench_poseidon2(c: &mut Criterion) {
    c.bench_function(
        &format!(
            "poseidon2<{}, {POSEIDON2_WIDTH}>",
            type_name::<GoldilocksField>()
        ),
        |b| {
            b.iter_batched(
                || {
                    let input: Vec<GoldilocksField> = (0..POSEIDON2_WIDTH)
                        .map(|_| GoldilocksField::rand())
                        .collect();
                    input
                },
                |input| Poseidon2Hash::hash_no_pad(&input),
                BatchSize::SmallInput,
            )
        },
    );
}

pub(crate) fn bench_poseidon2_two_to_one(c: &mut Criterion) {
    c.bench_function(
        &format!("poseidon2_two_to_one<{}>", type_name::<GoldilocksField>()),
        |b| {
            b.iter_batched(
                || {
                    let left = HashOut {
                        elements: [
                            GoldilocksField::rand(),
                            GoldilocksField::rand(),
                            GoldilocksField::rand(),
                            GoldilocksField::rand(),
                        ],
                    };
                    let right = HashOut {
                        elements: [
                            GoldilocksField::rand(),
                            GoldilocksField::rand(),
                            GoldilocksField::rand(),
                            GoldilocksField::rand(),
                        ],
                    };
                    (left, right)
                },
                |(left, right)| Poseidon2Hash::two_to_one(left, right),
                BatchSize::SmallInput,
            )
        },
    );
}

fn criterion_benchmark(c: &mut Criterion) {
    bench_poseidon::<GoldilocksField>(c);
    bench_poseidon2(c);
    bench_poseidon2_two_to_one(c);
    bench_keccak::<GoldilocksField>(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
