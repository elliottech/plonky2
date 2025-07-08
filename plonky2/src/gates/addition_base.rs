extern crate alloc;
use alloc::string::ToString;
#[cfg(not(feature = "std"))]
use alloc::{format, string::String, vec::Vec};

use anyhow::Result;

use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::{GeneratedValues, SimpleGenerator, WitnessGeneratorRef};
use crate::iop::target::Target;
use crate::iop::witness::{PartitionWitness, Witness, WitnessWrite};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::circuit_data::{CircuitConfig, CommonCircuitData};
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};
use crate::util::serialization::{Buffer, IoResult, Read, Write};

/// A gate specialized for additions
#[derive(Debug, Clone)]
pub struct AdditionGate {
    /// Number of additions operations performed by an addition gate.
    pub num_ops: usize,
}

impl AdditionGate {
    pub const fn new_from_config(config: &CircuitConfig) -> Self {
        Self {
            num_ops: Self::num_ops(config),
        }
    }

    /// Determine the maximum number of operations that can fit in one gate for the given config.
    pub(crate) const fn num_ops(config: &CircuitConfig) -> usize {
        let wires_per_op = 3;
        config.num_routed_wires / wires_per_op
    }

    pub(crate) const fn wire_ith_addend_0(i: usize) -> usize {
        3 * i
    }
    pub(crate) const fn wire_ith_addend_1(i: usize) -> usize {
        3 * i + 1
    }
    pub(crate) const fn wire_ith_output(i: usize) -> usize {
        3 * i + 2
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for AdditionGate {
    fn id(&self) -> String {
        format!("{self:?}")
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.num_ops)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let num_ops = src.read_usize()?;
        Ok(Self { num_ops })
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let addend_0 = vars.local_wires[Self::wire_ith_addend_0(i)];
            let addend_1 = vars.local_wires[Self::wire_ith_addend_1(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = addend_0 * const_0 + addend_1 * const_1;

            constraints.push(output - computed_output);
        }

        constraints
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        let mut constraints = Vec::with_capacity(self.num_ops);
        for i in 0..self.num_ops {
            let addend_0 = vars.local_wires[Self::wire_ith_addend_0(i)];
            let addend_1 = vars.local_wires[Self::wire_ith_addend_1(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];

            let true_addend_0 = builder.mul_extension(const_0, addend_0);
            let true_addend_1 = builder.mul_extension(const_1, addend_1);

            let computed_output = builder.add_extension(true_addend_0, true_addend_1);

            let diff = builder.sub_extension(output, computed_output);
            constraints.push(diff);
        }

        constraints
    }

    fn generators(&self, row: usize, local_constants: &[F]) -> Vec<WitnessGeneratorRef<F, D>> {
        (0..self.num_ops)
            .map(|i| {
                WitnessGeneratorRef::new(
                    AdditionBaseGenerator {
                        row,
                        const_0: local_constants[0],
                        const_1: local_constants[1],
                        i,
                    }
                    .adapter(),
                )
            })
            .collect()
    }

    fn num_wires(&self) -> usize {
        self.num_ops * 3
    }

    fn num_constants(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        2
    }

    fn num_constraints(&self) -> usize {
        self.num_ops
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for AdditionGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        let const_0 = vars.local_constants[0];
        let const_1 = vars.local_constants[1];

        for i in 0..self.num_ops {
            let addend_0 = vars.local_wires[Self::wire_ith_addend_0(i)];
            let addend_1 = vars.local_wires[Self::wire_ith_addend_1(i)];
            let output = vars.local_wires[Self::wire_ith_output(i)];
            let computed_output = addend_0 * const_0 + addend_1 * const_1;

            yield_constr.one(output - computed_output);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AdditionBaseGenerator<F: RichField + Extendable<D>, const D: usize> {
    row: usize,
    const_0: F,
    const_1: F,
    i: usize,
}

impl<F: RichField + Extendable<D>, const D: usize> SimpleGenerator<F, D>
    for AdditionBaseGenerator<F, D>
{
    fn id(&self) -> String {
        "AdditionBaseGenerator".to_string()
    }

    fn dependencies(&self) -> Vec<Target> {
        [
            AdditionGate::wire_ith_addend_0(self.i),
            AdditionGate::wire_ith_addend_1(self.i),
        ]
        .iter()
        .map(|&i| Target::wire(self.row, i))
        .collect()
    }

    fn run_once(
        &self,
        witness: &PartitionWitness<F>,
        out_buffer: &mut GeneratedValues<F>,
    ) -> Result<()> {
        let get_wire = |wire: usize| -> F { witness.get_target(Target::wire(self.row, wire)) };

        let addend_0 = get_wire(AdditionGate::wire_ith_addend_0(self.i));
        let addend_1 = get_wire(AdditionGate::wire_ith_addend_1(self.i));

        let output_target = Target::wire(self.row, AdditionGate::wire_ith_output(self.i));

        let computed_output = addend_0 * self.const_0 + addend_1 * self.const_1;

        out_buffer.set_target(output_target, computed_output)
    }

    fn serialize(&self, dst: &mut Vec<u8>, _common_data: &CommonCircuitData<F, D>) -> IoResult<()> {
        dst.write_usize(self.row)?;
        dst.write_field(self.const_0)?;
        dst.write_field(self.const_1)?;
        dst.write_usize(self.i)
    }

    fn deserialize(src: &mut Buffer, _common_data: &CommonCircuitData<F, D>) -> IoResult<Self> {
        let row = src.read_usize()?;
        let const_0 = src.read_field()?;
        let const_1 = src.read_field()?;
        let i = src.read_usize()?;
        Ok(Self {
            row,
            const_0,
            const_1,
            i,
        })
    }
}

#[cfg(test)]
mod tests {
    use core::marker::PhantomData;
    use anyhow::Result;
    use std::fs;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;
    use crate::gates::addition_base::AdditionGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::arithmetic_base::ArithmeticBaseGenerator;
    use crate::gates::poseidon::PoseidonGenerator;
    use crate::gates::poseidon_mds::PoseidonMdsGenerator;
    use crate::iop::generator::{ConstantGenerator, RandomValueGenerator};
    use crate::iop::target::Target;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};
    use crate::recursion::dummy_circuit::DummyProofGenerator;
    use crate::util::serialization::{DefaultGateSerializer, WitnessGeneratorSerializer};
    use crate::{get_generator_tag_impl, impl_generator_serializer, read_generator_impl};

    #[derive(Default)]
    pub struct AdditionTestGeneratorSerializer<C: GenericConfig<D>, const D: usize> {
        pub _phantom: PhantomData<C>,
    }

    impl<F, C, const D: usize> WitnessGeneratorSerializer<F, D> for AdditionTestGeneratorSerializer<C, D>
    where
        F: crate::hash::hash_types::RichField + crate::field::extension::Extendable<D>,
        C: GenericConfig<D, F = F> + 'static,
        C::Hasher: AlgebraicHasher<F>,
    {
        impl_generator_serializer! {
            AdditionTestGeneratorSerializer,
            DummyProofGenerator<F, C, D>,
            ArithmeticBaseGenerator<F, D>,
            ConstantGenerator<F>,
            PoseidonGenerator<F, D>,
            PoseidonMdsGenerator<D>,
            RandomValueGenerator,
            crate::gates::addition_base::AdditionBaseGenerator<F, D>
        }
    }

    #[test]
    fn low_degree() {
        let gate = AdditionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_low_degree::<GoldilocksField, _, 4>(gate);
    }

    #[test]
    fn eval_fns() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        let gate = AdditionGate::new_from_config(&CircuitConfig::standard_recursion_config());
        test_eval_fns::<F, C, _, D>(gate)
    }

    #[test]
    fn test_circuit_serialization_deserialization() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        // Step 1: Build a circuit with AdditionGate
        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());

        let gate = AdditionGate::new_from_config(&config);
        let constants = [F::ONE, F::ONE];

        // Create some addition operations
        let mut test_targets = Vec::new();
        for _ in 0..10 {
            let x = builder.add_virtual_target();
            let y = builder.add_virtual_target();
            let output_value = builder.add_virtual_target();

            let (gate_row, i) = builder.find_slot(gate.clone(), &constants, &constants);

            let wire_x = Target::wire(gate_row, AdditionGate::wire_ith_addend_0(i));
            let wire_y = Target::wire(gate_row, AdditionGate::wire_ith_addend_1(i));
            let wire_output = Target::wire(gate_row, AdditionGate::wire_ith_output(i));

            builder.connect(x, wire_x);
            builder.connect(y, wire_y);
            builder.connect(output_value, wire_output);

            test_targets.push((x, y, output_value));
        }

        let circuit_data = builder.build::<C>();

        // Step 2: Serialize circuit data to bytes and then to file
        let filename = "test_addition_circuit.dat";
        {
            let gate_serializer = DefaultGateSerializer;
            let generator_serializer = AdditionTestGeneratorSerializer::<C, D>::default();

            let data_bytes = circuit_data
                .to_bytes(&gate_serializer, &generator_serializer)
                .map_err(|_| anyhow::Error::msg("CircuitData serialization failed."))?;

            fs::write(filename, &data_bytes)?;
        }

        // Step 3: Deserialize circuit data from file
        let deserialized_circuit_data = {
            let gate_serializer = DefaultGateSerializer;
            let generator_serializer = AdditionTestGeneratorSerializer::<C, D>::default();

            let data_bytes = fs::read(filename)?;
            CircuitData::<F, C, D>::from_bytes(
                &data_bytes,
                &gate_serializer,
                &generator_serializer,
            )
            .map_err(|_| anyhow::Error::msg("CircuitData deserialization failed."))?
        };

        // Step 4: Verify the deserialized circuit works correctly
        // Create witness data
        let mut pw = PartialWitness::new();
        
        for (x, y, output_value) in &test_targets {
            let value1 = F::from_canonical_u64(42);
            let value2 = F::from_canonical_u64(58);
            let expected = value1 + value2; // Since constants are both 1, this is just addition
            
            pw.set_target(*x, value1)?;
            pw.set_target(*y, value2)?;
            pw.set_target(*output_value, expected)?;
        }

        // Step 5: Generate proof with deserialized circuit
        let proof = deserialized_circuit_data.prove(pw)?;
        
        // Step 6: Verify proof with deserialized circuit
        deserialized_circuit_data.verify(proof)?;

        // Step 7: Test that a new prover/verifier instance can use the deserialized data
        let mut pw2 = PartialWitness::new();
        for (x, y, output_value) in &test_targets {
            let value1 = F::from_canonical_u64(100);
            let value2 = F::from_canonical_u64(200);
            let expected = value1 + value2;
            
            pw2.set_target(*x, value1)?;
            pw2.set_target(*y, value2)?;
            pw2.set_target(*output_value, expected)?;
        }

        let proof2 = deserialized_circuit_data.prove(pw2)?;
        deserialized_circuit_data.verify(proof2)?;

        // Clean up the test file
        std::fs::remove_file(filename)?;

        println!("✅ Circuit serialization/deserialization test passed!");
        println!("   - Circuit with AdditionGate was successfully serialized to file");
        println!("   - Circuit was successfully deserialized from file");
        println!("   - Deserialized circuit can generate and verify proofs");
        println!("   - Multiple proofs can be generated with the same deserialized circuit");

        Ok(())
    }
}
