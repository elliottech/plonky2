extern crate alloc;
use alloc::string::ToString;

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
    use anyhow::Result;
    use plonky2_field::types::Sample;

    use crate::field::goldilocks_field::GoldilocksField;
    use crate::field::types::Field;
    #[allow(unused_imports)]
    use crate::field::types::Field64;
    use crate::gates::addition_base::AdditionGate;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::iop::target::Target;
    use crate::iop::witness::{PartialWitness, WitnessWrite};
    use crate::plonk::circuit_builder::CircuitBuilder;
    use crate::plonk::circuit_data::{CircuitConfig, CircuitData};
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    use crate::util::serialization::{DefaultGateSerializer, DefaultGeneratorSerializer};

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
    fn test_success() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        fn flag_test(flag: usize) -> Result<()> {
            let config = CircuitConfig {
                optimization_flags: flag,
                ..CircuitConfig::standard_recursion_config()
            };
            let mut builder = CircuitBuilder::<F, D>::new(config.clone());

            let mut pairs = vec![];

            let gate = AdditionGate::new_from_config(&config);
            let constants = [F::ONE, F::ONE];

            for _ in 0..100 {
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

                pairs.push((x, y, output_value));
            }

            let circuit_data = builder.build::<C>();

            let mut pw = PartialWitness::new();
            for (x, y, output_value) in pairs.iter() {
                let value1 = F::rand();
                let value2 = F::rand();
                let expected = value1 + value2;
                pw.set_target(*x, value1)?;
                pw.set_target(*y, value2)?;
                pw.set_target(*output_value, expected)?;
            }

            let proof = circuit_data.prove(pw)?;
            circuit_data.verify(proof)?;

            Ok(())
        }

        flag_test(63)?; // flag enabled
        flag_test(62)?; // flag disabled

        Ok(())
    }

    #[test]
    #[should_panic]
    fn test_failure() {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        fn flag_test(flag: usize) {
            let config = CircuitConfig {
                optimization_flags: flag,
                ..CircuitConfig::standard_recursion_config()
            };
            let mut builder = CircuitBuilder::<F, D>::new(config.clone());

            let mut pairs = vec![];

            let gate = AdditionGate::new_from_config(&config);
            let constants = [F::ONE, F::ONE];

            for _ in 0..100 {
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

                pairs.push((x, y, output_value));
            }

            let circuit_data = builder.build::<C>();

            let mut pw = PartialWitness::new();
            for (x, y, output_value) in pairs.iter() {
                let value1 = F::rand();
                let value2 = F::rand();
                let expected = value1 + value2;
                let mut incorrect_value = F::rand();
                while incorrect_value == expected {
                    incorrect_value = F::rand();
                }
                pw.set_target(*x, value1).unwrap();
                pw.set_target(*y, value2).unwrap();
                pw.set_target(*output_value, incorrect_value).unwrap();
            }

            let proof = circuit_data.prove(pw).unwrap();
            circuit_data.verify(proof).unwrap();
        }

        flag_test(63); // flag enabled
        flag_test(62); // flag disabled
    }

    #[test]
    fn test_serialization_addition() -> Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;

        let config = CircuitConfig::standard_recursion_config();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());

        let mut pairs = vec![];

        let gate = AdditionGate::new_from_config(&config);
        let constants = [F::ONE, F::ONE];

        for _ in 0..100 {
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

            pairs.push((x, y, output_value));
        }

        let circuit_data = builder.build::<C>();
        // let filename = "test_circuit";
        let gate_serializer = DefaultGateSerializer;
        let generator_serializer = DefaultGeneratorSerializer::<C, D>::default();

        let data_bytes = circuit_data
            .to_bytes(&gate_serializer, &generator_serializer)
            .map_err(|_| anyhow::Error::msg("Serialization failed."))?;

        // fs::write(filename, &data_bytes)?;

        // let read_data_bytes = fs::read(filename)?;

        let deserialized_circuit_data = CircuitData::<F, C, D>::from_bytes(
            &data_bytes,
            &gate_serializer,
            &generator_serializer,
        )
        .map_err(|_| anyhow::Error::msg("Deserialization failed."))?;

        assert_eq!(deserialized_circuit_data, circuit_data);

        let mut pw = PartialWitness::new();

        for (x, y, output_value) in pairs.iter() {
            let value1 = F::rand();
            let value2 = F::rand();
            let expected = value1 + value2;

            pw.set_target(*x, value1)?;
            pw.set_target(*y, value2)?;
            pw.set_target(*output_value, expected)?;
        }

        let proof = deserialized_circuit_data.prove(pw.clone())?;
        deserialized_circuit_data.verify(proof.clone())?;

        // std::fs::remove_file(filename)?;

        Ok(())
    }
}
