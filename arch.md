# Plonky2 Circuit Architecture

This document explains how circuits are laid out and structured in Plonky2.

## Table of Contents
- [Circuit Matrix Structure](#circuit-matrix-structure)
- [Gate Placement](#gate-placement)
- [Wire Organization](#wire-organization)
- [Data Structure Hierarchy](#data-structure-hierarchy)
- [Constraint System](#constraint-system)
- [Copy Constraints & Permutation](#copy-constraints--permutation)
- [Witness Generation Pipeline](#witness-generation-pipeline)
- [Polynomial Commitments](#polynomial-commitments)
- [Key Design Principles](#key-design-principles)

## Circuit Matrix Structure

The circuit is fundamentally a **2D matrix**:
- **Rows**: Gates (operations), numbered 0 to `degree`
- **Columns**: 135 wires total
  - 80 routed wires (participate in copy constraints/permutation argument)
  - 55 advice wires (local to gates, used for intermediate values)

**Reference**: [plonky2/src/plonk/circuit_builder.rs:141-207](plonky2/src/plonk/circuit_builder.rs#L141-L207)

### Wire Layout
```
Wire Index │ Type          │ Purpose
───────────┼───────────────┼──────────────────────────────
0-79       │ Routed        │ Can be connected across gates via permutation
80-134     │ Advice        │ Local helper wires, not part of permutation
```

**Reference**: [plonky2/src/plonk/circuit_data.rs:56-88](plonky2/src/plonk/circuit_data.rs#L56-L88)

```rust
pub const NUM_ROUTED_WIRES: usize = 80;
pub const NUM_ADVICE_WIRES: usize = 55;
pub const NUM_WIRES: usize = NUM_ROUTED_WIRES + NUM_ADVICE_WIRES;
```

## Gate Placement

Gates are placed **sequentially** in the circuit matrix using a greedy algorithm:

1. Each gate type defines how many constraint "slots" it needs
2. The builder searches for the next available slot using `find_slot()`
3. Gates are packed efficiently to minimize circuit size
4. Gates with the same degree are grouped together

**Reference**: [plonky2/src/plonk/circuit_builder.rs:815-845](plonky2/src/plonk/circuit_builder.rs#L815-L845)

### Selector Polynomials

Instead of having one selector per gate type, Plonky2 uses **selector polynomials** that partition gates by degree:
- Gates of degree D are grouped together
- Selector polynomial is 1 for gates of that degree, 0 elsewhere
- This enables efficient constraint evaluation without per-gate filtering

**Reference**: [plonky2/src/plonk/get_vecs.rs:12-68](plonky2/src/plonk/get_vecs.rs#L12-L68)

## Wire Organization

### Routed Wires (0-79)
- Participate in the **permutation argument**
- Can be connected across different gates
- Used for inputs/outputs that need to be constrained equal
- Example: connecting output of gate A to input of gate B

### Advice Wires (80-134)
- Local to individual gates
- Do NOT participate in permutation
- Used for intermediate computations
- Reduces pressure on routed wires
- Example: temporary values in arithmetic operations

**Reference**: [plonky2/src/plonk/circuit_data.rs:56-88](plonky2/src/plonk/circuit_data.rs#L56-L88)

## Data Structure Hierarchy

The circuit data is split into three components for efficiency:

```
CircuitData
├── ProverOnlyCircuitData
│   ├── Generators (compute witness values)
│   ├── Sigma polynomials (permutation mappings)
│   ├── Forest (union-find for copy constraints)
│   ├── Representative map
│   └── FFT precomputation tables
├── VerifierOnlyCircuitData
│   ├── Constants Merkle cap
│   └── Circuit digest
└── CommonCircuitData
    ├── CircuitConfig
    ├── Gates (list of all gate instances)
    ├── Selectors (degree-based partitioning)
    ├── Quotient degree factor
    ├── Public input indices
    ├── FRI parameters
    └── Circuit digest
```

### CircuitData
Main container holding all circuit information.

**Reference**: [plonky2/src/plonk/circuit_data.rs:185-191](plonky2/src/plonk/circuit_data.rs#L185-L191)

### ProverOnlyCircuitData
Information needed only by the prover:
- **Generators**: Compute witness values from partial witness
- **Sigma polynomials**: Encode the permutation mapping for copy constraints
- **Forest**: Union-find data structure tracking which wires are constrained equal
- **FFT tables**: Precomputed for polynomial operations

**Reference**: [plonky2/src/plonk/circuit_data.rs:428-440](plonky2/src/plonk/circuit_data.rs#L428-L440)

### VerifierOnlyCircuitData
Minimal information for verification:
- Constants Merkle cap (commitment to constants)
- Circuit digest (hash of circuit structure)

**Reference**: [plonky2/src/plonk/circuit_data.rs:402-426](plonky2/src/plonk/circuit_data.rs#L402-L426)

### CommonCircuitData
Shared between prover and verifier:
- Configuration parameters
- Gate definitions
- Selector polynomials
- Public input locations
- FRI parameters

**Reference**: [plonky2/src/plonk/circuit_data.rs:442-480](plonky2/src/plonk/circuit_data.rs#L442-L480)

## Constraint System

Each gate implements the `Gate` trait which defines:

```rust
pub trait Gate<F: RichField + Extendable<D>, const D: usize>: ... {
    fn num_wires(&self) -> usize;           // How many wires it uses
    fn num_constants(&self) -> usize;        // How many constants it needs
    fn degree(&self) -> usize;               // Max degree of constraints
    fn num_constraints(&self) -> usize;      // Number of polynomial equations

    fn eval_unfiltered(&self, ...);          // Evaluate constraints
    fn eval_filtered(&self, ...);            // Evaluate with selector
}
```

**Reference**: [plonky2/src/plonk/gates/gate.rs:53-260](plonky2/src/plonk/gates/gate.rs#L53-L260)

### Constraint Evaluation

Constraints are evaluated in **point-major order**:
- Evaluate all constraints at point 1
- Then all constraints at point 2
- Then all constraints at point 3
- ...

This is more SIMD-friendly than gate-major order.

**Reference**: [plonky2/src/plonk/get_vecs.rs:70-110](plonky2/src/plonk/get_vecs.rs#L70-L110)

### Gate Instance

A gate instance consists of:
- Gate index (which gate definition)
- Row index (which row in the circuit matrix)

**Reference**: [plonky2/src/plonk/gates/gate.rs:319-322](plonky2/src/plonk/gates/gate.rs#L319-L322)

## Copy Constraints & Permutation

Plonky2 uses the **PLONK permutation argument** to enforce that wires constrained to be equal actually have equal values.

### Forest (Union-Find)

Tracks which wires are constrained to be equal:
- Each wire starts in its own set
- `copy_constraint(w1, w2)` unions the sets
- Eventually computes a permutation mapping

**Reference**: [plonky2/src/plonk/permutation_argument.rs:13-156](plonky2/src/plonk/permutation_argument.rs#L13-L156)

### Sigma Polynomials

Encode the permutation mapping:
- For each wire `w`, `sigma(w)` tells you the next wire in its equivalence class
- Forms a cycle through all wires that must be equal
- Committed as part of the circuit structure

**Reference**: [plonky2/src/plonk/permutation_argument.rs:45-91](plonky2/src/plonk/permutation_argument.rs#L45-L91)

### Permutation Argument

During proving:
1. Compute partial products based on wires and sigmas
2. Accumulate these into the `Z` polynomial
3. Prove that `Z` forms a valid permutation product

During verification:
- Check that permutation constraints hold at random point

**Reference**: [plonky2/src/plonk/prover.rs:250-289](plonky2/src/plonk/prover.rs#L250-L289)

## Witness Generation Pipeline

The witness goes through three forms:

```
PartialWitness (sparse, user-provided)
    ↓ (apply generators)
PartitionWitness (respects copy constraints)
    ↓ (flatten to column-major matrix)
MatrixWitness (dense, ready for polynomials)
```

### PartialWitness

- Sparse representation (HashMap)
- User provides initial values (public inputs, private inputs)
- Not all wires need to be set

**Reference**: [plonky2/src/iop/witness.rs:283-308](plonky2/src/iop/witness.rs#L283-L308)

### PartitionWitness

- Organized by copy-constraint partitions
- Each partition has one representative wire
- Setting a wire sets all wires in its partition
- Generators fill in missing values

**Reference**: [plonky2/src/iop/witness.rs:310-377](plonky2/src/iop/witness.rs#L310-L377)

### MatrixWitness

- Dense 2D array
- Column-major layout (wires are contiguous)
- Ready to be converted to polynomials via FFT
- Used for final proof generation

**Reference**: [plonky2/src/iop/witness.rs:379-402](plonky2/src/iop/witness.rs#L379-L402)

### Generators

Generators compute derived witness values:
- Take some inputs and compute outputs
- Run in topological order based on dependencies
- Examples: arithmetic operations, hash outputs, lookup multiplicities

**Reference**: [plonky2/src/iop/generator.rs:33-142](plonky2/src/iop/generator.rs#L33-L142)

## Polynomial Commitments

Plonky2 uses FRI for polynomial commitments. There are four oracles:

### Oracle 1: CONSTANTS_SIGMAS
- Constants (gate constants, public inputs)
- Sigma polynomials (permutation mappings)
- **Not blinded** (deterministic, part of circuit structure)

**Reference**: [plonky2/src/plonk/prover.rs:119-148](plonky2/src/plonk/prover.rs#L119-L148)

### Oracle 2: WIRES
- Wire witness values
- **Blinded** with random salt
- Committed after witness generation

**Reference**: [plonky2/src/plonk/prover.rs:153-181](plonky2/src/plonk/prover.rs#L153-L181)

### Oracle 3: ZS_PARTIAL_PRODUCTS
- Permutation product polynomial (Z)
- Partial products for permutation argument
- **Blinded**

**Reference**: [plonky2/src/plonk/prover.rs:250-289](plonky2/src/plonk/prover.rs#L250-L289)

### Oracle 4: QUOTIENT
- Quotient polynomial from constraint division
- Proves all constraints are satisfied
- **Blinded**

**Reference**: [plonky2/src/plonk/prover.rs:291-339](plonky2/src/plonk/prover.rs#L291-L339)

### Polynomial Batch Process

For each oracle:
1. Coefficients in evaluation form
2. **FFT** to coefficient form
3. **Low-degree extension** (LDE) by interpolation
4. Add **blinding salt** (random polynomial)
5. Evaluate LDE on larger domain
6. Build **Merkle tree** over evaluations
7. Return Merkle cap as commitment

**Reference**: [plonky2/src/plonk/prover.rs:73-111](plonky2/src/plonk/prover.rs#L73-L111)

### Opening Points

Polynomials are opened at two points:
- `zeta`: Random challenge point
- `g * zeta`: Next point in coset (for permutation argument)

**Reference**: [plonky2/src/plonk/verifier.rs:42-167](plonky2/src/plonk/verifier.rs#L42-L167)

## Key Design Principles

### 1. Efficiency Through Selectors
Selector polynomials group gates by degree, enabling constraint evaluation without filtering by individual gate type. This is more efficient than standard PLONK.

### 2. Routed vs Advice Wires
Separating routed wires (participate in permutation) from advice wires (local to gates) reduces the cost of the permutation argument while maintaining flexibility.

### 3. Modularity
Gates are self-contained with their own constraint logic. New gates can be added without modifying the core proving system.

### 4. SIMD-Friendly Layout
- Point-major constraint evaluation
- Column-major witness layout
- Both enable efficient vectorization

### 5. Prover/Verifier Separation
Splitting data into ProverOnly, VerifierOnly, and Common minimizes what the verifier needs, reducing verification cost.

### 6. Generator Pipeline
The generator system allows complex witness computation while maintaining a clean separation between circuit definition and witness generation.

### 7. Lookup Arguments
Lookup tables enable efficient range checks, XOR operations, and other lookups without expensive bitwise constraints.

**Reference**: [plonky2/src/plonk/circuit_builder.rs:1357-1472](plonky2/src/plonk/circuit_builder.rs#L1357-L1472)

## Advanced Features

### Recursion
Plonky2 can verify its own proofs:
- Verifier circuit is built using the circuit builder
- Enables proof composition and aggregation
- Special gates for efficient field arithmetic

**Reference**: [plonky2/src/recursion/](plonky2/src/recursion/)

### Custom Gates
Users can define custom gates for specific operations:
- Implement the `Gate` trait
- Define constraints and evaluation logic
- Register with the circuit builder

**Reference**: [plonky2/src/plonk/gates/gate.rs:53-260](plonky2/src/plonk/gates/gate.rs#L53-L260)

### Lookup Tables
Efficient lookups for operations like:
- Range checks
- Bitwise operations (XOR, AND)
- Small field operations
- S-boxes (for hash functions)

**Reference**: [plonky2/src/gates/lookup.rs](plonky2/src/gates/lookup.rs), [plonky2/src/gates/lookup_table.rs](plonky2/src/gates/lookup_table.rs)

## File Reference Index

### Core Circuit Structure
- [plonky2/src/plonk/circuit_data.rs](plonky2/src/plonk/circuit_data.rs) - Main data structures
- [plonky2/src/plonk/circuit_builder.rs](plonky2/src/plonk/circuit_builder.rs) - Circuit construction

### Gates
- [plonky2/src/plonk/gates/gate.rs](plonky2/src/plonk/gates/gate.rs) - Gate trait and instances
- [plonky2/src/gates/](plonky2/src/gates/) - Concrete gate implementations

### Witness
- [plonky2/src/iop/witness.rs](plonky2/src/iop/witness.rs) - Witness types
- [plonky2/src/iop/generator.rs](plonky2/src/iop/generator.rs) - Generator system

### Proving & Verification
- [plonky2/src/plonk/prover.rs](plonky2/src/plonk/prover.rs) - Proof generation
- [plonky2/src/plonk/verifier.rs](plonky2/src/plonk/verifier.rs) - Proof verification

### Permutation Argument
- [plonky2/src/plonk/permutation_argument.rs](plonky2/src/plonk/permutation_argument.rs) - Copy constraints

### Polynomials
- [plonky2/src/plonk/get_vecs.rs](plonky2/src/plonk/get_vecs.rs) - Polynomial evaluation
- [plonky2/src/fri/](plonky2/src/fri/) - FRI commitment scheme

### Recursion
- [plonky2/src/recursion/](plonky2/src/recursion/) - Recursive proof verification

## Example: Simple Circuit

Here's how a simple circuit `c = a + b * 3` would be laid out:

```
Row 0: PublicInput gate (for input a)
Row 1: PublicInput gate (for input b)
Row 2: ArithmeticGate (b * 3)
Row 3: ArithmeticGate (a + result_from_row2)
Row 4-N: Padding to reach power-of-2 degree

Copy constraints:
- a (row 0, wire 0) = a (row 3, wire 0)
- b (row 1, wire 0) = b (row 2, wire 0)
- result (row 2, wire 2) = operand (row 3, wire 1)
```

The witness generation would:
1. User provides `a` and `b` in PartialWitness
2. Generators compute intermediate values
3. PartitionWitness ensures copy constraints are satisfied
4. MatrixWitness provides final polynomial values

See [plonky2/examples/](plonky2/examples/) for complete working examples.

---

**Last Updated**: 2025-11-04
**Plonky2 Version**: Based on plonky2-lighter repository
