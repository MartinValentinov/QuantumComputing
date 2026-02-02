# Qiskit Nature imports
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

# Qiskit algorithms
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

# Qiskit primitives
from qiskit.primitives import StatevectorEstimator

# Ansatz
from qiskit.circuit.library import TwoLocal

# Molecule + visualization
from ase.build import molecule
import py3Dmol

# ---- Input ----
formula = input("Enter the molecular formula (e.g., H2O, NH3, CH4): ")

# ---- Build molecule ----
ase_molecule = molecule(formula)

xyz_str = f"{len(ase_molecule)}\n\n"
for atom in ase_molecule:
    xyz_str += f"{atom.symbol} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}\n"

# ---- Visualize ----
view = py3Dmol.view(width=400, height=400)
view.addModel(xyz_str, "xyz")
view.setStyle({
    "sphere": {"scale": 0.3},
    "stick": {"radius": 0.1}
})
view.zoomTo()
view.show()

# ---- PySCF input ----
pyscf_atom_str = ""
for atom in ase_molecule:
    pyscf_atom_str += f"{atom.symbol} {atom.x} {atom.y} {atom.z}; "

driver = PySCFDriver(
    atom=pyscf_atom_str,
    basis="sto3g",
    unit=DistanceUnit.ANGSTROM,
)

driver_result = driver.run()

# ---- Freeze core ----
transformer = FreezeCoreTransformer()
driver_result = transformer.transform(driver_result)

# ---- Electronic structure problem ----
es_problem = ElectronicStructureProblem(driver_result.hamiltonian)

# ---- Hamiltonian mapping ----
second_q_op = es_problem.hamiltonian.second_q_op()
mapper = JordanWignerMapper()
qubit_op = mapper.map(second_q_op)

# ---- VQE ----
ansatz = TwoLocal(
    qubit_op.num_qubits,
    rotation_blocks="ry",
    entanglement_blocks="cx",
    reps=2,
)

optimizer = SLSQP(maxiter=200)
estimator = StatevectorEstimator()

vqe_solver = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
)

result = vqe_solver.compute_minimum_eigenvalue(qubit_op)

# ---- Total energy ----
electronic_energy = result.eigenvalue.real
total_energy = electronic_energy + es_problem.nuclear_repulsion_energy

print(f"Electronic energy: {electronic_energy:.6f} Hartree")
print(f"Total energy for {formula}: {total_energy:.6f} Hartree")

# Hartree = the standard energy unit for electrons
# Total electronic energy of all electrons interacting with all nuclei
# Nuclear repulsion energy = the repulsive energy between all nuclei
# 1 Hartree ≈ 27.2114 eV
# More negative = more strongly bound.

# ---- Color legend for 3D molecule visualization ----


# | Element | Color              |
# | ------- | ------------------ |
# | H       | White / light gray |
# | C       | Black / dark gray  |
# | N       | Blue               |
# | O       | Red                |
# | F       | Light green        |
# | Cl      | Green              |
# | Br      | Dark red           |
# | I       | Purple             |
# | S       | Yellow             |
# | P       | Orange             |
