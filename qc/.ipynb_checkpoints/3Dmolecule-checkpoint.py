# =========================
# Imports
# =========================

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import FreezeCoreTransformer

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

from qiskit.primitives import Estimator
from qiskit.circuit.library import TwoLocal

from ase.build import molecule
import py3Dmol

# =========================
# Molecule Input
# =========================

formula = input("Enter the molecular formula (e.g., H2, LiH): ")

ase_molecule = molecule(formula)

# =========================
# 3D Visualization
# =========================

xyz_str = f"{len(ase_molecule)}\n\n"
for atom in ase_molecule:
    xyz_str += f"{atom.symbol} {atom.x:.6f} {atom.y:.6f} {atom.z:.6f}\n"

view = py3Dmol.view(width=400, height=400)
view.addModel(xyz_str, "xyz")
view.setStyle({"stick": {}})
view.zoomTo()
view.show()

# =========================
# PySCF Driver
# =========================

pyscf_atom_str = "; ".join(
    f"{atom.symbol} {atom.x} {atom.y} {atom.z}" for atom in ase_molecule
)

driver = PySCFDriver(
    atom=pyscf_atom_str,
    basis="sto3g",
    unit=DistanceUnit.ANGSTROM,
)

# =========================
# Electronic Structure Problem (NEW API)
# =========================

problem = ElectronicStructureProblem.from_driver(
    driver,
    transformers=[FreezeCoreTransformer()],
)

# =========================
# Hamiltonian → Qubits
# =========================

second_q_ops = problem.second_q_ops()
hamiltonian = second_q_ops[0]

mapper = JordanWignerMapper()
qubit_op = mapper.map(hamiltonian)

# =========================
# VQE Setup
# =========================

ansatz = TwoLocal(
    rotation_blocks="ry",
    entanglement_blocks="cx",
    reps=2,
)

optimizer = SLSQP(maxiter=200)
estimator = Estimator()

vqe = VQE(
    estimator=estimator,
    ansatz=ansatz,
    optimizer=optimizer,
)

# =========================
# Solve
# =========================

result = vqe.compute_minimum_eigenvalue(qubit_op)
energy = result.eigenvalue.real

print(f"\nGround state energy for {formula}: {energy:.6f} Hartree")
