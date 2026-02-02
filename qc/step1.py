# ===============================
# STEP 1: Single Qubit "Ball"
# Bloch Sphere Visualization
# ===============================

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector

# -------------------------------
# 1. Create a single-qubit circuit
# -------------------------------
qc = QuantumCircuit(1)

# Initial state |0>
print("Initial state: |0>")

# -------------------------------
# 2. Apply quantum gates (rotations)
# -------------------------------

# Put qubit into superposition
qc.h(0)

# Rotate around Y axis
theta_y = np.pi / 4
qc.ry(theta_y, 0)

# Rotate around Z axis
theta_z = np.pi / 3
qc.rz(theta_z, 0)

print("Applied gates: H, Ry, Rz")

# -------------------------------
# 3. Get the quantum state
# -------------------------------
state = Statevector.from_instruction(qc)

print("Statevector:")
print(state)

# -------------------------------
# 4. Plot the Bloch sphere
# -------------------------------
fig = plot_bloch_multivector(state)
plt.show()