"""Choose where the circuits run: a local simulator, or real quantum hardware.

The algorithm does not change between the two. The same ansatz, the same
Hamiltonian, the same optimiser -- only the estimator is swapped, which is the
whole point of writing it this way. Everything in this package that takes an
estimator will run on hardware the moment one is configured.

To use real hardware you need a free IBM Quantum account and to save your own
token once, from your own shell:

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")

The token is a credential; save it yourself and it stays with you. Nothing here
asks for it or stores it.

Two honest warnings about running this on hardware today. Results will be
*worse* than the simulator, not better -- current devices are noisy enough that
a 10-qubit chemistry answer degrades well past chemical accuracy without error
mitigation. And there is a queue, so a calculation that takes seconds locally
can take hours in wall-clock. Hardware wins past roughly 40-50 qubits, where
the simulator stops being possible at all; below that the simulator is both
faster and more accurate. `qc_quantum.scaling` measures exactly where that
crossover sits.
"""

from __future__ import annotations


def local_estimator():
    """Exact statevector simulation. Noiseless, and the default."""
    from qiskit.primitives import StatevectorEstimator

    return StatevectorEstimator()


def hardware_estimator(min_qubits: int = 10, optimization_level: int = 3):
    """The least busy real device your account can reach.

    Raises RuntimeError with a usable message if no account is configured,
    rather than failing somewhere deeper with something cryptic.
    """
    try:
        from qiskit_ibm_runtime import EstimatorV2, QiskitRuntimeService
    except ImportError as e:
        raise RuntimeError(
            "qiskit-ibm-runtime is not installed: pip install qiskit-ibm-runtime"
        ) from e

    try:
        service = QiskitRuntimeService()
    except Exception as e:
        raise RuntimeError(
            "No IBM Quantum account found. Save one yourself with:\n"
            '  QiskitRuntimeService.save_account(channel="ibm_quantum", '
            'token="YOUR_TOKEN")'
        ) from e

    backend = service.least_busy(operational=True, simulator=False,
                                 min_num_qubits=min_qubits)
    estimator = EstimatorV2(mode=backend)
    estimator.options.optimization_level = optimization_level
    return estimator, backend


def describe(estimator) -> str:
    name = type(estimator).__name__
    if "Statevector" in name:
        return "local statevector simulator (noiseless, exact)"
    return f"{name} on real hardware (noisy)"
