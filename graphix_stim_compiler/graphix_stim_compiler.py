"""Clifford compilation pass using stim functionalities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import stim
from graphix.circ_ext.compilation import CliffordMapCompilationPass

if TYPE_CHECKING:
    from graphix.circ_ext.extraction import CliffordMap, PauliString
    from graphix.transpiler import Circuit

    _SYNTH_METHOD: TypeAlias = Literal["elimination", "graph_state"]


class StimCliffordPass(CliffordMapCompilationPass):
    """Compilation pass to synthetize a Clifford map by using stim's tableau synthesis.

    This pass only handles unitaries (Clifford maps with the same number of input and ouptut nodes).

    Gate set: H, S, CNOT
    """

    @staticmethod
    def add_to_circuit(clifford_map: CliffordMap, circuit: Circuit) -> None:
        """Add the Clifford map to a quantum circuit.

        The input circuit is modified in-place.

        Parameters
        ----------
        clifford_map: CliffordMap
            The Clifford map to be synthesized.
        circuit : Circuit
            The quantum circuit to which the Clifford map is added.
        """
        stim_circuit = StimCliffordPass.to_stim_circuit(clifford_map, method="elimination")  # Gate set: H, S, CX

        # "Circuit" has no attribute "__iter__"
        # (but __len__ and __getitem__)
        instruction: stim.CircuitInstruction
        for instruction in stim_circuit:  # type: ignore[attr-defined]
            match instruction.name:
                case "CX":
                    for control, target in instruction.target_groups():
                        circuit.cnot(control.qubit_value, target.qubit_value)
                case "H":
                    for (qubit,) in instruction.target_groups():
                        circuit.h(qubit.qubit_value)
                case "S":
                    for (qubit,) in instruction.target_groups():
                        circuit.s(qubit.qubit_value)

    @staticmethod
    def clifford_map_to_stim_tableau(clifford_map: CliffordMap) -> stim.Tableau:
        """Transpile the Clifford map into a stim tableau.

        Parameters
        ----------
        clifford_map: CliffordMap
            The Clifford map to be transpiled.

        Returns
        -------
        stim.Tableau

        Raises
        ------
        NotImplementedError
            If ``len(self.input_nodes) != len(self.output_nodes)``.
        """
        if len(clifford_map.input_nodes) != len(clifford_map.output_nodes):
            raise NotImplementedError(
                "StimCliffordPass does not support circuit compilation if the number of input and output nodes is different (isometry)."
            )

        xs: list[stim.PauliString] = []
        zs: list[stim.PauliString] = []
        n_qubits = len(clifford_map.output_nodes)

        for qubit in range(n_qubits):
            xs.append(pauli_string_to_stim(clifford_map.x_map[qubit], n_qubits))
            zs.append(pauli_string_to_stim(clifford_map.z_map[qubit], n_qubits))

        return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)

    @staticmethod
    def to_stim_circuit(clifford_map: CliffordMap, method: _SYNTH_METHOD) -> stim.Circuit:
        """Transpile the Clifford map into a stim circuit.

        Parameters
        ----------
        clifford_map: CliffordMap
            The Clifford map to be transpiled.
        method: Literal["elimination", "graph_state"]
            Stim method for synthetizing the circuit.

        Returns
        -------
        stim.Circuit

        Notes
        -----
        See https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Tableau.to_circuit for additional information.
        """
        return StimCliffordPass.clifford_map_to_stim_tableau(clifford_map).to_circuit(method)


def pauli_string_to_stim(ps: PauliString, n_qubits: int) -> stim.PauliString:
    """Transform a :class:`graphix.circ_extraction.extraction.PauliString` into a :class:`stim.PauliString` instance.

    This method assumes that the qubit sets in ``ps`` are pairwise disjoint. It also assumes that the Pauli string has been remap, i.e., it is defined on qubit indices and not on node values. See :meth:`graphix.circ_ext.PauliString.remap` for additional information.

    Parameters
    ----------
    ps: PauliString
        The Pauli string to be transformed.
    n_qubits : int
        Width of the circuit on which the Pauli string is defined.

    Returns
    -------
    stim.PauliString
        The Pauli string in `stim` format.

    Raises
    ------
    ValueError
        If the Pauli string is not compatible with ``n_qubits``.

    Notes
    -----
    Qubits not appearing in ``ps.x_nodes | ps.y_nodes | ps.z_nodes`` are assigned the identity operator in the returned `stim.PauliString`.
    """
    if not set(ps.x_nodes | ps.y_nodes | ps.z_nodes).issubset(range(n_qubits)):
        raise ValueError("The Pauli string contains qubit indices beyond the circuit's width.")

    pauli_str: list[str] = [str(ps.sign)]
    for qubit in range(n_qubits):
        if qubit in ps.x_nodes:
            pauli_str.append("X")
        elif qubit in ps.y_nodes:
            pauli_str.append("Y")
        elif qubit in ps.z_nodes:
            pauli_str.append("Z")
        else:
            pauli_str.append("_")

    return stim.PauliString("".join(pauli_str))
