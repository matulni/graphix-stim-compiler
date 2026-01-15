from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import stim
from graphix.circ_ext.compilation import CliffordMapCompilationPass, initialize_circuit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix.circ_ext.extraction import CliffordMap, PauliString
    from graphix.transpiler import Circuit

    _SYNTH_METHOD: TypeAlias = Literal["elimination", "graph_state"]


class StimCliffordPass(CliffordMapCompilationPass):
    @staticmethod
    def add_to_circuit(clifford_map: CliffordMap, circuit: Circuit | None = None, copy: bool = False) -> Circuit:
        """Add the Clifford map to a quantum circuit.

        This method does not handle isometries yet (only unitaries).

        Parameters
        ----------
        circuit : CircuitMBQC
            The quantum circuit to which the Clifford map is added.
        copy : bool, optional
            If ``True``, operate on a deep copy of ``circuit`` and return it.
            Otherwise, the input circuit is modified in place. Default is
            ``False``.

        Returns
        -------
        CircuitMBQC
            The circuit with the Pauli exponential applied.

        Raises
        ------
        ValueError
            If the input circuit is not compatible with ``self.output_nodes``.
        NotImplementedError
            If the Clifford map represents an isometry, i.e., ``len(self.input_nodes) != len(self.output_nodes)``.
        """
        circuit = initialize_circuit(clifford_map.output_nodes, circuit, copy)

        stim_circuit = StimCliffordPass.to_stim_circuit(clifford_map, method="elimination")  # Gate set: H, S, CX

        # "Circuit" has no attribute "__iter__"
        # (but __len__ and __getitem__)
        instruction: stim.CircuitInstruction
        for instruction in stim_circuit:
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
        return circuit

    @staticmethod
    def clifford_map_to_stim_tableau(clifford_map: CliffordMap) -> stim.Tableau:
        """Transpile the Clifford map into a stim tableau.

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

        for node in clifford_map.input_nodes:
            xs.append(pauli_string_to_stim(clifford_map.x_map[node], clifford_map.output_nodes))
            zs.append(pauli_string_to_stim(clifford_map.z_map[node], clifford_map.output_nodes))

        return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)

    @staticmethod
    def to_stim_circuit(clifford_map: CliffordMap, method: _SYNTH_METHOD) -> stim.Circuit:
        """Transpile the Clifford map into a stim circuit.

        Parameters
        ----------
        method: Literal["elimination", "graph_state"]
            Stim method for synthetizing the circuit.

        Returns
        -------
        stim.Circuit

        Raises
        ------
        NotImplementedError
            If ``len(self.input_nodes) != len(self.output_nodes)``.

        Notes
        -----
        See https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Tableau.to_circuit for additional information.
        """
        return StimCliffordPass.clifford_map_to_stim_tableau(clifford_map).to_circuit(method)


def pauli_string_to_stim(ps: PauliString, outputs: Sequence[int]) -> stim.PauliString:
    """Return a `stim.PauliString` instance.

    This method assumes that the node sets in ``ps`` are pairwise disjoint.

    Parameters
    ----------
    outputs : Sequence[int]
        Sequence of outputs nodes.

    Returns
    -------
    stim.PauliString
        The Pauli string in `stim` format.

    Notes
    -----
    Output nodes not appearing in `ps.x_nodes | ps.y_nodes | ps.z_nodes` are assigned the identity operator in the returned `stim.PauliString`.
    """
    if not set(ps.x_nodes | ps.y_nodes | ps.z_nodes).issubset(outputs):
        raise ValueError("The Pauli string contains nodes which are not in `outputs`.")

    pauli_str: list[str] = ["-" if ps.negative_sign else "+"]
    for node in outputs:
        if node in ps.x_nodes:
            pauli_str.append("X")
        elif node in ps.y_nodes:
            pauli_str.append("Y")
        elif node in ps.z_nodes:
            pauli_str.append("Z")
        else:
            pauli_str.append("_")

    return stim.PauliString("".join(pauli_str))
