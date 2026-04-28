"""Compilation pass for Clifford maps using stim functionalities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

import stim
from graphix.fundamentals import Sign

if TYPE_CHECKING:
    from graphix.circ_ext.extraction import CliffordMap, PauliString
    from graphix.transpiler import Circuit

    _SYNTH_METHOD: TypeAlias = Literal["elimination", "graph_state"]


def cm_stim_pass(clifford_map: CliffordMap, circuit: Circuit) -> None:
    """Add a Clifford map to a circuit by using stim's tableau synthesis.

    The input circuit is modified in-place.

    Parameters
    ----------
    clifford_map: CliffordMap
        The Clifford map to be transpiled.
    circuit : Circuit
        The circuit to which the operation is added. The input circuit is assumed to be compatible with ``CliffordMap.input_nodes`` and ``CliffordMap.output_nodes``.

    Notes
    -----
    This pass only handles unitaries (Clifford maps with the same number of input and ouptut nodes).

    Gate set: H, S, CNOT
    """

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
                ":func:`cm_stim_pass` does not support circuit compilation if the number of input and output nodes is different (isometry)."
            )

        xs: list[stim.PauliString] = []
        zs: list[stim.PauliString] = []

        for x, z in zip(clifford_map.x_map, clifford_map.z_map, strict=True):
            xs.append(pauli_string_to_stim(x))
            zs.append(pauli_string_to_stim(z))

        return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)

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
        return clifford_map_to_stim_tableau(clifford_map).to_circuit(method)

    stim_circuit = to_stim_circuit(clifford_map, method="elimination")  # Gate set: H, S, CX

    # "stim.Circuit" has no attribute "__iter__"
    # (but __len__ and __getitem__)
    instruction: stim.CircuitInstruction
    for instruction in stim_circuit:  # type: ignore[attr-defined]
        match instruction.name:
            case "CX":
                for control, target in instruction.target_groups():
                    assert control.qubit_value is not None
                    assert target.qubit_value is not None
                    circuit.cnot(control.qubit_value, target.qubit_value)
            case "H":
                for (qubit,) in instruction.target_groups():
                    assert qubit.qubit_value is not None
                    circuit.h(qubit.qubit_value)
            case "S":
                for (qubit,) in instruction.target_groups():
                    assert qubit.qubit_value is not None
                    circuit.s(qubit.qubit_value)


def pauli_string_to_stim(ps: PauliString) -> stim.PauliString:
    """Transform a :class:`graphix.circ_ext.extraction.PauliString` into a :class:`stim.PauliString` instance.

    Parameters
    ----------
    ps: PauliString
        The Pauli string to be transformed.

    Returns
    -------
    stim.PauliString
        The Pauli string in `stim` format.

    Notes
    -----
    Qubits not appearing in ``ps.axes.keys`` are assigned the identity operator in the returned `stim.PauliString`.
    """
    pauli_str = stim.PauliString(ps.dim)
    if ps.sign == Sign.MINUS:
        pauli_str *= -1
    for qubit, axis in ps.axes.items():
        pauli_str[qubit] = axis.name
    return pauli_str
