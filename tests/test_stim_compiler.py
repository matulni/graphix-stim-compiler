from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pytest
import stim
from graphix.circ_ext.compilation import CompilationPass, LadderPass
from graphix.circ_ext.extraction import PauliString
from graphix.measurements import Measurement
from graphix.opengraph import OpenGraph
from graphix.parameter import Placeholder
from graphix.random_objects import rand_circuit
from numpy.random import Generator

from graphix_stim_compiler import StimCliffordPass, pauli_string_to_stim

if TYPE_CHECKING:
    from numpy.random import PCG64


class TestStimCliffordPass:
    def test_pauli_string_to_stim(self) -> None:
        p_str = PauliString(x_nodes={1, 4}, y_nodes={2}, z_nodes={5}, negative_sign=True)

        stim_str = pauli_string_to_stim(p_str, outputs=range(7))

        assert stim_str == stim.PauliString("-_XY_XZ_")


class TestExtraction:
    @pytest.mark.parametrize("jumps", range(1, 11))
    def test_extract_rnd_circuit(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 2
        depth = 2
        circuit_ref = rand_circuit(nqubits, depth, rng, use_ccx=False)
        pattern = circuit_ref.transpile().pattern

        cp = CompilationPass(LadderPass(), StimCliffordPass())
        circuit = pattern.extract_opengraph().extract_pauli_flow().extract_circuit().to_circuit(cp)

        s_ref = circuit.simulate_statevector().statevec
        s_test = circuit_ref.simulate_statevector().statevec
        assert np.abs(np.dot(s_ref.flatten().conjugate(), s_test.flatten())) == pytest.approx(1)

    @pytest.mark.parametrize(
        "test_case",
        [
            OpenGraph(
                graph=nx.Graph([(0, 1), (1, 20), (20, 30), (30, 4), (4, 5)]),
                input_nodes=[0],
                output_nodes=[5],
                measurements={
                    0: Measurement.XY(0.1),
                    1: Measurement.XY(0.2),
                    20: Measurement.XY(0.3),
                    30: Measurement.XY(0.4),
                    4: Measurement.XY(0.5),
                },
            ),
            OpenGraph(
                graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
                input_nodes=[1, 2],
                output_nodes=[5, 6],
                measurements={
                    1: Measurement.XY(0.1),
                    2: Measurement.XY(0.2),
                    3: Measurement.XY(0.3),
                    4: Measurement.XY(0.4),
                },
            ),
            OpenGraph(
                graph=nx.Graph([(1, 4), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (3, 6)]),
                input_nodes=[1, 2, 3],
                output_nodes=[4, 5, 6],
                measurements={
                    1: Measurement.XY(0.1),
                    2: Measurement.XY(0.2),
                    3: Measurement.XY(0.3),
                },
            ),
            OpenGraph(
                graph=nx.Graph([(0, 1), (0, 2), (0, 4), (1, 5), (2, 4), (2, 5), (3, 5)]),
                input_nodes=[0, 1],
                output_nodes=[4, 5],
                measurements={
                    0: Measurement.XY(0.1),
                    1: Measurement.XY(0.1),
                    2: Measurement.XZ(0.2),
                    3: Measurement.YZ(0.3),
                },
            ),
            OpenGraph(
                graph=nx.Graph([(0, 1), (1, 2), (1, 4), (2, 3)]),
                input_nodes=[0],
                output_nodes=[4],
                measurements={
                    0: Measurement.XY(0.1),  # XY
                    1: Measurement.X,  # X
                    2: Measurement.XY(0.1),  # XY
                    3: Measurement.X,  # X
                },
            ),
            OpenGraph(
                graph=nx.Graph([(0, 1), (1, 2)]),
                input_nodes=[0],
                output_nodes=[2],
                measurements={
                    0: Measurement.XY(0.1),  # XY
                    1: Measurement.Y,  # Y
                },
            ),
            OpenGraph(
                graph=nx.Graph([(0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 7)]),
                input_nodes=[0, 1],
                output_nodes=[6, 7],
                measurements={
                    0: Measurement.XY(0.1),  # XY
                    1: Measurement.XY(0.1),  # XY
                    2: Measurement.X,  # X
                    3: Measurement.XY(0.1),  # XY
                    4: Measurement.X,  # X
                    5: Measurement.Y,  # Y
                },
            ),
        ],
    )
    def test_extract_og(self, test_case: OpenGraph[Measurement]) -> None:
        pattern = test_case.to_pattern()
        cp = CompilationPass(LadderPass(), StimCliffordPass())
        circuit = (
            pattern.extract_opengraph().infer_pauli_measurements().extract_pauli_flow().extract_circuit().to_circuit(cp)
        )

        state = circuit.simulate_statevector().statevec
        state_ref = pattern.simulate_pattern()
        assert state.isclose(state_ref)

    def test_extract_og_gflow(self) -> None:
        og = OpenGraph(
            graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
            input_nodes=[1, 2],
            output_nodes=[5, 6],
            measurements={
                1: Measurement.XY(0.1),
                2: Measurement.XY(0.2),
                3: Measurement.XY(0.3),
                4: Measurement.XY(0.4),
            },
        )
        pattern = og.to_pattern()
        cp = CompilationPass(LadderPass(), StimCliffordPass())
        circuit = og.extract_gflow().extract_circuit().to_circuit(cp)

        state = circuit.simulate_statevector().statevec
        state_ref = pattern.simulate_pattern()
        assert state.isclose(state_ref)

    @pytest.mark.parametrize("test_case", [0.2, 0.5, 1.0])
    def test_parametric_angles(self, test_case: float) -> None:
        alpha = Placeholder("alpha")
        alpha_val = test_case
        flow = OpenGraph(
            graph=nx.Graph([(1, 3), (2, 4), (3, 4), (3, 5), (4, 6)]),
            input_nodes=[1, 2],
            output_nodes=[5, 6],
            measurements={
                1: Measurement.XY(0.1),
                2: Measurement.XY(alpha),
                3: Measurement.XY(0.3),
                4: Measurement.XY(alpha),
            },
        ).extract_pauli_flow()
        cp = CompilationPass(LadderPass(), StimCliffordPass())

        # Substitute parameter at the level of the extracted circuit
        qc1 = flow.extract_circuit().to_circuit(cp)
        s1 = qc1.subs(alpha, alpha_val).simulate_statevector().statevec

        # Substitute parameter at the level of the flow object
        qc2 = flow.subs(alpha, alpha_val).extract_circuit().to_circuit(cp)
        s2 = qc2.simulate_statevector().statevec

        assert s1.isclose(s2)
