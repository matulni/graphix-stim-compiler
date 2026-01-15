# graphix-stim-compiler

Clifford compilation pass using stim functionalities.

This compilation pass together with the new tools implemented in `matulni/graphix.git@circuit-extraction` allow to do a round-trip conversion circuit -> MBQC pattern -> circuit.

### Example
```python
from graphix.circ_ext.compilation import CompilationPass, LadderPass
from graphix_stim_compiler import StimCliffordPass

from graphix.transpiler import Circuit
from graphix.fundamentals import ANGLE_PI

import numpy as np

qc = Circuit(2)
qc.cnot(1, 0)
qc.rz(0, 0.2 * ANGLE_PI)
qc.h(0)
qc.rx(1, 0.2 * ANGLE_PI)

pattern = qc.transpile().pattern

cp = CompilationPass(LadderPass(), StimCliffordPass())
qc_extracted = pattern.extract_opengraph().extract_pauli_flow().extract_circuit().to_circuit(cp)

s_ref = qc.simulate_statevector().statevec
s_extracted = qc_extracted.simulate_statevector().statevec
fidelity = np.abs(np.dot(s_ref.flatten().conjugate(), s_extracted.flatten()))
assert np.isclose(fidelity, 1)
```


### Installation

`git clone` the repository and run

```bash
cd graphix-stim-compiler
pip install .
```


