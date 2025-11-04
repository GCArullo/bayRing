import sys
import types
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class _FakeArray(list):
    """Minimal list-backed array used to satisfy numpy calls in lightweight tests."""

    @property
    def shape(self):
        return (len(self),)

    @property
    def ndim(self):
        if self and isinstance(self[0], _FakeArray):
            return 2
        return 1

    def __getitem__(self, item):
        if isinstance(item, tuple):
            value = self
            for idx in item:
                value = value[idx]
            return value

        result = super().__getitem__(item)
        if isinstance(item, slice):
            return _FakeArray(result)
        return result

    @property
    def T(self):
        if self.ndim == 1:
            return _FakeArray(self)
        rows = list(zip(*[list(row) for row in self]))
        return _FakeArray([_FakeArray(row) for row in rows])


class _FakeRandom:
    def uniform(self, low, high, size):
        import random

        return _FakeArray(random.uniform(low, high) for _ in range(size))


class _FakeNumpy(types.ModuleType):
    pi = 3.141592653589793
    newaxis = None

    def __init__(self, name: str):
        super().__init__(name)
        self.random = types.SimpleNamespace(uniform=_FakeRandom().uniform)

    def array(self, data: Iterable):
        if isinstance(data, _FakeArray):
            return _FakeArray(data)
        return _FakeArray(list(data))

    def insert(self, arr, index: int, value):
        values = list(arr)
        values.insert(index, value)
        return _FakeArray(values)

    def append(self, arr, value):
        values = list(arr)
        values.append(value)
        return _FakeArray(values)

    def linspace(self, start: float, stop: float, num: int):
        if num == 1:
            return _FakeArray([start])
        step = (stop - start) / (num - 1)
        return _FakeArray(start + i * step for i in range(num))

    def sum(self, values):
        return sum(values)

    def column_stack(self, values):
        if not values:
            return _FakeArray([])
        if isinstance(values[0], (list, tuple, _FakeArray)):
            rows = zip(*values)
            return _FakeArray(_FakeArray(row) for row in rows)
        return _FakeArray(values)

    def savetxt(self, filename, array, fmt="%s", header=""):
        data_lines = []
        if header:
            data_lines.append(f"# {header}")
        if isinstance(array, _FakeArray):
            iterable = array
        else:
            iterable = array
        for row in iterable:
            if isinstance(row, (list, tuple, _FakeArray)):
                values = row
            else:
                values = [row]
            data_lines.append(" ".join(fmt % value for value in values))
        content = "\n".join(data_lines) + "\n"
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write(content)

    def loadtxt(self, file_like):
        if hasattr(file_like, "read"):
            content = file_like.read()
        else:
            with open(file_like, "r", encoding="utf-8") as fh:
                content = fh.read()

        rows = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(_FakeArray(float(value) for value in line.split()))

        if not rows:
            return _FakeArray([])
        return _FakeArray(rows)


class _FakeTesting:
    @staticmethod
    def assert_allclose(actual, desired, rtol=1e-7, atol=0.0):
        assert len(actual) == len(desired)
        for a, b in zip(actual, desired):
            diff = abs(a - b)
            tolerance = atol + rtol * abs(b)
            assert diff <= tolerance, f"{a} !~= {b} (diff={diff}, tol={tolerance})"


if "numpy" not in sys.modules:
    fake_numpy = _FakeNumpy("numpy")
    fake_numpy.testing = _FakeTesting()
    sys.modules["numpy"] = fake_numpy


if "pandas" not in sys.modules:
    fake_pandas = types.ModuleType("pandas")
    fake_pandas.DataFrame = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
    sys.modules["pandas"] = fake_pandas


if "cpnest" not in sys.modules:
    fake_cpnest = types.ModuleType("cpnest")
    fake_cpnest_model = types.ModuleType("cpnest.model")
    fake_nest2pos = types.ModuleType("cpnest.nest2pos")

    def _fake_draw_posterior(chain, weights):
        return {name: values for name, values in chain.items()}

    def _fake_compute_weights(logL, nlive):
        return logL, [1] * len(logL)

    fake_nest2pos.draw_posterior = _fake_draw_posterior
    fake_nest2pos.compute_weights = _fake_compute_weights

    fake_cpnest.model = fake_cpnest_model

    sys.modules["cpnest"] = fake_cpnest
    sys.modules["cpnest.model"] = fake_cpnest_model
    sys.modules["cpnest.nest2pos"] = fake_nest2pos


if "pyRing" not in sys.modules:
    fake_pyRing = types.ModuleType("pyRing")
    fake_pyRing_utils = types.ModuleType("pyRing.utils")

    def _fake_print_section(message):
        return message

    def _fake_railing_check(samples, prior_bins, tolerance):
        return False, False

    fake_pyRing_utils.print_section = _fake_print_section
    fake_pyRing_utils.railing_check = _fake_railing_check

    fake_pyRing.utils = fake_pyRing_utils

    sys.modules["pyRing"] = fake_pyRing
    sys.modules["pyRing.utils"] = fake_pyRing_utils


if "scipy" not in sys.modules:
    fake_scipy = types.ModuleType("scipy")
    fake_optimize = types.ModuleType("scipy.optimize")
    fake_interpolate = types.ModuleType("scipy.interpolate")
    fake_signal = types.ModuleType("scipy.signal")

    def _fake_least_squares(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    def _fake_minimize(*args, **kwargs):
        return {"args": args, "kwargs": kwargs}

    def _fake_interp1d(*args, **kwargs):
        def _inner(x):
            return x

        return _inner

    def _fake_find_peaks(*args, **kwargs):
        return [], {}

    fake_optimize.least_squares = _fake_least_squares
    fake_optimize.minimize = _fake_minimize
    fake_optimize.fmin = _fake_minimize
    fake_interpolate.interp1d = _fake_interp1d
    fake_signal.find_peaks = _fake_find_peaks
    fake_scipy.optimize = fake_optimize
    fake_scipy.interpolate = fake_interpolate
    fake_scipy.signal = fake_signal

    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.optimize"] = fake_optimize
    sys.modules["scipy.interpolate"] = fake_interpolate
    sys.modules["scipy.signal"] = fake_signal


if "corner" not in sys.modules:
    fake_corner = types.ModuleType("corner")
    fake_corner.corner = lambda *args, **kwargs: None
    sys.modules["corner"] = fake_corner


if "h5py" not in sys.modules:
    class _FakeH5File(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self["combined"] = {"posterior_samples": []}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_h5py = types.ModuleType("h5py")
    fake_h5py.File = lambda *args, **kwargs: _FakeH5File()
    sys.modules["h5py"] = fake_h5py


if "matplotlib" not in sys.modules:
    fake_matplotlib = types.ModuleType("matplotlib")
    fake_pyplot = types.ModuleType("matplotlib.pyplot")

    def _fake_fig(*args, **kwargs):
        return types.SimpleNamespace()

    def _fake_subplots(*args, **kwargs):
        return _fake_fig(), []

    fake_pyplot.figure = _fake_fig
    fake_pyplot.subplots = _fake_subplots
    fake_pyplot.savefig = lambda *args, **kwargs: None
    fake_pyplot.close = lambda *args, **kwargs: None
    fake_pyplot.tight_layout = lambda *args, **kwargs: None
    fake_pyplot.rcParams = {}

    fake_matplotlib.pyplot = fake_pyplot

    sys.modules["matplotlib"] = fake_matplotlib
    sys.modules["matplotlib.pyplot"] = fake_pyplot


if "qnm" not in sys.modules:
    fake_qnm = types.ModuleType("qnm")
    fake_qnm.modes_cache = lambda *args, **kwargs: (lambda **inner_kwargs: types.SimpleNamespace())
    sys.modules["qnm"] = fake_qnm


if "seaborn" not in sys.modules:
    fake_seaborn = types.ModuleType("seaborn")
    fake_seaborn.set_style = lambda *args, **kwargs: None
    sys.modules["seaborn"] = fake_seaborn
