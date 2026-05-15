import cmath
import math
import operator
import sys
import types
from pathlib import Path
from typing import Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class _FakeArray(list):
    """Minimal list-backed array used to satisfy numpy calls in lightweight tests."""

    __array_priority__ = 1000

    def _binary_op(self, other, op):
        if isinstance(other, _FakeArray):
            return _FakeArray(op(a, b) for a, b in zip(self, other))
        if isinstance(other, Iterable) and not isinstance(other, (str, bytes)):
            return _FakeArray(op(a, b) for a, b in zip(self, other))
        return _FakeArray(op(a, other) for a in self)

    def _rbinary_op(self, other, op):
        if isinstance(other, Iterable) and not isinstance(other, (str, bytes)):
            return _FakeArray(op(a, b) for a, b in zip(other, self))
        return _FakeArray(op(other, a) for a in self)

    def __add__(self, other):
        return self._binary_op(other, operator.add)

    def __radd__(self, other):
        return self._rbinary_op(other, operator.add)

    def __sub__(self, other):
        return self._binary_op(other, operator.sub)

    def __rsub__(self, other):
        return self._rbinary_op(other, operator.sub)

    def __mul__(self, other):
        return self._binary_op(other, operator.mul)

    def __rmul__(self, other):
        return self._rbinary_op(other, operator.mul)

    def __truediv__(self, other):
        return self._binary_op(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._rbinary_op(other, operator.truediv)

    def __pow__(self, power):
        return self._binary_op(power, operator.pow)

    def __rpow__(self, other):
        return self._rbinary_op(other, operator.pow)

    def __neg__(self):
        return _FakeArray(-a for a in self)

    def __gt__(self, other):
        return self._binary_op(other, operator.gt)

    def __lt__(self, other):
        return self._binary_op(other, operator.lt)

    def __ge__(self, other):
        return self._binary_op(other, operator.ge)

    def __le__(self, other):
        return self._binary_op(other, operator.le)

    def __eq__(self, other):
        return self._binary_op(other, operator.eq)

    def __ne__(self, other):
        return self._binary_op(other, operator.ne)

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

        if isinstance(item, (list, tuple, _FakeArray)) and item and isinstance(item[0], bool):
            return _FakeArray(value for value, keep in zip(self, item) if keep)

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
    complex128 = complex

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

    def zeros(self, length, dtype=float):
        factory = 0
        if dtype is complex:
            factory = 0j
        return _FakeArray(factory for _ in range(length))

    def linspace(self, start: float, stop: float, num: int):
        if num == 1:
            return _FakeArray([start])
        step = (stop - start) / (num - 1)
        return _FakeArray(start + i * step for i in range(num))

    def sum(self, values):
        return sum(values)

    def max(self, values):
        return max(values)

    def abs(self, values):
        if isinstance(values, _FakeArray):
            return _FakeArray(abs(v) for v in values)
        return abs(values)

    def sqrt(self, values):
        if isinstance(values, _FakeArray):
            return _FakeArray(math.sqrt(v) for v in values)
        return math.sqrt(values)

    def angle(self, values):
        if isinstance(values, _FakeArray):
            return _FakeArray(math.atan2(v.imag if isinstance(v, complex) else 0.0, v.real if isinstance(v, complex) else v) for v in values)
        if isinstance(values, complex):
            return math.atan2(values.imag, values.real)
        return 0.0

    def unwrap(self, values):
        if not values:
            return _FakeArray([])
        unwrapped = []
        prev = None
        for value in values:
            current = value
            if prev is not None:
                while current - prev > math.pi:
                    current -= 2 * math.pi
                while current - prev < -math.pi:
                    current += 2 * math.pi
            unwrapped.append(current)
            prev = current
        return _FakeArray(unwrapped)

    def logical_and(self, left, right):
        return _FakeArray(bool(a and b) for a, b in zip(left, right))

    def real(self, values):
        if isinstance(values, _FakeArray):
            return _FakeArray((v.real if isinstance(v, complex) else v) for v in values)
        return values.real if isinstance(values, complex) else values

    def imag(self, values):
        if isinstance(values, _FakeArray):
            return _FakeArray((v.imag if isinstance(v, complex) else 0.0) for v in values)
        return values.imag if isinstance(values, complex) else 0.0

    def exp(self, values):
        if isinstance(values, _FakeArray):
            return _FakeArray(cmath.exp(v) for v in values)
        return cmath.exp(values)

    def cos(self, values):
        if isinstance(values, _FakeArray):
            return _FakeArray(math.cos(v) for v in values)
        return math.cos(values)

    def sin(self, values):
        if isinstance(values, _FakeArray):
            return _FakeArray(math.sin(v) for v in values)
        return math.sin(values)

    def argmax(self, values):
        return max(range(len(values)), key=lambda idx: values[idx])

    def roll(self, values, shift):
        seq = list(values)
        shift = shift % len(seq) if seq else 0
        return _FakeArray(seq[-shift:] + seq[:-shift])

    def asarray(self, value):
        return self.array(value)

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

    class _FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def log_likelihood(self, *args, **kwargs):
            return 0.0

    def _fake_draw_posterior(chain, weights):
        return {name: values for name, values in chain.items()}

    def _fake_compute_weights(logL, nlive):
        return logL, [1] * len(logL)

    fake_nest2pos.draw_posterior = _fake_draw_posterior
    fake_nest2pos.compute_weights = _fake_compute_weights

    fake_cpnest.model = fake_cpnest_model
    fake_cpnest_model.Model = _FakeModel

    sys.modules["cpnest"] = fake_cpnest
    sys.modules["cpnest.model"] = fake_cpnest_model
    sys.modules["cpnest.nest2pos"] = fake_nest2pos


if "pyRing" not in sys.modules:
    fake_pyRing = types.ModuleType("pyRing")
    fake_pyRing_utils = types.ModuleType("pyRing.utils")
    fake_pyRing_waveform = types.ModuleType("pyRing.waveform")

    def _fake_print_section(message):
        return message

    def _fake_railing_check(samples, prior_bins, tolerance):
        return False, False

    def _fake_compute_binary_quantities(m1, m2, chi1, chi2):
        return 1.0, 0.25, 0.1, -0.1

    fake_modes = {
        "linear": {(2, 2): ["mode"]},
        "quadratic": {(2, 2): ["quad"]},
    }

    fake_pyRing_utils.print_section = _fake_print_section
    fake_pyRing_utils.railing_check = _fake_railing_check
    fake_pyRing_utils.compute_KerrBinary_binary_quantities = _fake_compute_binary_quantities
    fake_pyRing_utils.available_modes_dict_KerrBinary = {"London2018": fake_modes, "Carullo2024": fake_modes}

    fake_pyRing_waveform.KerrBH = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}
    fake_pyRing_waveform.damped_sinusoid = lambda *args, **kwargs: 0j
    fake_pyRing_waveform.KerrBinary = lambda *args, **kwargs: {"args": args, "kwargs": kwargs}

    fake_pyRing.utils = fake_pyRing_utils
    fake_pyRing.waveform = fake_pyRing_waveform

    sys.modules["pyRing"] = fake_pyRing
    sys.modules["pyRing.utils"] = fake_pyRing_utils
    sys.modules["pyRing.waveform"] = fake_pyRing_waveform


if "scipy" not in sys.modules:
    fake_scipy = types.ModuleType("scipy")
    fake_optimize = types.ModuleType("scipy.optimize")
    fake_interpolate = types.ModuleType("scipy.interpolate")
    fake_linalg = types.ModuleType("scipy.linalg")
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
    fake_linalg.toeplitz = lambda values: _FakeArray(_FakeArray(values) for _ in values)
    fake_linalg.solve_toeplitz = lambda acf, values, check_finite=True: values
    fake_signal.find_peaks = _fake_find_peaks
    fake_scipy.optimize = fake_optimize
    fake_scipy.interpolate = fake_interpolate
    fake_scipy.linalg = fake_linalg
    fake_scipy.signal = fake_signal

    sys.modules["scipy"] = fake_scipy
    sys.modules["scipy.optimize"] = fake_optimize
    sys.modules["scipy.interpolate"] = fake_interpolate
    sys.modules["scipy.linalg"] = fake_linalg
    sys.modules["scipy.signal"] = fake_signal


if "pycbc" not in sys.modules:
    fake_pycbc = types.ModuleType("pycbc")
    fake_pycbc_psd = types.ModuleType("pycbc.psd")
    fake_pycbc_types = types.ModuleType("pycbc.types")
    fake_pycbc_timeseries = types.ModuleType("pycbc.types.timeseries")
    fake_pycbc_frequencyseries = types.ModuleType("pycbc.types.frequencyseries")
    fake_pycbc_filter = types.ModuleType("pycbc.filter")

    class _FakeSeries(_FakeArray):
        def __init__(self, values=None, **kwargs):
            super().__init__(values or [])
            self.__dict__.update(kwargs)

        def to_frequencyseries(self, delta_f=None):
            return _FakeSeries(self, delta_f=delta_f)

    fake_pycbc_psd.from_txt = lambda *args, **kwargs: _FakeSeries()
    fake_pycbc_timeseries.TimeSeries = _FakeSeries
    fake_pycbc_frequencyseries.FrequencySeries = _FakeSeries
    fake_pycbc_types.TimeSeries = _FakeSeries
    fake_pycbc_types.FrequencySeries = _FakeSeries
    fake_pycbc_filter.sigma = lambda *args, **kwargs: 0.0
    fake_pycbc_filter.match = lambda *args, **kwargs: (0.0, 0)

    fake_pycbc.psd = fake_pycbc_psd
    fake_pycbc.types = fake_pycbc_types
    fake_pycbc.filter = fake_pycbc_filter

    sys.modules["pycbc"] = fake_pycbc
    sys.modules["pycbc.psd"] = fake_pycbc_psd
    sys.modules["pycbc.types"] = fake_pycbc_types
    sys.modules["pycbc.types.timeseries"] = fake_pycbc_timeseries
    sys.modules["pycbc.types.frequencyseries"] = fake_pycbc_frequencyseries
    sys.modules["pycbc.filter"] = fake_pycbc_filter


if "lal" not in sys.modules:
    fake_lal = types.ModuleType("lal")
    fake_lal_antenna = types.ModuleType("lal.antenna")

    class _FakeAntennaResponse:
        def __init__(self, *args, **kwargs):
            pass

    fake_lal.MSUN_SI = 1.0
    fake_lal.G_SI = 1.0
    fake_lal.C_SI = 1.0
    fake_lal.PC_SI = 1.0
    fake_lal_antenna.AntennaResponse = _FakeAntennaResponse
    fake_lal.antenna = fake_lal_antenna

    sys.modules["lal"] = fake_lal
    sys.modules["lal.antenna"] = fake_lal_antenna


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
    fake_qnm.download_data = lambda: None
    sys.modules["qnm"] = fake_qnm


if "seaborn" not in sys.modules:
    fake_seaborn = types.ModuleType("seaborn")
    fake_seaborn.set_style = lambda *args, **kwargs: None
    sys.modules["seaborn"] = fake_seaborn


if "numba" not in sys.modules:
    fake_numba = types.ModuleType("numba")

    def _fake_decorator(*args, **kwargs):
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]
        return lambda function: function

    fake_numba.njit = _fake_decorator
    fake_numba.jit = _fake_decorator
    sys.modules["numba"] = fake_numba
