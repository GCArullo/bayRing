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


class _FakeNumpy(types.ModuleType):
    pi = 3.141592653589793
    newaxis = None

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
