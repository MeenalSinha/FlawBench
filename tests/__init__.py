# tests package marker — required so pytest's pythonpath="." resolves
# "from tests.conftest import ..." to this project's tests/ directory rather
# than any same-named package that may be installed in site-packages.
