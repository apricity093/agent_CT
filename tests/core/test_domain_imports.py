import subprocess
import sys


def test_domain_imports_do_not_load_ct_or_agent():
    code = """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'inverse_agent', 'iteris', 'astra'} or fullname.startswith('inv_framework.ct'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from inv_framework.mri.api import list_capabilities as mri
from inv_framework.deconvolution.api import list_capabilities as deconv
import inv_framework.core.fidelity
assert len(mri()['algorithms']) == 14
assert len(deconv()['algorithms']) == 14
assert len(mri()['planned_algorithms']) == 0
assert len(deconv()['planned_algorithms']) == 0
"""
    subprocess.run([sys.executable, '-c', code], check=True)
