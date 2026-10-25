"""Load lightweight production modules without importing MMCV extensions."""
import ast
import copy
import importlib
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
package = ModuleType('_acl_diag_test')
package.__path__ = [str(ROOT / 'mmdet/utils')]
sys.modules.setdefault(package.__name__, package)
events = importlib.import_module(package.__name__ + '.acl_diagnostic_events')
diagnostic_io = importlib.import_module(package.__name__ + '.acl_diagnostic_io')
capture_module = importlib.import_module(package.__name__ + '.acl_diagnostic_capture')


def without_diagnostic_events(node):
    """Only remove additive taps; all original model math must still match."""
    class RemoveTaps(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and
                    isinstance(node.value.func, ast.Name) and
                    node.value.func.id == 'diagnostic_event'):
                return None
            return self.generic_visit(node)
        def visit_ImportFrom(self, node):
            if node.module == 'mmdet.utils.acl_diagnostic_events':
                return None
            return node
    return RemoveTaps().visit(copy.deepcopy(node))
