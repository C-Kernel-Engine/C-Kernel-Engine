"""Model-neutral exact-ABI entry bindings reject untyped pointer coercions."""

import copy
import unittest
from version.v8.scripts.native_entry_binding_v8 import resolve_entry_argument


class NativeEntryBinding(unittest.TestCase):
    def setUp(self):
        self.arg = {"name": "context", "source": "entry:state"}
        self.entry = {"params": [{"name": "state", "c_type": "const void *"}]}
        self.kernel = {
            "impl": {"c_declaration": "int op(const void *context);"},
            "call_abi": {"params": [self.arg]},
        }

    def test_exact_pointer_binding(self):
        self.assertEqual(
            resolve_entry_argument(self.entry, self.kernel, self.arg), "state"
        )

    def test_bad_type_missing_duplicate_and_coercion(self):
        for params in (
            [],
            self.entry["params"] * 2,
            [{"name": "state", "c_type": "void *"}],
            [{"name": "state", "c_type": "size_t"}],
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                resolve_entry_argument({"params": params}, self.kernel, self.arg)
        for change in ({"cast": "const void *"}, {"source": "entry:state+1"}):
            arg = dict(self.arg, **change)
            kernel = copy.deepcopy(self.kernel)
            kernel["call_abi"]["params"] = [arg]
            with self.assertRaises(ValueError):
                resolve_entry_argument(self.entry, kernel, arg)

    def test_provider_declaration_mismatch(self):
        for declaration in ("int op(size_t context);", "int op(void);", "broken"):
            self.kernel["impl"]["c_declaration"] = declaration
            with self.assertRaises(ValueError):
                resolve_entry_argument(self.entry, self.kernel, self.arg)
