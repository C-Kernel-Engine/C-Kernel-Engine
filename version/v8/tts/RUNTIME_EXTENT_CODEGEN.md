# Bounded runtime extents in v8 codegen

`codegen_checked_calls_v8.emit_checked_calls()` consumes the ordinary resolved
IR Lower 3 operations and canonical kernel maps. The synthetic fixture currently
adds only the native host entry declaration (`entry`) and invokes this backend
component directly. It proves lowering, ABI order, checked provider status,
caller-owned arena, runtime lengths, and physical strides. It does **not** prove
that the normal `codegen_v8.py` command automatically emits the entry point.

The integration point is `codegen_v8.main()`, after it loads the call-ready IR
and layout and before it calls `codegen_core_v8.generate()`. For a graph with a
declared checked host entry and `runtime_extent_contract`, that wrapper should
pass the resolved call IR to `emit_checked_calls()` and write the resulting
native entry in the normal generated model artifact. The entry declaration
must come from the circuit/bundle host interface during lowering, rather than
from a test script. The existing core path remains responsible for graphs
without that contract. This integration needs a command-level generated-C test
before claiming complete Kokoro compilation.

The checked entry requires the caller's arena base to be aligned to at least
64 bytes, or more when a selected provider declares stronger alignment. It
checks the base and planned byte capacity before any provider executes.

X-Ray records two distinct forms of library evidence. `artifact_library` is
the on-disk file hash. `runtime_library` is an optional capture-time record
resolved from an invoked runtime symbol with `dladdr`, including its path and
hash. An artifact hash alone does not establish which library executed.
