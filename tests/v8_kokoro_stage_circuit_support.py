"""Materialize a Kokoro stage in the test's own build directory."""

import json


def materialize_stage_circuit(directory, author):
    path = directory / author.OUTPUT.name
    path.write_text(json.dumps(author.build_circuit(), indent=2) + "\n",
                    encoding="utf-8")
    return path
