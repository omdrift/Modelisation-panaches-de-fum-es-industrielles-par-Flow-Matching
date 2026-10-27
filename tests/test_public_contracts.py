"""Public signatures and CLI help remain stable across internal refactoring."""
import ast
import json
from pathlib import Path
import sys

import pytest

from scientific import cli
from scientific.pipeline import source_fingerprint
from scientific.protocol import _source_manifest, digest, file_hash


CONTRACTS = Path(__file__).parent / 'contracts'
PUBLIC_API = json.loads((CONTRACTS / 'public_api.json').read_text())
CLI_HELP = json.loads((CONTRACTS / 'cli_help.json').read_text())


@pytest.mark.parametrize('module', PUBLIC_API)
def test_public_function_class_and_method_signatures(module):
    def signature(node):
        return dict(arguments=ast.unparse(node.args), returns=ast.unparse(node.returns) if node.returns else None)

    definitions = {}
    for node in ast.parse(Path(module).read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
            definitions[node.name] = signature(node)
        if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
            definitions[node.name] = {
                method.name: signature(method) for method in node.body
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    assert definitions == PUBLIC_API[module]


@pytest.mark.parametrize('command', CLI_HELP)
def test_cli_help_and_arguments_are_unchanged(command, monkeypatch, capsys):
    # argparse normally derives this name from the module invocation.
    monkeypatch.setattr(sys, 'argv', ['cli.py'])
    monkeypatch.setenv('COLUMNS', '80')
    with pytest.raises(SystemExit) as exc:
        cli.main([*command.split(), '--help'])
    assert exc.value.code == 0
    assert capsys.readouterr().out == CLI_HELP[command]


def test_shared_manifest_preserves_relative_paths_and_fingerprint():
    expected = {str(path): file_hash(path) for folder in ('scientific', 'model', 'training', 'lutils')
                for path in sorted(Path(folder).rglob('*.py'))}
    assert _source_manifest() == _source_manifest(Path.cwd()) == expected
    assert source_fingerprint() == digest(expected)
