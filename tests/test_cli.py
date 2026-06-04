import importlib
import sys
import types

import pytest

import alphafold3tools.cli as cli


def test_help_lists_all_subcommands(capsys):
    rc = cli.main([])
    out = capsys.readouterr().out
    assert rc == 0
    for name in cli.SUBCOMMANDS:
        assert name in out


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_flags(capsys, flag):
    assert cli.main([flag]) == 0
    assert "subcommand" in capsys.readouterr().out.lower()


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_version(capsys, flag):
    from alphafold3tools import __version__

    assert cli.main([flag]) == 0
    assert __version__ in capsys.readouterr().out


def test_unknown_subcommand(capsys):
    rc = cli.main(["does-not-exist"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown subcommand" in err


def test_map_modules_match_names():
    # Every mapped module is importable and exposes a callable main().
    for name, modpath in cli.SUBCOMMANDS.items():
        mod = importlib.import_module(modpath)
        assert callable(mod.main), f"{name} -> {modpath} has no callable main()"


def test_dispatch_rewrites_argv_and_delegates(monkeypatch):
    # Stub a subcommand module to assert argv rewrite + delegation without
    # running real (heavy) logic.
    captured = {}
    fake = types.ModuleType("alphafold3tools.fake")

    def fake_main():
        captured["argv"] = list(sys.argv)

    fake.main = fake_main
    monkeypatch.setitem(sys.modules, "alphafold3tools.fake", fake)
    monkeypatch.setitem(cli.SUBCOMMANDS, "fakecmd", "alphafold3tools.fake")
    monkeypatch.setattr(sys, "argv", ["af3tools"])  # protect the real argv
    cli.main(["fakecmd", "-i", "x"])
    assert captured["argv"] == ["af3tools fakecmd", "-i", "x"]


def test_subcommand_help_shows_prefixed_prog(capsys, monkeypatch):
    # Delegating -h must exit (argparse SystemExit) and show "af3tools <cmd>".
    # fastatojson is the lightest real module (no matplotlib/rdkit/gemmi/numpy).
    monkeypatch.setattr(sys, "argv", ["af3tools"])  # protect the real argv
    with pytest.raises(SystemExit) as ei:
        cli.main(["fastatojson", "-h"])
    assert ei.value.code == 0
    assert "af3tools fastatojson" in capsys.readouterr().out
