import json
from pathlib import Path
from airline_market.cli.main import main


def test_cli_validation_exact_and_simulation(tmp_path):
    out = tmp_path / "spec"
    main(
        [
            "spec",
            "validate",
            "airline_market/spec/library/toy_hub.json",
            "--out",
            str(out),
        ]
    )
    assert json.loads((out / "results.json").read_text())["valid"]
    out = tmp_path / "dp"
    main(
        [
            "solve",
            "exact-dp",
            "airline_market/spec/library/cx_example_X.json",
            "--focal",
            "F",
            "--arm",
            "empty",
            "--out",
            str(out),
        ]
    )
    assert json.loads((out / "results.json").read_text())["value"] == 200.0
    out = tmp_path / "sim"
    main(
        [
            "simulate",
            "airline_market/spec/library/cx_example_X.json",
            "--seeds",
            "0:2",
            "--out",
            str(out),
        ]
    )
    assert len(json.loads((out / "results.json").read_text())["episodes"]) == 2
    assert len((tmp_path / "manifest.jsonl").read_text().splitlines()) == 3


def test_cli_compile_accepts_file_output(tmp_path):
    from airline_market.spec import load_spec

    path = tmp_path / "compiled.json"
    main(
        [
            "spec",
            "compile",
            "airline_market/spec/library/toy_hub.json",
            "--out",
            str(path),
        ]
    )
    compiled = json.loads(path.read_text())
    assert (
        compiled["schema"] == "airline-compiled-market/1"
        and len(compiled["A_global"]) == 4
    )
    assert (
        load_spec(path).name == "toy_hub"
        and (tmp_path / "compiled.metadata.json").exists()
    )
