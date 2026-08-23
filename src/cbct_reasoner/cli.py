"""Command line interface.

``cbct-reasoner run-all`` executes the whole pipeline; every stage is also
available on its own so a long run can be resumed or moved between machines.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from cbct_reasoner.config import ExperimentConfig, Paths, default_paths, env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cbct-reasoner",
        description="ToothFairy4 CBCT-to-report pipeline (ODIN 2026 Task 1)",
    )
    parser.add_argument("--config", type=Path, help="experiment configuration JSON")
    parser.add_argument("--data", type=Path, help="dataset root (overrides TOOTHFAIRY_DATA)")
    parser.add_argument(
        "--work",
        type=Path,
        help="workspace root; work/ and artifacts/ are created inside it",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="report what a dataset directory contains")
    inspect.add_argument("--json", action="store_true", help="emit machine-readable output")

    doctor = commands.add_parser("doctor", help="check the environment and credentials")
    doctor.add_argument("--json", action="store_true")

    synthetic = commands.add_parser(
        "synthetic", help="generate a synthetic dataset for rehearsing the pipeline"
    )
    synthetic.add_argument("--output", type=Path, required=True)
    synthetic.add_argument("--cases", type=int, default=24)
    synthetic.add_argument("--seed", type=int, default=7)

    index = commands.add_parser("index", help="validate a dataset and write a case manifest")
    index.add_argument("--output", type=Path)

    prepare = commands.add_parser("prepare", help="build the corpus and cache normalized volumes")
    prepare.add_argument("--force", action="store_true", help="rebuild cached volumes")
    prepare.add_argument(
        "--skip-errors", action="store_true", help="continue past unreadable volumes"
    )
    prepare.add_argument("--limit", type=int, help="only process the first N cases")

    commands.add_parser("prototypes", help="build the sentence-prototype label space")

    splits = commands.add_parser("splits", help="write cross-validation folds")
    splits.add_argument("--strategy", choices=("stratified", "center"), default="stratified")

    train = commands.add_parser("train", help="train the finding predictor")
    train.add_argument("--folds", type=int, nargs="*", help="only train these fold indices")
    train.add_argument("--device", help="cuda, cpu, or mps")
    train.add_argument("--epochs", type=int, help="override the configured epoch count")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--backbone", choices=("slice2d", "resnet3d"))

    calibrate = commands.add_parser(
        "calibrate", help="fit decoder thresholds on out-of-fold scores"
    )
    calibrate.add_argument(
        "--prior-only", action="store_true", help="calibrate without an image model"
    )
    calibrate.add_argument("--rounds", type=int)
    calibrate.add_argument("--oof", type=Path, help="alternative out-of-fold probability file")
    calibrate.add_argument(
        "--init-decoder", type=Path, help="seed the threshold ascent with these thresholds"
    )

    evaluate = commands.add_parser("evaluate", help="score predictions")
    evaluate.add_argument("--pairs", type=Path, help="score a prediction/reference JSONL instead")
    evaluate.add_argument("--prior-only", action="store_true")
    evaluate.add_argument("--radfact-lite", action="store_true", help="use the real LLM metric")
    evaluate.add_argument("--radfact-model")
    evaluate.add_argument("--radfact-provider", choices=("openai", "ollama"), default="openai")
    evaluate.add_argument("--radfact-base-url")
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--oof", type=Path)

    commands.add_parser("ablation", help="score the prior against the trained model")
    commands.add_parser("plots", help="render every diagnostic figure")

    package = commands.add_parser("package", help="assemble the submission bundle")
    package.add_argument("--no-checkpoints", action="store_true", help="omit fold checkpoints")
    package.add_argument("--shallow", type=Path, help="ship this linear model in the bundle")

    run_all = commands.add_parser(
        "run-all", help="prepare, prototypes, splits, train, calibrate, package"
    )
    run_all.add_argument(
        "--skip-train", action="store_true", help="calibrate the prior decoder only"
    )
    run_all.add_argument("--skip-errors", action="store_true")
    run_all.add_argument("--device")
    run_all.add_argument("--epochs", type=int)
    run_all.add_argument("--strategy", choices=("stratified", "center"), default="stratified")
    run_all.add_argument("--push", action="store_true", help="push artifacts to the Hub when done")

    predict = commands.add_parser("predict", help="generate one report from a CBCT volume")
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument(
        "--output", type=Path, default=Path("predictions/diagnostic-imaging-report.json")
    )
    predict.add_argument("--bundle", type=Path)

    validate = commands.add_parser("validate-output", help="validate challenge output JSON")
    validate.add_argument("path", type=Path)

    hub = commands.add_parser("hub", help="push or pull artifacts on the Hugging Face Hub")
    hub.add_argument("action", choices=("push", "pull", "whoami"))
    hub.add_argument("--namespace")
    hub.add_argument("--project", default="cbct-clinical-reasoner")
    hub.add_argument(
        "--public", action="store_true", help="publish publicly (see the data agreement)"
    )
    hub.add_argument("--no-data", action="store_true", help="skip the derived-data repository")
    hub.add_argument("--destination", type=Path)
    return parser


def _resolve(args: argparse.Namespace) -> tuple[Paths, ExperimentConfig]:
    paths = default_paths()
    if args.work:
        paths = paths.with_root(args.work)
    if args.data:
        paths = replace(paths, data=Path(args.data))
    config = ExperimentConfig.load(args.config)
    return paths, config


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except (
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
        ImportError,
        json.JSONDecodeError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace) -> int:
    paths, config = _resolve(args)
    command = args.command

    if command == "inspect":
        return _inspect(paths, args)
    if command == "doctor":
        return _doctor(paths, args)
    if command == "synthetic":
        from cbct_reasoner.data.synthetic import generate_dataset

        root = generate_dataset(args.output, num_cases=args.cases, seed=args.seed)
        _emit({"dataset": str(root), "cases": args.cases})
        return 0
    if command == "index":
        from cbct_reasoner.data.discovery import discover_cases, write_manifest

        records = discover_cases(paths.data)
        destination = args.output or paths.manifest
        write_manifest(records, destination)
        _emit({"cases": len(records), "manifest": str(destination)})
        return 0
    if command == "validate-output":
        from cbct_reasoner.reporting import read_challenge_output

        report = read_challenge_output(args.path)
        _emit({"valid": True, "characters": len(report)})
        return 0
    if command == "predict":
        return _predict(paths, args)
    if command == "hub":
        return _hub(paths, args)

    from cbct_reasoner import pipeline

    if command == "prepare":
        _emit(
            pipeline.prepare(
                paths, config, force=args.force, skip_errors=args.skip_errors, limit=args.limit
            )
        )
        return 0
    if command == "prototypes":
        _emit(pipeline.prototypes(paths, config))
        return 0
    if command == "splits":
        _emit(pipeline.splits(paths, config, strategy=args.strategy))
        return 0
    if command == "train":
        config = _override_encoder(config, args)
        _emit(pipeline.train(paths, config, only_folds=args.folds or None, device=args.device))
        return 0
    if command == "calibrate":
        _emit(
            pipeline.calibrate_decoder(
                paths,
                config,
                prior_only=args.prior_only,
                rounds=args.rounds,
                oof=args.oof,
                init_decoder=args.init_decoder,
            )
        )
        return 0
    if command == "evaluate":
        if args.pairs:
            _emit(_evaluate_jsonl(args.pairs))
            return 0
        options: dict[str, Any] = {}
        if args.radfact_model:
            options["model"] = args.radfact_model
        if args.radfact_provider:
            options["provider"] = args.radfact_provider
        base_url = args.radfact_base_url or env("RADFACT_BASE_URL")
        if base_url:
            options["base_url"] = base_url
        _emit(
            pipeline.evaluate(
                paths,
                config,
                prior_only=args.prior_only,
                use_radfact_lite=args.radfact_lite,
                radfact_options=options,
                output=args.output,
                oof=args.oof,
            )
        )
        return 0
    if command == "ablation":
        _emit(pipeline.ablation(paths, config))
        return 0
    if command == "plots":
        _emit(pipeline.figures(paths, config))
        return 0
    if command == "package":
        _emit(
            pipeline.package(
                paths,
                config,
                include_checkpoints=not args.no_checkpoints,
                shallow=args.shallow,
            )
        )
        return 0
    if command == "run-all":
        return _run_all(paths, config, args)

    raise AssertionError(f"Unhandled command: {command}")


def _override_encoder(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    encoder = config.encoder
    if getattr(args, "epochs", None):
        encoder = replace(encoder, epochs=args.epochs)
    if getattr(args, "batch_size", None):
        encoder = replace(encoder, batch_size=args.batch_size)
    if getattr(args, "backbone", None):
        encoder = replace(encoder, backbone=args.backbone)
    return replace(config, encoder=encoder)


def _run_all(paths: Paths, config: ExperimentConfig, args: argparse.Namespace) -> int:
    from cbct_reasoner import pipeline

    config = _override_encoder(config, args)
    summary: dict[str, Any] = {}
    summary["prepare"] = pipeline.prepare(paths, config, skip_errors=args.skip_errors)
    summary["prototypes"] = pipeline.prototypes(paths, config)
    summary["splits"] = pipeline.splits(paths, config, strategy=args.strategy)
    if not args.skip_train:
        summary["train"] = pipeline.train(paths, config, device=args.device)
    summary["calibrate"] = pipeline.calibrate_decoder(paths, config, prior_only=args.skip_train)
    summary["evaluate"] = pipeline.evaluate(paths, config, prior_only=args.skip_train)
    summary["ablation"] = pipeline.ablation(paths, config)
    summary["figures"] = pipeline.figures(paths, config)
    summary["package"] = pipeline.package(paths, config, include_checkpoints=not args.skip_train)

    (paths.artifacts / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    if args.push:
        from cbct_reasoner.hub import push_all

        summary["hub"] = push_all(paths, summary=summary)
    _emit({"stages": list(summary), "final": summary["evaluate"]["surrogate"]["final"]})
    return 0


def _inspect(paths: Paths, args: argparse.Namespace) -> int:
    from cbct_reasoner.data.discovery import inspect_layout

    if not paths.data.exists():
        message = (
            f"Dataset directory {paths.data} does not exist yet.\n"
            "Paste the ToothFairy4 release there (or pass --data /path/to/release), then re-run.\n"
            "Expected: {CASE_ID}/cbct/volume.nii.gz and {CASE_ID}/reports_en/*.txt"
        )
        if args.json:
            _emit({"ok": False, "reason": "missing", "expected_root": str(paths.data)})
        else:
            print(message)
        return 1
    report = inspect_layout(paths.data)
    if args.json:
        _emit(report.to_dict())
    else:
        print(report.render())
    return 0 if report.ok else 1


def _doctor(paths: Paths, args: argparse.Namespace) -> int:
    import importlib.util
    import platform

    def version(module: str) -> str | None:
        try:
            import importlib.metadata as metadata

            return metadata.version(module)
        except Exception:
            return "installed" if importlib.util.find_spec(module.replace("-", "_")) else None

    packages = {
        name: version(name)
        for name in (
            "numpy",
            "SimpleITK",
            "torch",
            "timm",
            "scikit-learn",
            "huggingface-hub",
            "modal",
            "nltk",
            "transformers",
            "radfact-lite",
        )
    }
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass

    status = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": cuda,
        "hf_token": bool(env("HF_TOKEN")),
        "openai_key": bool(env("OPENAI_API_KEY")),
        "data_root": str(paths.data),
        "data_root_exists": paths.data.exists(),
        "work_root": str(paths.work),
        "artifacts": {
            "corpus": paths.corpus.is_file(),
            "prototypes": paths.prototypes.is_file(),
            "folds": paths.folds.is_file(),
            "oof": paths.oof.is_file(),
            "decoder": paths.decoder.is_file(),
            "bundle": paths.bundle.is_dir(),
        },
    }
    required_missing = [name for name in ("numpy", "SimpleITK") if not packages.get(name)]
    status["ready_to_prepare"] = not required_missing and paths.data.exists()
    status["ready_to_train"] = bool(packages.get("torch")) and bool(packages.get("timm"))

    if args.json:
        _emit(status)
    else:
        print(f"python           : {status['python']}")
        for name, value in packages.items():
            print(f"{name:<17}: {value or 'MISSING'}")
        print(f"cuda available   : {cuda}")
        print(
            f"hf token         : {'set' if status['hf_token'] else 'MISSING (add hf=... to .env)'}"
        )
        print(
            f"dataset root     : {paths.data} ({'present' if paths.data.exists() else 'MISSING'})"
        )
        print(f"ready to prepare : {status['ready_to_prepare']}")
        print(f"ready to train   : {status['ready_to_train']}")
        for name, present in status["artifacts"].items():  # type: ignore[union-attr]
            print(f"artifact {name:<8}: {'yes' if present else 'no'}")
    return 0


def _predict(paths: Paths, args: argparse.Namespace) -> int:
    from cbct_reasoner.pipeline.bundle import InferenceBundle
    from cbct_reasoner.reporting import write_challenge_output

    bundle = InferenceBundle.load(args.bundle or paths.bundle)
    report = bundle.predict(args.input)
    write_challenge_output(report, args.output)
    _emit({"output": str(args.output), "characters": len(report)})
    return 0


def _hub(paths: Paths, args: argparse.Namespace) -> int:
    from cbct_reasoner.hub import HubClient, push_all

    if args.action == "whoami":
        client = HubClient.create(namespace=args.namespace, project=args.project)
        _emit({"namespace": client.config.namespace, "model_repo": client.config.model_repo})
        return 0
    if args.action == "push":
        if args.public:
            print(
                "warning: --public publishes verbatim clinical report sentences derived from the "
                "access-controlled ToothFairy4 release. Confirm the data agreement permits this.",
                file=sys.stderr,
            )
        urls = push_all(
            paths,
            namespace=args.namespace,
            project=args.project,
            private=not args.public,
            include_data=not args.no_data,
        )
        _emit(urls)
        return 0

    client = HubClient.create(namespace=args.namespace, project=args.project)
    destination = args.destination or paths.artifacts / "downloaded"
    client.download(client.config.model_repo, repo_type="model", destination=destination)
    _emit({"downloaded": str(destination)})
    return 0


def _evaluate_jsonl(path: Path) -> dict[str, float | int]:
    from cbct_reasoner.metrics import evaluate_pairs

    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} must be a JSON object")
            try:
                prediction, reference = row["prediction"], row["reference"]
            except KeyError as error:
                raise ValueError(f"line {line_number} lacks {error.args[0]!r}") from error
            if not isinstance(prediction, str) or not isinstance(reference, str):
                raise ValueError(f"line {line_number} prediction/reference must be strings")
            pairs.append((prediction, reference))
    return evaluate_pairs(pairs)


def _emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
