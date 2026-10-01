import logging
import multiprocessing
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def fix_streams() -> None:
    """A windowed exe has no console; libraries that write to stdout or stderr would crash without these."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))


def setup_logging() -> Path:
    from common.settings import ROOT_DIR

    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"
    handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler], format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    for name in ("httpx", "huggingface_hub", "urllib3", "transformers", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    return log_file


def selftest(args: list[str]) -> int:
    """Headless check of a built exe: `OUTDIR photo... [--ramp view...]` analyzes the photos and writes the outcome next to the results."""
    from gui.pipeline import Inputs, build_input_folder, friendly_error, run_job
    from gui.settings_store import AppSettings

    output, rest = Path(args[0]), args[1:]
    split = rest.index("--ramp") if "--ramp" in rest else len(rest)
    inputs = Inputs([Path(a) for a in rest[:split]], [Path(a) for a in rest[split + 1:]])
    output.mkdir(parents=True, exist_ok=True)
    report = output / "selftest.txt"
    missing = [str(p) for p in inputs.main + inputs.ramp if not p.is_file()]
    if missing:
        report.write_text("FAILED\nThese photos do not exist, use real paths:\n" + "\n".join(missing) + "\n", encoding="utf-8")
        return 1
    try:
        folder = build_input_folder(inputs, "selftest")
        result = run_job(folder, "", AppSettings(), lambda fraction, text: report.write_text(f"{fraction:.0%} {text}\n", encoding="utf-8"), output)
        report.write_text(f"OK\nfindings: {result.findings}\nwarnings: {result.warnings}\nimages: {[str(p) for p in result.images]}\n", encoding="utf-8")
        return 0
    except Exception as exc:
        logging.getLogger("selftest").exception("Self-test failed")
        report.write_text(f"FAILED\n{friendly_error(exc)}\n{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return 1


def main() -> int:
    fix_streams()
    setup_logging()
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        return selftest(sys.argv[2:])
    from gui.window import run_gui

    return run_gui()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
