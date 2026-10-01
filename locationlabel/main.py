import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

from common.addresses import Job, read_jobs
from common.settings import output_dir

logger = logging.getLogger("locationlabel")


def add_source(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("address", nargs="?", help="a single US address")
    source.add_argument("--csv", type=Path, help="CSV file with an 'address' column (and optional 'images_dir')")
    parser.add_argument("--force", action="store_true", help="ignore caches and redo the work")


def add_user_images(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images-dir", type=Path, help="use your own images for the address instead of Street View")
    parser.add_argument("--camera-height", type=float, help="camera height above the road in metres, for all your images")
    parser.add_argument("--source-type", choices=["phone", "google_screenshot"], help="what the images are (sets default height/FOV)")
    parser.add_argument("--crop-ui", action="store_true", help="crop app/browser UI bars and configured margins from screenshots")


def add_detect(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detect", default=os.getenv("DETECT", "all"),
        help="what to detect: any of ramps, driveways, signs, poles, curbs (comma separated), or all",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocationLabel: address -> coordinates -> images -> object labels -> measurements.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    stages = parser.add_subparsers(dest="stage", required=True)

    geocode = stages.add_parser("geocode", help="Stage 1: geocode addresses to lat/lng")
    add_source(geocode)

    streetview = stages.add_parser("streetview", help="Stage 2: fetch Street View images, or import your own with --images-dir")
    add_source(streetview)
    add_user_images(streetview)
    streetview.add_argument("--dry-run", action="store_true", help="resolve panos and views but download nothing")
    streetview.add_argument("--refine-with-google", action="store_true", help="use a Google ROOFTOP geocode as the target")

    detect = stages.add_parser("detect", help="Stage 3: label objects in fetched images (no Google calls)")
    add_source(detect)
    detect.add_argument("--model", help="override the segmentation model id")
    add_detect(detect)

    measure = stages.add_parser("measure", help="Stage 4: estimate sign height, curb height and ramp slopes (no Google calls)")
    add_source(measure)
    measure.add_argument("--export-pointcloud", action="store_true", help="save a .ply point cloud per image")

    evaluate = stages.add_parser("evaluate", help="compare measurements with a ground_truth.csv")
    evaluate.add_argument("ground_truth", type=Path, help="CSV: address, stop_sign_height_m, curb_height_m, ramp_running_slope_pct, ramp_cross_slope_pct")

    groundtruth = stages.add_parser("groundtruth", help="build and evaluate against public curb-ramp/sign/sidewalk inventories")
    gt_commands = groundtruth.add_subparsers(dest="gt_command", required=True)

    gt_inspect = gt_commands.add_parser("inspect", help="check each source is reachable and review its field mapping")
    gt_inspect.add_argument("--source", action="append", dest="only", help="limit to this source name (repeatable)")

    gt_download = gt_commands.add_parser("download", help="fetch and normalize every confirmed source")
    gt_download.add_argument("--force", action="store_true", help="ignore the raw-page cache and re-download")

    gt_commands.add_parser("build", help="sample objects and write groundTruth/data/ground_truth.csv")

    gt_run = gt_commands.add_parser("run-groundtruth", help="run Stages 2-4 on every ground_truth.csv row")
    gt_run.add_argument("--csv", type=Path, help="defaults to groundTruth/data/ground_truth.csv")
    gt_run.add_argument("--split", choices=["train", "test"], help="only this split")
    gt_run.add_argument("--limit", type=int, help="only the first N rows")
    gt_run.add_argument("--force", action="store_true", help="ignore already-measured rows and the Stage 2-4 caches")
    gt_run.add_argument("--dry-run", action="store_true", help="resolve panos/views and count image requests, download nothing")

    gt_bench = gt_commands.add_parser("rampbench", help="score the ramp detector on RampNet's labeled panoramas (no Google key needed)")
    gt_bench.add_argument("--panos", type=int, default=60, help="panoramas to render views from")
    gt_bench.add_argument("--seg-sizes", type=int, nargs="+", default=[640, 1024], help="segmentation input sizes to compare")

    gt_probe = gt_commands.add_parser("rampnetprobe", help="test RampNet's two models on the rampbench perspective views")
    gt_probe.add_argument("--rescore", action="store_true", help="re-run the models instead of reusing cached peaks")

    gt_evaluate = gt_commands.add_parser("evaluate", help="score detection and measurement against ground_truth.csv")
    gt_evaluate.add_argument("--csv", type=Path, help="defaults to groundTruth/data/ground_truth.csv")

    everything = stages.add_parser("all", help="run geocode, images (Street View or yours), detect and measure in order")
    add_source(everything)
    add_user_images(everything)
    everything.add_argument("--refine-with-google", action="store_true")
    everything.add_argument("--model")
    add_detect(everything)
    everything.add_argument("--export-pointcloud", action="store_true")
    everything.set_defaults(dry_run=False)

    args = parser.parse_args()
    if getattr(args, "detect", None) is not None:
        from label_objs.config import parse_targets

        try:
            args.targets = parse_targets(args.detect)
        except ValueError as exc:
            parser.error(str(exc))
    if getattr(args, "images_dir", None) and args.csv:
        parser.error("--images-dir is for a single address; use an 'images_dir' column in the CSV for batches")
    return args


def load_jobs(args: argparse.Namespace) -> list[Job]:
    if args.csv:
        return read_jobs(args.csv)
    return [Job(args.address, getattr(args, "images_dir", None))]


def run_images(jobs: list[Job], args: argparse.Namespace) -> Counter[str]:
    """Stage 2: import user images where given; fetch Street View only for the remaining addresses."""
    statuses: Counter[str] = Counter()
    own = [(job.address, job.images_dir) for job in jobs if job.images_dir]
    fetch = [job.address for job in jobs if not job.images_dir]
    if own:
        from user_images.run import run as run_import

        statuses += run_import(own, args.camera_height, args.source_type, args.crop_ui)
    if fetch:
        from cood_to_sview.run import run as run_streetview

        statuses += run_streetview(fetch, args.dry_run, args.force, args.refine_with_google)
    return statuses


def run_stages(args: argparse.Namespace, jobs: list[Job]) -> dict[str, tuple[Counter[str], float]]:
    """Run the requested stages in order; returns each stage's status counts and duration in seconds."""
    addresses = [job.address for job in jobs]
    summary_name = args.csv.stem if args.csv else None
    results: dict[str, tuple[Counter[str], float]] = {}

    def timed(name: str, run) -> None:
        started = time.perf_counter()
        results[name] = (run(), time.perf_counter() - started)

    # Stages are imported on demand so geocode/streetview work without the ML packages.
    if args.stage in ("geocode", "all"):
        from addr_To_coord.run import run as run_geocode

        to_geocode = addresses if args.stage == "geocode" else [job.address for job in jobs if not job.images_dir]
        if to_geocode:
            timed("geocode", lambda: run_geocode(to_geocode, args.force, summary_name))
    if args.stage in ("streetview", "all"):
        timed("images", lambda: run_images(jobs, args))
    if args.stage in ("detect", "all"):
        from label_objs.run import run as run_detect

        timed("detect", lambda: run_detect(addresses, args.force, args.model, args.targets))
    if args.stage in ("measure", "all"):
        from measure.run import run as run_measure

        timed("measure", lambda: run_measure(addresses, args.force, args.export_pointcloud, summary_name))
    return results


def run_groundtruth(args: argparse.Namespace) -> None:
    # The groundTruth package lives next to this folder, so its parent has to be importable.
    parent = str(Path(__file__).resolve().parent.parent)
    if parent not in sys.path:
        sys.path.append(parent)

    if args.gt_command == "rampbench":
        from groundTruth.rampbench import run_rampbench

        logger.info("groundtruth rampbench: wrote %s", run_rampbench(args.panos, tuple(args.seg_sizes)))
        return

    if args.gt_command == "rampnetprobe":
        from groundTruth.rampnet_probe import run_probe

        logger.info("groundtruth rampnetprobe: wrote %s", run_probe(args.rescore))
        return

    from groundTruth.config import GROUND_TRUTH_CSV
    from groundTruth import run as gt

    if args.gt_command == "inspect":
        gt.inspect(args.only)
    elif args.gt_command == "download":
        counts = gt.download(args.force)
        logger.info("groundtruth download: %s", counts)
    elif args.gt_command == "build":
        path = gt.build()
        logger.info("groundtruth build: wrote %s", path)
    elif args.gt_command == "run-groundtruth":
        statuses = gt.run_groundtruth(args.csv or GROUND_TRUTH_CSV, args.split, args.limit, args.force, args.dry_run)
        logger.info("groundtruth run-groundtruth: %s", dict(statuses))
    elif args.gt_command == "evaluate":
        path = gt.evaluate(args.csv or GROUND_TRUTH_CSV)
        logger.info("groundtruth evaluate: wrote %s", path)


def log_summary(results: dict[str, tuple[Counter[str], float]], n_addresses: int) -> None:
    total = sum(seconds for _, seconds in results.values())
    logger.info("Finished %d address%s in %s", n_addresses, "" if n_addresses == 1 else "es", format_duration(total))
    for stage, (statuses, seconds) in results.items():
        counts = "  ".join(f"{status} {count}" for status, count in sorted(statuses.items()))
        logger.info("  %-8s %-8s %s", stage, format_duration(seconds), counts)
    logger.info("  output   %s", output_dir())


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m {secs:02d}s" if minutes else f"{seconds:.1f}s"


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    # Third-party chatter (HTTP requests to the model hub, weight-loading reports) only at DEBUG.
    quiet = logging.DEBUG if args.log_level == "DEBUG" else logging.WARNING
    for name in ("httpx", "huggingface_hub", "urllib3", "transformers"):
        logging.getLogger(name).setLevel(quiet)
    if args.log_level != "DEBUG":
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    if args.stage == "evaluate":
        from measure.evaluate import evaluate

        evaluate(args.ground_truth, output_dir())
        return 0
    if args.stage == "groundtruth":
        with logging_redirect_tqdm():
            run_groundtruth(args)
        return 0

    jobs = load_jobs(args)
    with logging_redirect_tqdm():
        results = run_stages(args, jobs)
    log_summary(results, len(jobs))
    failed = any("error" in statuses for statuses, _ in results.values())
    return 1 if failed and args.address else 0


if __name__ == "__main__":
    sys.exit(main())
