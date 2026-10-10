"""Offline inspection and reviewed ingestion CLI. No downloads or inference."""

import argparse
import json
import sys

from pydantic import ValidationError

from .adapters.base import IngestionError
from .adapters.chb_mit import inspect_summary
from .edf import inspect_edf
from .ingestion import ingest_cohort, load_plan, write_artifacts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    edf = commands.add_parser("inspect-edf", help="Inspect a local legacy EDF; no approval is implied")
    edf.add_argument("file")
    summary = commands.add_parser("inspect-summary", help="Inspect a local CHB-MIT summary and its hash")
    summary.add_argument("file")
    summary.add_argument("--subject", required=True)
    ingest = commands.add_parser("ingest", help="Validate reviewed local sources and write JSON artifacts")
    ingest.add_argument("--root", required=True)
    ingest.add_argument("--plan", required=True)
    ingest.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-edf":
            output = inspect_edf(args.file).model_dump(mode="json")
        elif args.command == "inspect-summary":
            records, sha256, size = inspect_summary(args.file, args.subject)
            output = {"sha256": sha256, "size_bytes": size,
                      "records": [item.model_dump(mode="json") for item in records]}
        else:
            result = ingest_cohort(args.root, load_plan(args.plan))
            write_artifacts(result, args.output_dir)
            output = {"cohort_approved": result.audit.cohort_approved,
                      "validated_recordings": result.audit.validated_recordings,
                      "rejected_recordings": result.audit.rejected_recordings,
                      "manifest_sha256": result.audit.manifest_sha256}
            print(json.dumps(output, sort_keys=True))
            return 0 if result.audit.cohort_approved else 2
        print(json.dumps(output, sort_keys=True, indent=2))
        return 0
    except IngestionError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
    except (ValidationError, ValueError, UnicodeError):
        print(json.dumps({"error": "invalid_input_or_review_plan"}), file=sys.stderr)
    except OSError:
        print(json.dumps({"error": "local_file_or_artifact_io_failed"}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
