"""Public CHB-MIT metadata inventory. This module never downloads EDFs/sidecars."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from .splitting import digest

BASE = "https://physionet.org/files/chbmit/1.0.0/"
PAGE = "https://physionet.org/content/chbmit/1.0.0/"
LICENSE = "https://physionet.org/content/chbmit/view-license/1.0.0/"
METADATA_LIMIT = 2 * 1024**2
CASE = re.compile(r"chb[0-9]{2}")
EDF = re.compile(r"(chb[0-9]{2})[a-z]?_[0-9]+\+?\.edf")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_patient(case: str) -> str:
    if not CASE.fullmatch(case):
        raise ValueError("invalid_case_directory")
    return "chb01" if case == "chb21" else case


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or value != path.as_posix() or any(ord(c) < 32 for c in value)):
        raise ValueError("unsafe_inventory_path")
    return value


def parse_listing(text: str) -> dict[str, int | None]:
    """Parse exact byte sizes from the official nginx listing, not rounded UI sizes."""
    result = {}
    for match in re.finditer(r'<a href="([^"]+)">.*?</a>([^\r\n]*)', text):
        name, rest = html.unescape(match[1]), match[2].strip()
        if name == "../":
            continue
        name = urllib.parse.unquote(name)
        safe_relative(name.rstrip("/"))
        if "/" in name.rstrip("/") or name in result:
            raise ValueError("directory_listing_duplicate_or_nested_entry")
        size = rest.split()[-1] if rest else ""
        if name.endswith("/"):
            if not CASE.fullmatch(name[:-1]) or size != "-":
                raise ValueError("invalid_case_directory")
            result[name] = None
        elif size.isdigit():
            result[name] = int(size)
        else:
            raise ValueError("directory_listing_size_missing")
    if not result:
        raise ValueError("empty_official_directory_listing")
    return result


def parse_checksums(data: bytes) -> dict[str, str]:
    result = {}
    for line in data.decode().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+[* ]?(.+)", line)
        if not match:
            raise ValueError("invalid_official_checksum_line")
        checksum, path = match.groups()
        path = safe_relative(path.removeprefix("./"))
        if path in result:
            raise ValueError("duplicate_official_checksum_path")
        result[path] = checksum
    return result


def parse_records(data: bytes, cases: set[str]) -> set[str]:
    result = set()
    for line in data.decode().splitlines():
        if not line.strip():
            continue
        name = safe_relative(line.strip())
        parts = PurePosixPath(name).parts
        match = EDF.fullmatch(parts[-1])
        if len(parts) != 2 or parts[0] not in cases or not match or match[1] != parts[0]:
            raise ValueError("record_case_or_filename_invalid")
        if name in result:
            raise ValueError("duplicate_official_record")
        result.add(name)
    return result


def parse_ages(data: bytes, cases: set[str]) -> dict:
    result = {}
    for line in data.decode().splitlines():
        if not line.strip() or line.startswith("Case"):
            continue
        match = re.fullmatch(r"(chb[0-9]{2})\s+([MF])\s+([0-9]+(?:\.[0-9]+)?)\s*", line)
        if not match or match[1] not in cases or match[1] in result:
            raise ValueError("subject_info_invalid")
        years = float(match[3])
        if not 0 < years < 120:
            raise ValueError("subject_age_invalid")
        # Preserve published decimal years. Never invent completed days or DOB.
        group = "unknown" if years < 1 else "child" if years < 13 else "adolescent" if years < 18 else "adult" if years < 65 else "older_adult"
        result[match[1]] = {"age_years": years, "sex": match[2], "age_group": group,
                           "source": BASE + "SUBJECT-INFO", "verification": "official_published_metadata"}
    return result


def parse_summary_inventory(data: bytes, case: str) -> dict:
    """Inventory declared events/montages only; not reconciled training labels."""
    text = data.decode("utf-8-sig")
    blocks = re.split(r"(?m)^File Name:\s*", text)
    prefix = blocks[0]
    rate_match = re.search(r"Data Sampling Rate:\s*([0-9.]+)\s*Hz", prefix)
    rate = float(rate_match[1]) if rate_match else None
    channels = re.findall(r"(?m)^Channel\s+[0-9]+:\s*(.*?)\s*$", prefix)
    result = {}
    for block in blocks[1:]:
        lines = block.splitlines()
        name = lines[0].strip()
        match = EDF.fullmatch(name)
        if not match or match[1] != case or name in result:
            raise ValueError("summary_record_identifier_invalid")
        count = re.search(r"Number of Seizures in File:\s*([0-9]+)", block)
        starts = re.findall(r"Seizure(?:\s+[0-9]+)? Start Time:\s*([0-9.]+) seconds", block)
        ends = re.findall(r"Seizure(?:\s+[0-9]+)? End Time:\s*([0-9.]+) seconds", block)
        events = [{"start_seconds": float(a), "end_seconds": float(b)} for a, b in zip(starts, ends)]
        declared = int(count[1]) if count else None
        valid = (declared is not None and declared == len(starts) == len(ends)
                 and all(0 <= e["start_seconds"] < e["end_seconds"] for e in events)
                 and all(a["end_seconds"] <= b["start_seconds"] for a, b in zip(events, events[1:])))
        result[name] = {"declared_seizures": declared, "summary_intervals": events,
                        "summary_event_structure_valid": valid, "sampling_rate_hz": rate,
                        "summary_channels": list(channels)}
        # Channel-change sections after this recording apply to the next one.
        changed = re.findall(r"(?m)^Channel\s+[0-9]+:\s*(.*?)\s*$", block)
        if changed:
            channels = changed
    return result


def build_inventory(metadata: dict[str, bytes]) -> dict:
    root = parse_listing(metadata["root.html"].decode())
    cases = {name[:-1] for name, size in root.items() if size is None}
    checksums = parse_checksums(metadata["SHA256SUMS.txt"])
    official_records = parse_records(metadata["RECORDS"], cases)
    official_positive = parse_records(metadata["RECORDS-WITH-SEIZURES"], cases)
    ages = parse_ages(metadata["SUBJECT-INFO"], cases)
    files = {}
    for name, size in root.items():
        if size is not None:
            files[name] = size
    for case in sorted(cases):
        for name, size in parse_listing(metadata[case + "/listing.html"].decode()).items():
            if size is None:
                raise ValueError("nested_case_directory_not_supported")
            files[case + "/" + name] = size
    missing_checksums = set(files) - set(checksums) - {"SHA256SUMS.txt"}
    if missing_checksums or set(checksums) - set(files):
        raise ValueError("official_listing_checksum_inventory_mismatch")
    for name, data in metadata.items():
        if name in files:
            if len(data) != files[name] or (name in checksums and sha256(data) != checksums[name]):
                raise ValueError("official_metadata_checksum_or_size_mismatch:" + name)
    recordings = []
    issues = []
    for case in sorted(cases):
        summary_path = f"{case}/{case}-summary.txt"
        summaries = parse_summary_inventory(metadata[summary_path], case) if summary_path in metadata else {}
        for path in sorted(name for name in files if name.startswith(case + "/") and name.endswith(".edf")):
            parse_records((path + "\n").encode(), cases)
            summary = summaries.get(PurePosixPath(path).name)
            sidecar = path + ".seizures" if path + ".seizures" in files else None
            gaps = []
            if path not in official_records:
                gaps.append("absent_from_RECORDS")
            if summary is None:
                gaps.append("summary_record_missing")
            elif not summary["summary_event_structure_valid"]:
                gaps.append("summary_events_invalid")
            positive = bool(summary and summary["summary_event_structure_valid"] and summary["declared_seizures"] > 0)
            if positive != (path in official_positive) or positive != bool(sidecar):
                gaps.append("summary_sidecar_seizure_list_conflict")
            if gaps:
                issues.append({"path": path, "gaps": gaps})
            recordings.append({"path": path, "case_id": case, "canonical_patient_id": canonical_patient(case),
                "bytes": files[path], "sha256": checksums[path], "url": BASE + path,
                "in_RECORDS": path in official_records, "in_RECORDS_WITH_SEIZURES": path in official_positive,
                "summary_path": summary_path if summary_path in files else None, "sidecar_path": sidecar,
                "age": ages.get(case), "summary": summary,
                "official_seizure_positive": positive and not gaps,
                "background_candidate": bool(summary and summary["declared_seizures"] == 0 and not gaps),
                "verified_background": False, "annotation_review": "unreviewed",
                "annotation_gaps": gaps, "archive_eligible": not gaps,
                "age_specific_training_eligible": False, "channel_review": "summary_only_unverified_against_EDF"})
    listed_edfs = {item["path"] for item in recordings}
    if official_records - listed_edfs or official_positive - listed_edfs:
        raise ValueError("RECORDS_references_missing_EDF")
    payload = {"schema_version": "chbmit-complete-inventory-v1", "dataset_version": "1.0.0",
        "official_source": PAGE, "license": "Open Data Commons Attribution License v1.0",
        "license_url": LICENSE, "citation": "Guttag, J. (2010). CHB-MIT Scalp EEG Database (version 1.0.0). PhysioNet. doi:10.13026/C2K01R",
        "canonical_identity_policy": {"chb21": "chb01", "source": PAGE,
                                      "other_cases": "distinct case subjects per official description; chb24 added later"},
        "cases": sorted(cases), "case_count": len(cases),
        "canonical_patient_count": len({canonical_patient(case) for case in cases}),
        "recording_count": len(recordings), "RECORDS_count": len(official_records),
        "RECORDS_WITH_SEIZURES_count": len(official_positive),
        "edf_bytes": sum(item["bytes"] for item in recordings), "all_official_file_bytes": sum(files.values()),
        "declared_seizures": sum((item["summary"] or {}).get("declared_seizures") or 0 for item in recordings),
        "verified_background_count": 0, "annotation_issues": issues,
        "files": [{"path": name, "bytes": size, "sha256": checksums.get(name, sha256(metadata["SHA256SUMS.txt"])),
                   "url": BASE + name, "hash_source": "official_SHA256SUMS" if name in checksums else "retrieved_metadata_body"}
                  for name, size in sorted(files.items())],
        "source_metadata_sha256": {name: sha256(data) for name, data in sorted(metadata.items())},
        "recordings": recordings, "training_labels_created": False}
    payload["inventory_sha256"] = digest(payload)
    return payload


def fetch_metadata(cache: Path) -> dict[str, bytes]:
    """Bounded public text/listing fetch; EDF, sidecar and archive URLs forbidden."""
    cache.mkdir(parents=True, exist_ok=True)
    def fetch(pair):
        name, url = pair
        if not (url in (PAGE, LICENSE, BASE) or re.fullmatch(re.escape(BASE) +
                r"(?:chb[0-9]{2}/(?:chb[0-9]{2}-summary\.txt)?|RECORDS|RECORDS-WITH-SEIZURES|SUBJECT-INFO|ANNOTATORS|SHA256SUMS\.txt)", url)):
            raise ValueError("metadata_fetch_url_not_allowlisted")
        target = cache / name
        if target.is_file():
            return name, target.read_bytes()
        with urllib.request.urlopen(url, timeout=60) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("metadata_http_status_or_redirect_invalid")
            data = response.read(METADATA_LIMIT + 1)
        if len(data) > METADATA_LIMIT:
            raise ValueError("metadata_size_limit")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return name, data
    root_name, root = fetch(("root.html", BASE))
    listing = parse_listing(root.decode())
    cases = sorted(name[:-1] for name, size in listing.items() if size is None)
    requests = [(name, BASE + name) for name in ("RECORDS", "RECORDS-WITH-SEIZURES", "SUBJECT-INFO", "ANNOTATORS", "SHA256SUMS.txt")]
    requests += [("dataset-page.html", PAGE), ("license.html", LICENSE)]
    requests += [(case + "/listing.html", BASE + case + "/") for case in cases]
    with ThreadPoolExecutor(max_workers=4) as pool:
        metadata = {root_name: root, **dict(pool.map(fetch, requests))}
        summary_requests = [(f"{case}/{case}-summary.txt", f"{BASE}{case}/{case}-summary.txt")
                            for case in cases if f"{case}-summary.txt" in parse_listing(metadata[case + "/listing.html"].decode())]
        metadata.update(pool.map(fetch, summary_requests))
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fetch-public-metadata", action="store_true")
    args = parser.parse_args(argv)
    metadata = (fetch_metadata(args.cache) if args.fetch_public_metadata else
                {p.relative_to(args.cache).as_posix(): p.read_bytes() for p in args.cache.rglob("*") if p.is_file()})
    result = build_inventory(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("case_count", "canonical_patient_count", "recording_count", "edf_bytes", "declared_seizures", "inventory_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
