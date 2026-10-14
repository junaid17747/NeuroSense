"""Synthetic-only D7 inventory and lossless archive checks."""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from seizure_research.archive import (ArchiveBudget, MIN_SYSTEM_FREE, archive_plan,
                                      compress_recording, verify_zip, verify_original,
                                      download_original, recover_temporary_archive)
from seizure_research.inventory import (build_inventory, canonical_patient, parse_listing,
                                        parse_records, parse_summary_inventory, safe_relative)
from seizure_research.archive_selection import select_cohort
from seizure_research.archive import prepare_local_storage
from seizure_research.annotation_review import (audit_review_packet, build_review_packet, main as annotation_review_main,
                                                 seal_review_packet)


class InventoryD7Tests(unittest.TestCase):
    @staticmethod
    def mini_inventory_source():
        raw = {
            "ANNOTATORS": b"synthetic metadata\n",
            "RECORDS": b"chb01/chb01_01.edf\nchb21/chb21_01.edf\nchb24/chb24_01.edf\n",
            "RECORDS-WITH-SEIZURES": b"chb01/chb01_01.edf\n",
            "SUBJECT-INFO": b"Case Gender Age\nchb01 F 11\nchb21 F 13\n",
            "chb01/chb01-summary.txt": b"Data Sampling Rate: 256 Hz\nFile Name: chb01_01.edf\nNumber of Seizures in File: 1\nSeizure Start Time: 1 seconds\nSeizure End Time: 2 seconds\n",
            "chb21/chb21-summary.txt": b"Data Sampling Rate: 256 Hz\nFile Name: chb21_01.edf\nNumber of Seizures in File: 0\n",
        }
        files = {"chb01/chb01_01.edf": 100, "chb01/chb01_01.edf.seizures": 16,
                 "chb21/chb21_01.edf": 100, "chb24/chb24_01.edf": 100}
        checksums = {path: hashlib.sha256(data).hexdigest() for path, data in raw.items()}
        checksums.update({path: hashlib.sha256(path.encode()).hexdigest() for path in files})
        raw["SHA256SUMS.txt"] = ("".join(f"{value}  {path}\n" for path, value in sorted(checksums.items()))).encode()
        case_entries = {
            "chb01": {"chb01-summary.txt": raw["chb01/chb01-summary.txt"], "chb01_01.edf": b"" , "chb01_01.edf.seizures": b""},
            "chb21": {"chb21-summary.txt": raw["chb21/chb21-summary.txt"], "chb21_01.edf": b""},
            "chb24": {"chb24_01.edf": b""},
        }
        for case, entries in case_entries.items():
            listing = "".join(f'<a href="{name}">{name}</a> {len(raw.get(case + "/" + name, b"")) if name.endswith("summary.txt") else files[case + "/" + name]}\n' for name in entries)
            raw[case + "/listing.html"] = listing.encode()
        root_entries = {"chb01/": None, "chb21/": None, "chb24/": None}
        root_entries.update({name: len(data) for name, data in raw.items() if "/" not in name})
        raw["root.html"] = "".join(f'<a href="{name}">{name}</a> {size if size is not None else "-"}\n' for name, size in root_entries.items()).encode()
        return raw

    def test_complete_inventory_hash_alias_annotation_gaps_and_selection(self):
        inventory = build_inventory(self.mini_inventory_source())
        self.assertEqual(inventory["case_count"], 3)
        self.assertEqual(inventory["canonical_patient_count"], 2)
        self.assertEqual(inventory["recording_count"], 3)
        rows = {row["path"]: row for row in inventory["recordings"]}
        self.assertEqual(rows["chb01/chb01_01.edf"]["canonical_patient_id"], "chb01")
        self.assertEqual(rows["chb21/chb21_01.edf"]["canonical_patient_id"], "chb01")
        self.assertTrue(rows["chb01/chb01_01.edf"]["official_seizure_positive"])
        self.assertTrue(rows["chb21/chb21_01.edf"]["background_candidate"])
        self.assertEqual(rows["chb24/chb24_01.edf"]["annotation_gaps"], ["summary_record_missing"])
        proposal = select_cohort(inventory)
        self.assertEqual(proposal["selected_patient_count"], 1)
        self.assertEqual(next(p for p in proposal["patients"] if p["canonical_patient_id"] == "chb24")["missing_patient_reason"],
                         "no_annotation_consistent_candidate")
        self.assertGreaterEqual(proposal["selected_official_seizure_positive_recordings"], 1)
        self.assertEqual(proposal["selected_verified_background_recordings"], 0)
        self.assertEqual(proposal, select_cohort(inventory))

    def test_annotation_review_packet_is_actionable_but_creates_no_labels(self):
        inventory = build_inventory(self.mini_inventory_source())
        selection = select_cohort(inventory)
        packet = build_review_packet(inventory, selection)
        self.assertEqual(packet["selected_recording_count"], len(packet["recordings"]))
        self.assertEqual(packet["verified_background_recordings"], 0)
        self.assertFalse(packet["training_labels_created"])
        self.assertTrue(all(row["review_status"] == "unreviewed" for row in packet["recordings"]))
        self.assertTrue(all(row["verified_seizure_free"] is False for row in packet["recordings"]))
        audit = audit_review_packet(inventory, selection, packet)
        self.assertFalse(audit["annotation_evidence_complete"])
        self.assertEqual(audit["verified_background_recording_count"], 0)
        self.assertTrue(audit["blockers"])

    def test_completed_synthetic_review_requires_full_coverage_and_bound_sources(self):
        inventory = build_inventory(self.mini_inventory_source())
        selection = select_cohort(inventory)
        packet = build_review_packet(inventory, selection)
        for row, selected in zip(packet["recordings"], selection["recordings"]):
            summary = selected.get("summary") or {}
            intervals = summary.get("summary_intervals", [])
            last_end = max((event["end_seconds"] for event in intervals), default=0)
            row.update({
                "recording_duration_seconds": max(30, last_end + 1),
                "review_status": "reviewed",
                "reviewer": "synthetic reviewer fixture",
                "reviewed_on": "2026-01-01",
                "evidence_ref": "synthetic review fixture",
                "sidecar_parsed_and_reconciled": bool(intervals),
                "reconciliation_notes": "synthetic fixture reconciled to test source metadata",
                "reviewed_event_intervals_seconds": intervals,
                "complete_record_coverage_attested": True,
                "verified_seizure_free": not bool(intervals),
                "quality_and_excluded_intervals_reviewed": True,
            })
        packet = seal_review_packet(packet)
        audit = audit_review_packet(inventory, selection, packet)
        self.assertTrue(audit["annotation_evidence_complete"], audit["blockers"])
        self.assertEqual(audit["reviewed_recording_count"], len(selection["recordings"]))
        self.assertEqual(audit["verified_background_recording_count"], 1)
        self.assertFalse(audit["training_labels_created"])

    def test_review_packet_audit_rejects_tampering_and_incomplete_negative_coverage(self):
        inventory = build_inventory(self.mini_inventory_source())
        selection = select_cohort(inventory)
        packet = build_review_packet(inventory, selection)
        row = packet["recordings"][0]
        row["original_edf_sha256"] = "0" * 64
        audit = audit_review_packet(inventory, selection, packet)
        reasons = {item["reason"] for item in audit["blockers"]}
        self.assertIn("review_source_binding_mismatch:original_edf_sha256", reasons)
        self.assertIn("review_packet_checksum_mismatch", reasons)

    def test_annotation_review_cli_reports_incomplete_without_labels(self):
        inventory = build_inventory(self.mini_inventory_source())
        selection = select_cohort(inventory)
        packet = build_review_packet(inventory, selection)
        with tempfile.TemporaryDirectory() as directory:
            inventory_path = Path(directory) / "inventory.json"
            selection_path = Path(directory) / "selection.json"
            packet_path = Path(directory) / "review.json"
            inventory_path.write_text(json.dumps(inventory))
            selection_path.write_text(json.dumps(selection))
            packet_path.write_text(json.dumps(packet))
            output = StringIO()
            with redirect_stdout(output):
                result = annotation_review_main([
                    "--inventory", str(inventory_path), "--selection", str(selection_path),
                    "--audit-packet", str(packet_path),
                ])
        report = json.loads(output.getvalue())
        self.assertEqual(result, 2)
        self.assertEqual(report["status"], "annotation_review_incomplete")
        self.assertFalse(report["training_labels_created"])

    def test_official_directory_listing_sizes_and_directories_are_exact(self):
        listing = '<a href="../">Parent Directory</a> -\n<a href="chb01/">chb01/</a> -\n<a href="RECORDS">RECORDS</a> 42\n'
        self.assertEqual(parse_listing(listing), {"chb01/": None, "RECORDS": 42})
        with self.assertRaisesRegex(ValueError, "size_missing"):
            parse_listing('<a href="RECORDS">RECORDS</a> -\n')

    def test_case_file_validation_and_alias(self):
        self.assertEqual(canonical_patient("chb21"), "chb01")
        self.assertEqual(canonical_patient("chb24"), "chb24")
        with self.assertRaisesRegex(ValueError, "invalid_case"):
            canonical_patient("chb1")
        self.assertEqual(parse_records(b"chb21/chb21_01.edf\n", {"chb21"}),
                         {"chb21/chb21_01.edf"})
        with self.assertRaisesRegex(ValueError, "unsafe"):
            safe_relative("../chb01.edf")

    def test_summary_intervals_are_declared_not_verified_labels(self):
        summary = ("Data Sampling Rate: 256 Hz\nChannel 1: Fp1-F7\n"
                   "File Name: chb02_01.edf\nNumber of Seizures in File: 1\n"
                   "Seizure Start Time: 10 seconds\nSeizure End Time: 20 seconds\n")
        parsed = parse_summary_inventory(summary.encode(), "chb02")
        row = parsed["chb02_01.edf"]
        self.assertTrue(row["summary_event_structure_valid"])
        self.assertEqual(row["summary_intervals"], [{"start_seconds": 10.0, "end_seconds": 20.0}])


class ArchiveD7Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root, self.temp = base / "archive", base / "scratch"
        self.root.mkdir()
        self.temp.mkdir()
        self.payload = (b"synthetic-edf-header" + bytes(range(256))) * 512
        self.item = {"path": "chb02/chb02_01.edf", "bytes": len(self.payload),
                     "sha256": hashlib.sha256(self.payload).hexdigest(),
                     "url": "https://physionet.org/files/chbmit/1.0.0/chb02/chb02_01.edf"}
        self.budget = ArchiveBudget(self.root, self.temp, limit=2_147_483_648,
                                    temporary_limit=8 * 1024**2)

    def test_zip_roundtrip_preserves_all_bytes_and_checksum(self):
        source, packed = self.temp / "source.part", self.temp / "recording.zip.part"
        source.write_bytes(self.payload)
        verify_original(source, self.item)
        result = compress_recording(source, packed, self.item, self.budget)
        self.assertTrue(result["exact_decompression_verified"])
        self.assertEqual(result["original_sha256"], self.item["sha256"])
        self.assertEqual(verify_zip(packed, self.item), result)

    def test_zip_corruption_and_mismatched_original_are_rejected(self):
        source, packed = self.temp / "source.part", self.temp / "recording.zip.part"
        source.write_bytes(self.payload)
        compress_recording(source, packed, self.item, self.budget)
        with packed.open("r+b") as stream:
            stream.seek(-8, io.SEEK_END)
            stream.write(b"BROKEN!!")
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            verify_zip(packed, self.item)
        source.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "checksum_or_size"):
            verify_original(source, self.item)

    def test_interrupted_zip_is_discarded_only_after_source_is_verified(self):
        source, packed = self.temp / "source.part", self.temp / "recording.zip.part"
        source.write_bytes(self.payload)
        packed.write_bytes(b"partial zip")
        self.assertFalse(recover_temporary_archive(packed, source, self.item))
        self.assertTrue(source.exists())
        self.assertFalse(packed.exists())
        packed.write_bytes(b"partial zip")
        with self.assertRaisesRegex(ValueError, "checksum_or_size"):
            recover_temporary_archive(packed, source, {**self.item, "sha256": "0" * 64})

    def test_partial_download_resumes_only_with_exact_range_and_checksum(self):
        half = len(self.payload) // 2
        partial = self.temp / (self.item["sha256"] + ".part")
        partial.write_bytes(self.payload[:half])

        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": f"bytes {half}-{len(self.payload)-1}/{len(self.payload)}",
                       "Content-Length": str(len(self.payload) - half)}
            def geturl(self):
                return self.item_url

        response = Response(self.payload[half:])
        response.item_url = self.item["url"]
        result = download_original(self.item, self.temp, self.budget,
                                   lambda path: self.assertEqual(path, self.item["path"]),
                                   opener=lambda *_args, **_kwargs: response)
        self.assertEqual(result.read_bytes(), self.payload)

    def test_interrupted_transfer_preserves_verified_prefix_for_resume(self):
        source_url = self.item["url"]
        class ShortResponse(io.BytesIO):
            status = 200
            headers = {"Content-Length": str(len(self.payload))}
            def geturl(self):
                return source_url

        with self.assertRaisesRegex(OSError, "interrupted_or_truncated"):
            download_original(self.item, self.temp, self.budget, lambda _path: None,
                              opener=lambda *_a, **_k: ShortResponse(self.payload[:100]))
        partial = self.temp / (self.item["sha256"] + ".part")
        self.assertEqual(partial.read_bytes(), self.payload[:100])
        with self.assertRaisesRegex(OSError, "interrupted_or_truncated"):
            download_original(self.item, self.temp, self.budget, lambda _path: None,
                              opener=lambda *_a, **_k: ShortResponse(self.payload[:100]))

    def test_retained_budget_is_hard_capped_at_two_gib(self):
        with self.assertRaisesRegex(OSError, "retained_archive_2gib"):
            self.budget.check(retained_add=2_147_483_649)

    def test_local_storage_tree_is_private_and_outside_repository(self):
        home = Path(self.temporary.name)
        with patch("seizure_research.archive.Path.home", return_value=home):
            paths = prepare_local_storage()
        expected = {name for name in ("archive", "temporary", "manifests", "reports", "runs")}
        self.assertEqual(set(paths), expected)
        for path in (home / "NeuroSense-EEG-Data", *paths.values()):
            self.assertTrue(path.is_dir())
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)

    def test_plan_reserves_archive_and_peak_temporary_bytes_together(self):
        inventory = build_inventory(InventoryD7Tests.mini_inventory_source())
        selection = select_cohort(inventory)
        retained = selection["total_retained_byte_bound"]
        largest = max(item["bytes"] for item in selection["recordings"])
        peak = 2 * largest + 65536
        available = MIN_SYSTEM_FREE + retained + peak - 1
        usage = __import__("shutil")._ntuple_diskusage(10**12, 0, available)
        with patch("seizure_research.archive.shutil.disk_usage", return_value=usage):
            with self.assertRaisesRegex(OSError, "reserve_or_volume_space_insufficient"):
                archive_plan(self.root, self.temp, inventory, selection)

    def test_unauthorized_callback_blocks_before_network(self):
        with self.assertRaisesRegex(PermissionError, "test"):
            download_original(self.item, self.temp, self.budget,
                              lambda _path: (_ for _ in ()).throw(PermissionError("test")),
                              opener=lambda *_a, **_k: self.fail("network must not run"))

    def test_temporary_quota_stops_before_write(self):
        self.budget.temporary_limit = 1
        with self.assertRaisesRegex(OSError, "temporary_storage_limit"):
            self.budget.check(temporary_add=2)


if __name__ == "__main__":
    unittest.main()
