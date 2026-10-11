"""Generated EDFs and fictitious annotations only; no clinical dataset is used.

All reviews, ages, identities, devices, and events are TEST_ONLY fabrications.
Known alias names appear only to exercise the published alias constraint.
"""

import contextlib
import copy
import hashlib
import io
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from seizure_research.__main__ import main
from seizure_research.adapters.base import HumanReview, IngestionError, IngestionPlan
from seizure_research.adapters.chb_mit import CHBMITAdapter, parse_summary
from seizure_research.edf import inspect_edf
from seizure_research.eligibility import REGISTERED_MODELS, assess_eligibility
from seizure_research.ingestion import (
    REPOSITORY_ROOT, CohortManifest, artifact_bytes, ingest_cohort, load_plan,
    validate_manifest, write_artifacts,
)
from seizure_research.registry import DATASET_REGISTRY
from seizure_research.schemas import AgeGroup
from seizure_research.splitting import build_split_manifest


def sha(data):
    return hashlib.sha256(data).hexdigest()


def field(value, length):
    encoded = str(value).encode("ascii")
    assert len(encoded) <= length
    return encoded.ljust(length, b" ")


def edf_bytes(token="TEST_ONLY", labels=("FP1-F7", "F7-T7"), samples=None, seconds=1, records=60, units=None):
    """Write the fixed-width ordinary EDF layout with small artificial signals."""
    count = len(labels)
    samples = samples or (256,) * count
    units = units or ("uV",) * count
    fixed = b"".join((field("0", 8), field("TEST_ONLY_NEVER_EXPORT_PATIENT", 80),
                      field(token, 80), field("01.01.00", 8), field("00.00.00", 8),
                      field(256 * (count + 1), 8), field("", 44), field(records, 8),
                      field(seconds, 8), field(count, 4)))
    signal_fields = ((16, labels), (80, ("",) * count), (8, units), (8, (-100,) * count),
                     (8, (100,) * count), (8, (-32768,) * count), (8, (32767,) * count),
                     (80, ("",) * count), (8, samples), (32, ("",) * count))
    variable = b"".join(field(value, width) for width, values in signal_fields for value in values)
    values = tuple((index % 101) - 50 for index in range(sum(samples)))
    data = struct.pack("<" + "h" * len(values), *values) * records
    return fixed + variable + data


def summary_text(filename="chb90_01.edf", labels=("FP1-F7", "F7-T7"), events=((10, 20),),
                 start="00:00:00", end="00:01:00", numbered=False):
    header = "Data Sampling Rate: 256 Hz\n***\nChannels in EDF Files:\n***\n"
    header += "\n".join(f"Channel {i}: {label}" for i, label in enumerate(labels, 1)) + "\n"
    block = (f"File Name: {filename}\nFile Start Time: {start}\nFile End Time: {end}\n"
             f"Number of Seizures in File: {len(events)}\n")
    for index, (onset, offset) in enumerate(events, 1):
        prefix = f"Seizure {index}" if numbered else "Seizure"
        block += f"{prefix} Start Time: {onset} seconds\n{prefix} End Time: {offset} seconds\n"
    return header + block


def verified():
    return {"verified": True, "reviewer": "TEST_ONLY", "reviewed_on": "2026-10-11",
            "evidence_ref": "TEST_ONLY_review"}


def plan_fixture(root, subjects=("chb90", "chb91", "chb92"), years=(10, 15, 20), events=((10, 20),)):
    records = []
    for subject, years_old in zip(subjects, years):
        directory = root / subject
        directory.mkdir(exist_ok=True)
        filename = f"{subject}_01.edf"
        signal = edf_bytes(f"TEST_ONLY_{subject}")
        summary = summary_text(filename, events=events).encode()
        (directory / filename).write_bytes(signal)
        (directory / f"{subject}-summary.txt").write_bytes(summary)
        records.append({
            "edf_path": f"{subject}/{filename}", "summary_path": f"{subject}/{subject}-summary.txt",
            "subject_id": subject, "age": {"completed_years": years_old, "source": "TEST_ONLY_age_review"},
            "device": {"manufacturer": "TEST_ONLY", "model": "TEST_ONLY",
                       "acquisition_software_version": "TEST_ONLY", "clinical_eeg": True},
            "montage": {"name": "TEST_ONLY", "kind": "bipolar", "channels": ["FP1-F7", "F7-T7"],
                        "reference": "pairwise", "units": "uV"},
            "channels": [{"source_index": i, "source_label": label, "action": "keep", "output_label": label}
                         for i, label in enumerate(("FP1-F7", "F7-T7"))],
            "expected_sampling_rate_hz": 256.0, "expected_signal_sha256": sha(signal),
            "expected_summary_sha256": sha(summary), "metadata_review": verified(),
            "annotation_review": verified(), "annotation_coverage": "complete",
            "annotation_provenance_ref": "TEST_ONLY_expert_annotation_provenance",
            "annotation_protocol_ref": "TEST_ONLY", "annotator_ref": "TEST_ONLY",
            "quality": {"status": "acceptable", "review_ref": "TEST_ONLY"},
        })
    return {"adapter_id": "chb_mit", "dataset_version": "1.0.0", "dataset_review": verified(),
            "permissions": {"status": "approved", "deidentified": True,
                            "terms_review_ref": "TEST_ONLY_terms", "ethics_review_ref": "TEST_ONLY_ethics",
                            "access_review_ref": "TEST_ONLY_access", "reviewed_by": "TEST_ONLY",
                            "reviewed_on": "2026-10-11"},
            "identities": {"review": verified(),
                           "entries": [{"subject_id": subject, "canonical_patient_id": f"TEST_ONLY_{subject}"}
                                       for subject in subjects]}, "recordings": records}


class EDFInspectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic.edf"

    def test_rates_duration_fingerprint_and_identity_redaction(self):
        data = edf_bytes(samples=(256, 128), seconds=2, records=3)
        self.path.write_bytes(data)
        metadata = inspect_edf(self.path)
        self.assertEqual(metadata.duration_seconds, 6)
        self.assertEqual([item.sampling_rate_hz for item in metadata.channels], [128, 64])
        self.assertEqual(metadata.sha256, sha(data))
        self.assertEqual(metadata.size_bytes, len(data))
        self.assertNotIn("TEST_ONLY_NEVER_EXPORT_PATIENT", metadata.model_dump_json())
        self.assertNotIn("01.01.00", metadata.model_dump_json())

    def test_truncated_extra_payload_and_wrong_header_length_rejected(self):
        good = edf_bytes()
        variants = (good[:100], good[:300], good[:-1], good + b"extra",
                    good[:184] + field(257, 8) + good[192:])
        for data in variants:
            self.path.write_bytes(data)
            with self.subTest(length=len(data)), self.assertRaises(IngestionError):
                inspect_edf(self.path)

    def test_unknown_length_nonpositive_or_nonfinite_duration_and_variants_rejected(self):
        for offset, width, value in ((236, 8, -1), (236, 8, 0), (244, 8, 0),
                                     (244, 8, "nan"), (244, 8, "inf"), (252, 4, 0),
                                     (192, 44, "EDF+D"), (192, 44, "EDF+C"), (0, 8, "BDF")):
            data = edf_bytes()
            self.path.write_bytes(data[:offset] + field(value, width) + data[offset + width:])
            with self.subTest(value=value), self.assertRaises(IngestionError):
                inspect_edf(self.path)

    def test_malformed_channel_sample_count_and_calibration_rejected(self):
        # The first sample-count field begins after 216 bytes per signal.
        for offset, width, value in ((256, 16, ""), (256 + 216 * 2, 8, 0),
                                     (256 + 104 * 2, 8, 100), (256 + 120 * 2, 8, -40000)):
            data = edf_bytes()
            self.path.write_bytes(data[:offset] + field(value, width) + data[offset + width:])
            with self.subTest(offset=offset), self.assertRaises(IngestionError):
                inspect_edf(self.path)

    def test_repeated_labels_are_preserved_for_explicit_index_review(self):
        self.path.write_bytes(edf_bytes(labels=("FP1-F7", "-", "FP1-F7")))
        self.assertEqual([item.label for item in inspect_edf(self.path).channels], ["FP1-F7", "-", "FP1-F7"])

    def test_inspection_cli_needs_no_approval_or_download(self):
        self.path.write_bytes(edf_bytes())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["inspect-edf", str(self.path)]), 0)
        self.assertEqual(json.loads(output.getvalue())["format"], "EDF")
        self.assertNotIn("TEST_ONLY_NEVER_EXPORT_PATIENT", output.getvalue())


class SummaryParsingTests(unittest.TestCase):
    def test_numbered_and_unnumbered_annotations(self):
        for numbered in (True, False):
            result = parse_summary(summary_text(numbered=numbered), "chb90")
            self.assertEqual(result[0].events[0].start_seconds, 10)
            self.assertEqual(result[0].events[0].end_seconds, 20)
            self.assertEqual(result[0].duration_seconds, 60)

    def test_multiple_events_and_explicit_zero(self):
        result = parse_summary(summary_text(events=((0, 10), (30, 60)), numbered=True), "chb90")
        self.assertEqual(len(result[0].events), 2)
        zero = parse_summary(summary_text(events=()), "chb90")[0]
        self.assertEqual(zero.declared_seizure_count, 0)
        self.assertEqual(zero.events, ())

    def test_missing_counts_are_not_zero_and_truncated_events_rejected(self):
        text = summary_text()
        for bad in (text.replace("Number of Seizures in File: 1\n", ""),
                    text.replace("Seizure End Time: 20 seconds\n", ""),
                    text.replace("Number of Seizures in File: 1", "Number of Seizures in File: 0")):
            with self.subTest(text=bad), self.assertRaises(IngestionError):
                parse_summary(bad, "chb90")

    def test_invalid_bounds_negative_nan_overlapping_or_unordered_events_rejected(self):
        for events in (((20, 10),), ((10, 61),), ((-1, 10),), (("nan", 20),),
                       ((10, 10),), ((10, 30), (20, 40)), ((30, 40), (10, 20))):
            with self.subTest(events=events), self.assertRaises(IngestionError):
                parse_summary(summary_text(events=events, numbered=True), "chb90")

    def test_huge_declared_event_count_is_rejected_without_allocating_it(self):
        text = summary_text().replace("Number of Seizures in File: 1", "Number of Seizures in File: 999999999999")
        with self.assertRaisesRegex(IngestionError, "event_count_mismatch"):
            parse_summary(text, "chb90")

    def test_duplicate_fields_blocks_and_ambiguous_numbering_rejected(self):
        original = summary_text()
        variants = (original + "Seizure Start Time: 10 seconds\n",
                    original + "Number of Seizures in File: 1\n",
                    original + original[original.index("File Name:"):],
                    original.replace("Seizure End", "Seizure 1 End"),
                    original.replace("Seizure Start", "Seizure 2 Start"),
                    original + "File Start Time: 00:00:00\n")
        for text in variants:
            with self.subTest(text=text), self.assertRaises(IngestionError):
                parse_summary(text, "chb90")

    def test_identifiers_path_traversal_and_unknown_lines_rejected(self):
        for filename in ("../chb90_01.edf", "chb91_01.edf", "arbitrary.edf", "chb90_01.edf.bak"):
            with self.subTest(filename=filename), self.assertRaises(IngestionError):
                parse_summary(summary_text(filename=filename), "chb90")
        with self.assertRaises(IngestionError):
            parse_summary(summary_text() + "Sezure Start: 35\n", "chb90")

    def test_filename_variants_channel_changes_and_rollover(self):
        first = summary_text(filename="chb90a_01+.edf", start="23:59:30", end="24:00:30")
        second = summary_text(filename="chb90b_02.edf", labels=("FZ-CZ",), events=())
        second = second[second.index("Channel 1:"):]
        parsed = parse_summary(first + "Channels changed:\n***\n" + second, "chb90")
        self.assertEqual(parsed[0].channels, ("FP1-F7", "F7-T7"))
        self.assertEqual(parsed[1].channels, ("FZ-CZ",))
        self.assertEqual(parsed[0].duration_seconds, 60)
        self.assertEqual(parse_summary(summary_text(start="23:59:30", end="00:00:30"), "chb90")[0].duration_seconds, 60)

    def test_invalid_clocks_or_missing_acquisition_metadata_rejected(self):
        for text in (summary_text(end="00:99:00"), summary_text(end="00:00:00"),
                     summary_text().replace("Data Sampling Rate: 256 Hz\n", ""),
                     summary_text().replace("Channel 2:", "Channel 3:")):
            with self.subTest(text=text), self.assertRaises(IngestionError):
                parse_summary(text, "chb90")


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = plan_fixture(self.root)

    def ingest(self, data=None):
        return ingest_cohort(self.root, IngestionPlan.model_validate(self.data if data is None else data))

    def assert_rejected(self, data, code):
        result = self.ingest(data)
        self.assertIsNone(result.manifest)
        self.assertFalse(result.audit.cohort_approved)
        self.assertTrue(any(code in entry.reasons for entry in result.audit.entries), result.audit)
        return result

    def test_approved_ingestion_preserves_three_age_groups_and_real_checksums(self):
        result = self.ingest()
        self.assertTrue(result.audit.cohort_approved)
        self.assertEqual(result.audit.validated_recordings, 3)
        records = [item.recording for item in result.manifest.records]
        self.assertEqual([item.age.group for item in records], [AgeGroup.CHILD, AgeGroup.ADOLESCENT, AgeGroup.ADULT])
        for item in result.manifest.records:
            self.assertEqual(item.signal.sha256, sha((self.root / item.signal.relative_path).read_bytes()))
            self.assertEqual(item.recording.provenance.annotation_sha256,
                             sha((self.root / item.annotations.relative_path).read_bytes()))
            self.assertEqual(item.recording.provenance.annotation_provenance_ref,
                             "TEST_ONLY_expert_annotation_provenance")
        self.assertEqual(len(build_split_manifest(records).assignments), 3)
        validate_manifest(self.root, result.manifest)

    def test_seizure_free_recordings_require_complete_annotation_review(self):
        zero_data = plan_fixture(self.root, events=())
        result = self.ingest(zero_data)
        self.assertEqual(result.audit.validated_seizure_free_recordings, 3)
        for item in result.manifest.records:
            self.assertEqual(item.recording.annotations, ())
        for coverage in ("unknown", "partial"):
            data = copy.deepcopy(zero_data)
            data["recordings"][0]["annotation_coverage"] = coverage
            self.assert_rejected(data, "annotation_completeness_required")
        zero_data["recordings"][0]["annotation_review"] = {}
        self.assert_rejected(zero_data, "annotation_provenance_review_required")

    def test_malformed_summary_and_empty_annotation_provenance_block_approval(self):
        data = copy.deepcopy(self.data)
        data["recordings"][0]["annotation_provenance_ref"] = " "
        with self.assertRaises(ValidationError):
            self.ingest(data)
        item = self.data["recordings"][0]
        path = self.root / item["summary_path"]
        content = path.read_text().replace("Number of Seizures in File: 1\n", "")
        path.write_text(content)
        item["expected_summary_sha256"] = sha(content.encode())
        self.assert_rejected(self.data, "summary_incomplete_recording_block")

    def test_all_human_reviews_and_explicit_permissions_required(self):
        for field in ("dataset_review", "permissions"):
            data = copy.deepcopy(self.data)
            data[field] = {}
            self.assert_rejected(data, "dataset_permissions_review_required")
        data = copy.deepcopy(self.data)
        data["identities"]["review"] = {}
        self.assert_rejected(data, "identity_review_required")
        for field, code in (("metadata_review", "recording_metadata_review_required"),
                            ("annotation_review", "annotation_provenance_review_required")):
            data = copy.deepcopy(self.data)
            data["recordings"][0][field] = {}
            self.assert_rejected(data, code)

    def test_verified_reviews_cannot_omit_reviewer_date_or_reference(self):
        for field in ("reviewer", "reviewed_on", "evidence_ref"):
            data = verified()
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                HumanReview.model_validate(data)
        with self.assertRaises(ValidationError):
            HumanReview(verified="true")

    def test_missing_unknown_or_ambiguous_identity_never_guessed(self):
        data = copy.deepcopy(self.data)
        data["identities"]["entries"] = data["identities"]["entries"][1:]
        self.assert_rejected(data, "canonical_identity_missing_or_ambiguous")
        data = copy.deepcopy(self.data)
        data["identities"]["entries"].append({"subject_id": "chb90", "canonical_patient_id": "TEST_ONLY_other"})
        with self.assertRaises(ValidationError):
            self.ingest(data)
        data = copy.deepcopy(self.data)
        data["identities"]["entries"][0]["canonical_patient_id"] = " "
        with self.assertRaises(ValidationError):
            self.ingest(data)

    def test_known_alias_requires_both_reviewed_entries_and_same_patient(self):
        data = plan_fixture(self.root, subjects=("chb01", "chb21", "chb92"))
        self.assert_rejected(data, "known_patient_alias_conflict")
        data["identities"]["entries"][1]["canonical_patient_id"] = "TEST_ONLY_chb01"
        result = self.ingest(data)
        self.assertTrue(result.audit.cohort_approved)
        self.assertEqual(result.audit.validated_patients, 2)
        data["identities"]["entries"] = [data["identities"]["entries"][0], data["identities"]["entries"][2]]
        self.assert_rejected(data, "canonical_identity_missing_or_ambiguous")

    def test_alias_sessions_stay_together_in_phase1_splits(self):
        data = plan_fixture(self.root, subjects=("chb01", "chb21", "chb92", "chb93"), years=(10, 12, 17, 20))
        data["identities"]["entries"][1]["canonical_patient_id"] = "TEST_ONLY_chb01"
        result = self.ingest(data)
        manifest = build_split_manifest(item.recording for item in result.manifest.records)
        self.assertEqual(len({item.split for item in manifest.assignments
                              if item.patient_group_id == "TEST_ONLY_chb01"}), 1)

    def test_missing_age_or_under_one_day_precision_rejected(self):
        for value in (None, {"completed_years": 0, "source": "TEST_ONLY"},
                      {"completed_years": 17.8, "source": "TEST_ONLY"}):
            data = copy.deepcopy(self.data)
            data["recordings"][0]["age"] = value
            with self.subTest(age=value), self.assertRaises(ValidationError):
                self.ingest(data)

    def test_no_new_model_or_adult_fallback(self):
        from test_seizure_research import model_spec
        result = self.ingest()
        child, _, adult = [item.recording for item in result.manifest.records]
        self.assertEqual(REGISTERED_MODELS, ())
        self.assertFalse(assess_eligibility(child, None).offline_research_eligible)
        decision = assess_eligibility(child, model_spec(adult))
        self.assertIn("unsupported_age_group", decision.reasons)
        self.assertFalse(decision.live_prediction_allowed)

    def test_missing_summary_target_is_not_assumed_seizure_free(self):
        item = self.data["recordings"][0]
        path = self.root / item["summary_path"]
        content = summary_text(filename="chb90_99.edf", events=()).encode()
        path.write_bytes(content)
        item["expected_summary_sha256"] = sha(content)
        self.assert_rejected(self.data, "recording_absent_from_summary")

    def test_changed_signal_or_summary_requires_new_human_review(self):
        for key in ("expected_signal_sha256", "expected_summary_sha256"):
            data = copy.deepcopy(self.data)
            data["recordings"][0][key] = "a" * 64
            self.assert_rejected(data, "reviewed_signal_hash_mismatch" if "signal" in key else "reviewed_summary_hash_mismatch")

    def test_edf_summary_duration_channel_and_rate_disagreement_rejected(self):
        item = self.data["recordings"][0]
        path = self.root / item["summary_path"]
        original = path.read_text()
        for content, code in ((original.replace("00:01:00", "00:02:00"), "edf_summary_duration_mismatch"),
                              (original.replace("FP1-F7", "FP1-F3"), "edf_summary_channel_mismatch"),
                              (original.replace("256 Hz", "128 Hz"), "sampling_rate_mismatch")):
            data = copy.deepcopy(self.data)
            path.write_text(content)
            data["recordings"][0]["expected_summary_sha256"] = sha(content.encode())
            self.assert_rejected(data, code)

    def test_incompatible_channel_selection_rate_and_units_rejected(self):
        variants = []
        item = copy.deepcopy(self.data["recordings"][0])
        item["channels"] = item["channels"][:1]
        variants.append((item, "channel_selection_must_account_for_all_indices_in_order"))
        item = copy.deepcopy(self.data["recordings"][0])
        item["channels"][0]["source_label"] = "FP1-F3"
        item["channels"][0]["reason"] = "TEST_ONLY_alias"
        variants.append((item, "reviewed_channel_label_mismatch"))
        item = copy.deepcopy(self.data["recordings"][0])
        item["expected_sampling_rate_hz"] = 128
        variants.append((item, "sampling_rate_mismatch"))
        item = copy.deepcopy(self.data["recordings"][0])
        item["montage"]["units"] = "mV"
        variants.append((item, "channel_units_mismatch"))
        for item, code in variants:
            data = copy.deepcopy(self.data)
            data["recordings"][0] = item
            self.assert_rejected(data, code)

    def test_dummy_and_duplicate_channels_need_explicit_index_dispositions(self):
        item = self.data["recordings"][0]
        labels = ("FP1-F7", "-", "FP1-F7")
        signal = edf_bytes("TEST_ONLY_duplicates", labels=labels)
        summary = summary_text(labels=labels).encode()
        (self.root / item["edf_path"]).write_bytes(signal)
        (self.root / item["summary_path"]).write_bytes(summary)
        item.update(expected_signal_sha256=sha(signal), expected_summary_sha256=sha(summary))
        item["channels"] = [
            {"source_index": 0, "source_label": "FP1-F7", "action": "keep", "output_label": "FP1-F7"},
            {"source_index": 1, "source_label": "-", "action": "exclude", "reason": "TEST_ONLY_dummy"},
            {"source_index": 2, "source_label": "FP1-F7", "action": "exclude", "reason": "TEST_ONLY_duplicate"},
        ]
        item["montage"]["channels"] = ["FP1-F7"]
        self.assertTrue(self.ingest().audit.cohort_approved)
        item["channels"][1] = {"source_index": 1, "source_label": "-", "action": "keep", "output_label": "-"}
        item["montage"]["channels"].append("-")
        self.assert_rejected(self.data, "dummy_channel_cannot_be_kept")

    def test_quality_rejection_withholds_entire_cohort(self):
        data = copy.deepcopy(self.data)
        data["recordings"][0]["quality"] = {"status": "unreviewed"}
        result = self.assert_rejected(data, "recording_contract_or_quality_invalid")
        self.assertEqual(result.audit.validated_recordings, 2)
        self.assertEqual(result.audit.rejected_recordings, 1)
        self.assertEqual(set(artifact_bytes(result)), {"audit.json"})

    def test_duplicate_signals_with_different_filenames_reject_cohort(self):
        first, second = self.data["recordings"][:2]
        duplicate = (self.root / first["edf_path"]).read_bytes()
        (self.root / second["edf_path"]).write_bytes(duplicate)
        second["expected_signal_sha256"] = sha(duplicate)
        result = self.ingest()
        self.assertIsNone(result.manifest)
        self.assertIn("cohort_duplicate_or_identity_conflict", result.audit.cohort_errors)

    def test_source_file_missing_or_symlink_outside_root_rejected(self):
        first = self.data["recordings"][0]
        data = copy.deepcopy(self.data)
        data["recordings"][0]["edf_path"] = "chb90/chb90_99.edf"
        self.assert_rejected(data, "recording_absent_from_summary")
        with tempfile.TemporaryDirectory() as elsewhere:
            other = Path(elsewhere) / "external.edf"
            other.write_bytes(edf_bytes())
            path = self.root / first["edf_path"]
            path.unlink()
            path.symlink_to(other)
            self.assert_rejected(self.data, "source_path_outside_root_or_not_regular")
            path.unlink()
            self.assert_rejected(self.data, "source_unavailable")

    def test_paths_and_subject_file_mismatch_rejected(self):
        for path in ("/tmp/patient.edf", "../patient.edf", "chb90/../patient.edf", "chb90\\patient.edf"):
            data = copy.deepcopy(self.data)
            data["recordings"][0]["edf_path"] = path
            with self.subTest(path=path), self.assertRaises(ValidationError):
                self.ingest(data)
        data = copy.deepcopy(self.data)
        data["recordings"][0]["subject_id"] = "chb91"
        self.assert_rejected(data, "source_recording_identifier_mismatch")

    def test_manifest_and_audit_are_deterministic_and_relocatable(self):
        first = self.ingest()
        data = copy.deepcopy(self.data)
        data["recordings"].reverse()
        data["identities"]["entries"].reverse()
        self.assertEqual(artifact_bytes(first), artifact_bytes(self.ingest(data)))
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / "moved"
            shutil.copytree(self.root, target)
            second = ingest_cohort(target, IngestionPlan.model_validate(data))
            self.assertEqual(artifact_bytes(first), artifact_bytes(second))
        roundtrip = CohortManifest.model_validate_json(first.manifest.model_dump_json())
        self.assertEqual(first.manifest.sha256, roundtrip.sha256)

    def test_manifest_tampering_or_source_change_is_detected_on_revalidation(self):
        manifest = self.ingest().manifest
        data = manifest.model_dump(mode="json")
        data["records"][0]["recording"]["age"]["completed_years"] = 18
        with self.assertRaises(IngestionError):
            validate_manifest(self.root, CohortManifest.model_validate(data))
        path = self.root / self.data["recordings"][0]["edf_path"]
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)
        with self.assertRaises(IngestionError):
            validate_manifest(self.root, manifest)

    def test_artifacts_are_json_only_external_and_never_overwritten(self):
        result = self.ingest()
        output = self.root / "artifacts"
        write_artifacts(result, output)
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        self.assertEqual(set(before), {"audit.json", "cohort_manifest.json"})
        self.assertNotIn(b"TEST_ONLY_NEVER_EXPORT_PATIENT", b"".join(before.values()))
        write_artifacts(result, output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        data = copy.deepcopy(self.data)
        data["recordings"][0]["annotation_coverage"] = "unknown"
        with self.assertRaisesRegex(IngestionError, "conflict"):
            write_artifacts(self.ingest(data), output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})
        with self.assertRaisesRegex(IngestionError, "outside_repository"):
            write_artifacts(result, REPOSITORY_ROOT / "reports" / "clinical")

    def test_duplicate_json_keys_rejected(self):
        path = self.root / "bad-plan.json"
        path.write_text('{"adapter_id":"chb_mit","adapter_id":"other"}')
        with self.assertRaisesRegex(IngestionError, "duplicate_review_json_key"):
            load_plan(path)

    def test_cli_approved_and_rejected_runs_and_sanitized_bad_input(self):
        path = self.root / "plan.json"
        path.write_text(json.dumps(self.data))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["ingest", "--root", str(self.root), "--plan", str(path),
                         "--output-dir", str(self.root / "approved")])
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(output.getvalue())["cohort_approved"])
        self.data["recordings"][0]["annotation_review"] = {}
        path.write_text(json.dumps(self.data))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["ingest", "--root", str(self.root), "--plan", str(path),
                                   "--output-dir", str(self.root / "rejected")]), 2)
        self.assertEqual({p.name for p in (self.root / "rejected").iterdir()}, {"audit.json"})
        path.write_text('{"private": "TEST_ONLY_DO_NOT_PRINT"}')
        error = io.StringIO()
        with contextlib.redirect_stderr(error):
            self.assertEqual(main(["ingest", "--root", str(self.root), "--plan", str(path),
                                   "--output-dir", str(self.root / "invalid")]), 2)
        self.assertNotIn("TEST_ONLY_DO_NOT_PRINT", error.getvalue())

    def test_pipeline_does_not_mutate_inputs_or_call_network(self):
        before = copy.deepcopy(self.data)
        with patch("socket.socket", side_effect=AssertionError("Network access is forbidden")):
            self.assertTrue(self.ingest().audit.cohort_approved)
        self.assertEqual(before, self.data)


class CandidateRegistryTests(unittest.TestCase):
    def test_published_source_notes_do_not_grant_local_access(self):
        for item in DATASET_REGISTRY:
            self.assertFalse(item.ready_for_use)
            self.assertIsNone(item.verified_access)
            self.assertIsNone(item.verified_license)
            if item.status == "candidate_unreviewed":
                self.assertTrue(item.annotation_formats)
                self.assertIsNotNone(item.source_checked_on)
                self.assertTrue(item.published_terms_notes)


if __name__ == "__main__":
    unittest.main()
