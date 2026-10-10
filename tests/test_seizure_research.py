"""Contract tests only. Every recording, approval, model, and report is fictitious.

No clinical EEG or actual model is loaded. Fixture approvals/evidence must never
be copied into a real research manifest.
"""

import copy
import hashlib
import subprocess
import sys
import unittest
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from seizure_research.eligibility import (
    REGISTERED_MODELS, SeizureModelSpec, assess_eligibility,
)
from seizure_research.evaluation import (
    EvaluationProvenance, OfflineDetection, evaluate_test_set,
)
from seizure_research.registry import DATASET_REGISTRY, candidates_for_age
from seizure_research.schemas import (
    AgeAtRecording, AgeGroup, EEGRecording, ResearchPermissions,
)
from seizure_research.splitting import (
    SplitManifest, build_split_manifest, validate_split_manifest,
)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def age(years=30, days=None):
    return AgeAtRecording(completed_years=years, completed_days=days, source="TEST_ONLY_age")


def recording(recording_id="r1", patient="p1", subject_age=None):
    return EEGRecording.model_validate({
        "recording_id": recording_id, "subject_id": patient,
        "patient_group_id": patient, "patient_identity_review_ref": "TEST_ONLY_identity",
        "age": (subject_age or age()).model_dump(),
        "montage": {"name": "TEST_ONLY_bipolar", "kind": "bipolar",
                    "channels": ["Fp1-F7", "F7-T7"], "reference": "pairwise", "units": "uV"},
        "sampling_rate_hz": 256.0,
        "device": {"manufacturer": "TEST_ONLY", "model": "TEST_ONLY_device",
                   "acquisition_software_version": "TEST_ONLY_v1", "clinical_eeg": True},
        "preprocessing_id": "TEST_ONLY_raw_v1",
        "provenance": {"dataset_id": "TEST_ONLY", "version": "TEST_ONLY_v1",
                       "source_url": "https://example.invalid/test-only",
                       "source_recording_id": recording_id, "signal_sha256": sha(recording_id),
                       "annotation_sha256": sha("TEST_ONLY_annotations"),
                       "annotation_source": "TEST_ONLY_events.csv"},
        "duration_seconds": 3600.0, "annotation_coverage": "complete",
        "annotation_protocol_ref": "TEST_ONLY_adjudication",
        "annotations": [{"event_id": "e1", "start_seconds": 10.0, "end_seconds": 20.0,
                         "seizure_type": "TEST_ONLY", "annotator_ref": "TEST_ONLY_expert",
                         "clinical_review_ref": "TEST_ONLY_review"}],
        "quality": {"status": "acceptable", "review_ref": "TEST_ONLY_quality"},
        "permissions": {"status": "approved", "deidentified": True,
                        "terms_review_ref": "TEST_ONLY_terms", "ethics_review_ref": "TEST_ONLY_ethics",
                        "access_review_ref": "TEST_ONLY_access", "reviewed_by": "TEST_ONLY_reviewer",
                        "reviewed_on": "2026-10-11"},
    })


def modified(record, **changes):
    data = record.model_dump(mode="json")
    data.update(changes)
    return EEGRecording.model_validate(data)


def model_spec(record=None):
    record = record or recording()
    return SeizureModelSpec.model_validate({
        "model_id": "TEST_ONLY_model", "version": "TEST_ONLY_v1", "model_sha256": sha("TEST_ONLY_model"),
        "purpose": "offline_seizure_event_research", "supported_age_groups": [record.age.group],
        "acquisition_profiles": [{
            "profile_id": "TEST_ONLY_profile", "montage": record.montage.model_dump(),
            "sampling_rate_hz": record.sampling_rate_hz, "device": record.device.model_dump(),
            "preprocessing_id": record.preprocessing_id,
        }],
        "validation_evidence": [{
            "age_group": record.age.group, "acquisition_profile_id": "TEST_ONLY_profile",
            "model_sha256": sha("TEST_ONLY_model"), "dataset_refs": ["TEST_ONLY_cohort"],
            "split_manifest_sha256": sha("TEST_ONLY_split"),
            "evaluation_protocol_ref": "TEST_ONLY_protocol", "metrics_report_ref": "TEST_ONLY_metrics",
            "review_ref": "TEST_ONLY_review", "reviewed": True, "patient_independent": True,
            "clinical_annotations": True, "patient_count": 10, "seizure_event_count": 20,
            "evaluated_hours": 100.0,
        }],
    })


def cohort(count=10):
    return [recording(f"r{i}", f"p{i}") for i in range(count)]


def evaluation_inputs(records):
    manifest = build_split_manifest(records)
    patients = {split: sorted({a.patient_group_id for a in manifest.assignments if a.split == split})
                for split in ("train", "validation", "test")}
    provenance = EvaluationProvenance(
        model_id="TEST_ONLY", model_version="TEST_ONLY_v1", model_sha256=sha("TEST_ONLY_model"),
        split_manifest_sha256=manifest.sha256, prediction_artifact_sha256=sha("TEST_ONLY_predictions"),
        frozen_protocol_ref="TEST_ONLY_protocol", training_patient_ids=patients["train"],
        selection_patient_ids=patients["validation"],
    )
    detections = {a.recording_id: [] for a in manifest.assignments if a.split == "test"}
    return manifest, provenance, detections


class AgeTests(unittest.TestCase):
    def test_every_age_boundary(self):
        for years, days, expected in (
            (0, 0, AgeGroup.NEONATAL), (0, 28, AgeGroup.NEONATAL),
            (0, 29, AgeGroup.INFANT), (0, 364, AgeGroup.INFANT), (0, 365, AgeGroup.INFANT),
            (1, None, AgeGroup.CHILD), (12, None, AgeGroup.CHILD),
            (13, None, AgeGroup.ADOLESCENT), (17, None, AgeGroup.ADOLESCENT),
            (18, None, AgeGroup.ADULT), (64, None, AgeGroup.ADULT),
            (65, None, AgeGroup.OLDER_ADULT), (100, None, AgeGroup.OLDER_ADULT),
        ):
            with self.subTest(years=years, days=days):
                self.assertEqual(age(years, days).group, expected)

    def test_under_one_requires_days_and_rejects_inconsistent_precision(self):
        for years, days in ((0, None), (0, 366), (1, 365), (-1, None), (0, -1),
                            (1.5, None), (True, None), ("18", None)):
            with self.subTest(years=years, days=days), self.assertRaises(ValidationError):
                age(years, days)

    def test_calendar_birthdays_and_leap_years(self):
        cases = (
            (date(2023, 3, 1), date(2024, 2, 29), AgeGroup.INFANT),
            (date(2023, 3, 1), date(2024, 3, 1), AgeGroup.CHILD),
            (date(2024, 2, 29), date(2025, 2, 28), AgeGroup.INFANT),
            (date(2024, 2, 29), date(2025, 3, 1), AgeGroup.CHILD),
            (date(2008, 10, 12), date(2026, 10, 11), AgeGroup.ADOLESCENT),
            (date(2008, 10, 12), date(2026, 10, 12), AgeGroup.ADULT),
        )
        for born, recorded, expected in cases:
            with self.subTest(born=born, recorded=recorded):
                self.assertEqual(AgeAtRecording.from_dates(born, recorded, source="TEST_ONLY").group, expected)

    def test_recording_before_birth_rejected(self):
        with self.assertRaises(ValueError):
            AgeAtRecording.from_dates(date(2026, 1, 2), date(2026, 1, 1), source="TEST_ONLY")

    def test_unknown_age_never_defaults_to_adult(self):
        for bad in ({}, {"completed_years": None, "source": "TEST_ONLY"},
                    {"completed_years": 18, "source": " "}):
            with self.subTest(bad=bad), self.assertRaises(ValidationError):
                AgeAtRecording.model_validate(bad)


class MetadataTests(unittest.TestCase):
    def test_json_roundtrip(self):
        item = recording()
        self.assertEqual(EEGRecording.model_validate_json(item.model_dump_json()), item)

    def test_every_required_metadata_field_is_required(self):
        for field, definition in EEGRecording.model_fields.items():
            if definition.is_required():
                data = recording().model_dump(mode="json")
                del data[field]
                with self.subTest(field=field), self.assertRaises(ValidationError):
                    EEGRecording.model_validate(data)

    def test_missing_nested_metadata_rejected(self):
        for section, field in (("age", "source"), ("montage", "reference"),
                               ("device", "model"), ("device", "clinical_eeg"),
                               ("provenance", "annotation_sha256"), ("provenance", "version")):
            data = recording().model_dump(mode="json")
            del data[section][field]
            with self.subTest(section=section, field=field), self.assertRaises(ValidationError):
                EEGRecording.model_validate(data)

    def test_nonfinite_nonpositive_and_coerced_signal_metadata_rejected(self):
        for field in ("sampling_rate_hz", "duration_seconds"):
            for value in (0, -1, float("nan"), float("inf"), True, "256"):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    modified(recording(), **{field: value})

    def test_empty_duplicate_channels_and_unknown_fields_rejected(self):
        for channels in ([], ["Fp1-F7", "Fp1-F7"], [" "]):
            data = recording().montage.model_dump()
            data["channels"] = channels
            with self.subTest(channels=channels), self.assertRaises(ValidationError):
                modified(recording(), montage=data)
        with self.assertRaises(ValidationError):
            modified(recording(), live_prediction=True)

    def test_event_intervals_ids_and_review_are_validated(self):
        for change in ({"start_seconds": -1}, {"end_seconds": 10}, {"end_seconds": 3601},
                       {"start_seconds": float("nan")}, {"clinical_review_ref": " "}):
            data = recording().model_dump(mode="json")
            data["annotations"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValidationError):
                EEGRecording.model_validate(data)
        data = recording().model_dump(mode="json")
        data["annotations"].append(copy.deepcopy(data["annotations"][0]))
        with self.assertRaises(ValidationError):
            EEGRecording.model_validate(data)
        data["annotations"][1]["event_id"] = "another"
        with self.assertRaisesRegex(ValidationError, "overlapping"):
            EEGRecording.model_validate(data)

    def test_quality_exclusions_cannot_overlap_or_exceed_duration(self):
        for gaps in (({"start_seconds": 3600, "end_seconds": 3601},),
                     ({"start_seconds": 100, "end_seconds": 200},
                      {"start_seconds": 150, "end_seconds": 250})):
            quality = recording().quality.model_dump()
            quality["excluded_intervals"] = gaps
            with self.subTest(gaps=gaps), self.assertRaises(ValidationError):
                modified(recording(), quality=quality)

    def test_permissions_default_to_unknown_and_approval_needs_reviews(self):
        self.assertEqual(ResearchPermissions().status, "unknown")
        for field in ("deidentified", "terms_review_ref", "ethics_review_ref", "access_review_ref",
                      "reviewed_by", "reviewed_on"):
            data = recording().permissions.model_dump(mode="json")
            del data[field]
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ResearchPermissions.model_validate(data)


class EligibilityTests(unittest.TestCase):
    def assert_rejected(self, record, model, reason):
        result = assess_eligibility(record, model)
        self.assertFalse(result.offline_research_eligible)
        self.assertIn(reason, result.reasons)
        self.assertFalse(result.live_prediction_allowed)
        self.assertFalse(result.medical_alerts_allowed)

    def test_explicit_matching_fixture_is_only_eligible_for_offline_research(self):
        result = assess_eligibility(recording(), model_spec())
        self.assertTrue(result.offline_research_eligible)
        self.assertEqual(result.reasons, ())
        self.assertFalse(result.live_prediction_allowed)
        self.assertFalse(result.medical_alerts_allowed)

    def test_no_models_bundled_and_no_model_fallback(self):
        self.assertEqual(REGISTERED_MODELS, ())
        self.assert_rejected(recording(), None, "no_model_registered")

    def test_every_cross_age_pair_is_rejected(self):
        ages = (age(0, 28), age(0, 29), age(12), age(17), age(18), age(65))
        for source_age in ages:
            for target_age in ages:
                if source_age.group != target_age.group:
                    with self.subTest(model=source_age.group, recording=target_age.group):
                        self.assert_rejected(recording(subject_age=target_age),
                                             model_spec(recording(subject_age=source_age)),
                                             "unsupported_age_group")

    def test_adding_supported_age_cannot_reuse_adult_evidence(self):
        for target in (age(0, 0), age(0, 29), age(10), age(17), age(65)):
            data = model_spec().model_dump(mode="json")
            data["supported_age_groups"].append(target.group.value)
            with self.subTest(age=target.group):
                self.assert_rejected(recording(subject_age=target), data, "missing_age_specific_validation")

    def test_missing_and_inadequate_evidence_rejected(self):
        model = model_spec().model_dump(mode="json")
        model["validation_evidence"] = []
        self.assert_rejected(recording(), model, "missing_age_specific_validation")
        for field, value in (("reviewed", False), ("patient_independent", False),
                             ("clinical_annotations", False), ("model_sha256", sha("different"))):
            data = model_spec().model_dump(mode="json")
            data["validation_evidence"][0][field] = value
            with self.subTest(field=field):
                self.assert_rejected(recording(), data, "missing_age_specific_validation")

    def test_missing_evidence_reference_rejected(self):
        for field in ("metrics_report_ref", "review_ref", "split_manifest_sha256", "dataset_refs"):
            data = model_spec().model_dump(mode="json")
            del data["validation_evidence"][0][field]
            with self.subTest(field=field):
                self.assert_rejected(recording(), data, "invalid_model_metadata")

    def test_neonatal_requires_specialist_evidence(self):
        neonate = recording(subject_age=age(0, 10))
        data = model_spec(neonate).model_dump(mode="json")
        self.assert_rejected(neonate, data, "missing_neonatal_specialist_review")
        data["validation_evidence"][0]["neonatal_specialist_review_ref"] = "TEST_ONLY_neonatal_review"
        self.assertTrue(assess_eligibility(neonate, data).offline_research_eligible)

    def test_device_manufacturer_model_version_and_consumer_status_rejected(self):
        for field, value in (("manufacturer", "other"), ("model", "other"),
                             ("acquisition_software_version", "other"), ("clinical_eeg", False)):
            data = recording().model_dump(mode="json")
            data["device"][field] = value
            with self.subTest(field=field):
                self.assert_rejected(data, model_spec(), "unsupported_device")

    def test_montage_order_reference_kind_and_units_rejected(self):
        for field, value in (("channels", ["F7-T7", "Fp1-F7"]), ("channels", ["Fp1-F7"]),
                             ("reference", "other"), ("kind", "referential"), ("units", "V")):
            data = recording().model_dump(mode="json")
            data["montage"][field] = value
            with self.subTest(field=field):
                self.assert_rejected(data, model_spec(), "incompatible_montage")

    def test_unsupported_rate_and_preprocessing_rejected(self):
        self.assert_rejected(modified(recording(), sampling_rate_hz=512), model_spec(), "unsupported_sampling_rate")
        self.assert_rejected(modified(recording(), preprocessing_id="attention_v1"), model_spec(), "unsupported_preprocessing")

    def test_individually_supported_fields_cannot_form_unvalidated_combination(self):
        model = model_spec().model_dump(mode="json")
        second = copy.deepcopy(model["acquisition_profiles"][0])
        second.update(profile_id="TEST_ONLY_other", sampling_rate_hz=512)
        second["device"]["model"] = "TEST_ONLY_other_device"
        model["acquisition_profiles"].append(second)
        record = modified(recording(), sampling_rate_hz=512)
        self.assert_rejected(record, model, "unsupported_acquisition_combination")

    def test_age_evidence_does_not_transfer_between_acquisition_profiles(self):
        data = model_spec().model_dump(mode="json")
        second = copy.deepcopy(data["acquisition_profiles"][0])
        second.update(profile_id="TEST_ONLY_other", sampling_rate_hz=512)
        data["acquisition_profiles"].append(second)
        self.assert_rejected(modified(recording(), sampling_rate_hz=512), data, "missing_age_specific_validation")

    def test_missing_or_unchecked_metadata_fails_closed(self):
        for field in ("age", "device", "montage", "permissions", "provenance"):
            data = recording().model_dump(mode="json")
            del data[field]
            with self.subTest(field=field):
                self.assert_rejected(data, model_spec(), "invalid_recording_metadata")
        self.assert_rejected(recording().model_copy(update={"age": None}), model_spec(), "invalid_recording_metadata")
        self.assert_rejected(recording(), model_spec().model_copy(update={"purpose": "attention"}), "invalid_model_metadata")

    def test_unapproved_partial_or_unreviewed_recordings_rejected(self):
        for changes in ({"permissions": {}}, {"permissions": {"status": "denied"}},
                        {"quality": {"status": "unreviewed"}}, {"quality": {"status": "unusable"}},
                        {"annotation_coverage": "partial"}, {"annotation_coverage": "unknown"}):
            with self.subTest(changes=changes):
                self.assert_rejected(modified(recording(), **changes), model_spec(), "recording_not_research_ready")

    def test_no_valid_time_or_events_in_exclusions_rejected(self):
        for start, end in ((0, 3600), (15, 25)):
            quality = {"status": "acceptable", "review_ref": "TEST_ONLY",
                       "excluded_intervals": [{"start_seconds": start, "end_seconds": end}]}
            self.assert_rejected(modified(recording(), quality=quality), model_spec(), "recording_not_research_ready")


class RegistryTests(unittest.TestCase):
    def test_registry_is_unapproved_and_has_all_age_groups(self):
        self.assertEqual(len({item.dataset_id for item in DATASET_REGISTRY}), len(DATASET_REGISTRY))
        for item in DATASET_REGISTRY:
            self.assertFalse(item.ready_for_use)
            self.assertIsNone(item.verified_license)
            self.assertIsNone(item.verified_access)
            self.assertIsNone(item.ethics_approval_ref)
        for group in AgeGroup:
            self.assertTrue(candidates_for_age(group))
        self.assertEqual(candidates_for_age(AgeGroup.INFANT)[0].status, "unresolved_gap")


class SplitTests(unittest.TestCase):
    def test_split_is_reproducible_independent_of_input_order_and_roundtrips(self):
        records = cohort()
        first = build_split_manifest(records)
        second = build_split_manifest(reversed(records))
        self.assertEqual(first, second)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(SplitManifest.model_validate_json(first.model_dump_json()), first)
        self.assertNotEqual(first, build_split_manifest(records, seed=43))
        validate_split_manifest(records, first)

    def test_all_sessions_aliases_and_ages_of_patient_stay_together(self):
        records = cohort()
        extra = recording("alias_recording", "p0", age(17))
        data = extra.model_dump(mode="json")
        data["subject_id"] = "another_dataset_alias"
        data["provenance"]["dataset_id"] = "TEST_ONLY_other_dataset"
        records.append(EEGRecording.model_validate(data))
        manifest = build_split_manifest(records)
        memberships = {a.split for a in manifest.assignments if a.patient_group_id == "p0"}
        self.assertEqual(len(memberships), 1)
        self.assertEqual(len(manifest.assignments), len(records))
        self.assertEqual({a.split for a in manifest.assignments}, {"train", "validation", "test"})

    def test_partition_counts_are_whole_patients(self):
        manifest = build_split_manifest(cohort())
        self.assertEqual({split: sum(a.split == split for a in manifest.assignments)
                          for split in ("train", "validation", "test")},
                         {"train": 6, "validation": 2, "test": 2})
        self.assertEqual(len({a.split for a in build_split_manifest(cohort(3)).assignments}), 3)

    def test_too_few_patients_invalid_fractions_and_seed_rejected(self):
        with self.assertRaises(ValueError):
            build_split_manifest(cohort(2))
        for kwargs in ({"train_fraction": 0}, {"train_fraction": .9, "validation_fraction": .2},
                       {"validation_fraction": float("nan")}, {"seed": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build_split_manifest(cohort(), **kwargs)

    def test_duplicate_recordings_and_duplicate_signal_files_rejected(self):
        records = cohort()
        with self.assertRaisesRegex(ValueError, "Duplicate recording"):
            build_split_manifest([*records, records[0]])
        duplicate = recording("r_new", "new_patient")
        provenance = duplicate.provenance.model_dump(mode="json")
        provenance["signal_sha256"] = records[0].provenance.signal_sha256
        with self.assertRaisesRegex(ValueError, "Duplicate signal"):
            build_split_manifest([*records, modified(duplicate, provenance=provenance)])

    def test_conflicting_patient_identity_for_same_subject_is_rejected(self):
        extra = modified(recording("r_extra", "p_other"), subject_id="p0")
        with self.assertRaisesRegex(ValueError, "conflicting patient"):
            build_split_manifest([*cohort(), extra])

    def test_repeated_source_recording_in_another_release_is_rejected(self):
        extra = recording("r_extra", "p_other")
        provenance = extra.provenance.model_dump(mode="json")
        provenance.update(source_recording_id="r0", version="TEST_ONLY_v2")
        with self.assertRaisesRegex(ValueError, "Repeated source"):
            build_split_manifest([*cohort(), modified(extra, provenance=provenance)])

    def test_changed_age_annotations_permissions_or_membership_invalidates_manifest(self):
        records = cohort()
        manifest = build_split_manifest(records)
        for change in ({"age": age(17).model_dump()}, {"annotations": []}, {"permissions": {}}):
            changed = [modified(records[0], **change), *records[1:]]
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_split_manifest(changed, manifest)
        data = manifest.model_dump(mode="json")
        train = next(item for item in data["assignments"] if item["split"] == "train")
        test = next(item for item in data["assignments"] if item["split"] == "test")
        train["split"], test["split"] = test["split"], train["split"]
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_split_manifest(records, SplitManifest.model_validate(data))
        with self.assertRaises(ValueError):
            validate_split_manifest(records[:-1], manifest)

    def test_explicit_patient_leakage_rejected(self):
        records = [*cohort(), recording("r_extra", "p0")]
        data = build_split_manifest(records).model_dump(mode="json")
        item = next(a for a in data["assignments"] if a["recording_id"] == "r_extra")
        item["split"] = "test" if item["split"] != "test" else "train"
        with self.assertRaisesRegex(ValidationError, "Patient leakage"):
            SplitManifest.model_validate(data)


class EvaluationTests(unittest.TestCase):
    def test_known_event_sensitivity_false_alarms_latency_and_subgroups(self):
        records = cohort()
        manifest, provenance, detections = evaluation_inputs(records)
        first, second = sorted(detections)
        detections[first] = [OfflineDetection(detection_id="d1", decision_seconds=12),
                             OfflineDetection(detection_id="duplicate_alarm", decision_seconds=15),
                             OfflineDetection(detection_id="outside", decision_seconds=100)]
        report = evaluate_test_set(records, manifest, detections, provenance)
        self.assertEqual(report.overall.event_sensitivity, .5)
        self.assertEqual(report.overall.missed_events, 1)
        self.assertEqual(report.overall.false_alarms, 2)
        self.assertEqual(report.overall.false_alarms_per_hour, 1)
        self.assertEqual(report.overall.detection_latencies_seconds, (2,))
        self.assertEqual(report.overall.median_detection_latency_seconds, 2)
        self.assertEqual(report.macro_patient_event_sensitivity, .5)
        self.assertEqual(report.patients_with_reference_events, 2)
        self.assertEqual(report.per_age_group[AgeGroup.ADULT], report.overall)
        self.assertIsNone(report.per_age_group[AgeGroup.NEONATAL].event_sensitivity)
        self.assertEqual(report.per_age_group[AgeGroup.NEONATAL].patient_count, 0)
        self.assertEqual(len(report.per_patient), 2)
        self.assertEqual(evaluate_test_set(reversed(records), manifest, detections, provenance), report)

    def test_reference_boundaries_use_half_open_intervals(self):
        records = cohort(3)
        manifest, provenance, detections = evaluation_inputs(records)
        key = next(iter(detections))
        for time, sensitivity, false_alarms in ((9.99, 0, 1), (10, 1, 0), (19.99, 1, 0), (20, 0, 1)):
            detections[key] = [OfflineDetection(detection_id="d", decision_seconds=time)]
            with self.subTest(time=time):
                metrics = evaluate_test_set(records, manifest, detections, provenance).overall
                self.assertEqual(metrics.event_sensitivity, sensitivity)
                self.assertEqual(metrics.false_alarms, false_alarms)

    def test_no_events_and_no_detections_are_not_perfect_sensitivity(self):
        records = [modified(item, annotations=[]) for item in cohort(3)]
        manifest, provenance, detections = evaluation_inputs(records)
        report = evaluate_test_set(records, manifest, detections, provenance)
        self.assertIsNone(report.overall.event_sensitivity)
        self.assertIsNone(report.overall.median_detection_latency_seconds)
        self.assertEqual(report.overall.false_alarms_per_hour, 0)
        self.assertIsNone(report.macro_patient_event_sensitivity)
        self.assertEqual(report.patients_with_reference_events, 0)

    def test_missed_events_remain_in_denominator(self):
        records = cohort(3)
        manifest, provenance, detections = evaluation_inputs(records)
        metrics = evaluate_test_set(records, manifest, detections, provenance).overall
        self.assertEqual(metrics.event_sensitivity, 0)
        self.assertEqual(metrics.missed_events, 1)
        self.assertIsNone(metrics.median_detection_latency_seconds)

    def test_excluded_time_is_removed_from_false_alarm_denominator(self):
        quality = {"status": "acceptable", "review_ref": "TEST_ONLY",
                   "excluded_intervals": [{"start_seconds": 1800, "end_seconds": 3600}]}
        records = [modified(item, quality=quality) for item in cohort(3)]
        manifest, provenance, detections = evaluation_inputs(records)
        key = next(iter(detections))
        detections[key] = [OfflineDetection(detection_id="fp", decision_seconds=100)]
        metrics = evaluate_test_set(records, manifest, detections, provenance).overall
        self.assertEqual(metrics.evaluated_hours, .5)
        self.assertEqual(metrics.excluded_hours, .5)
        self.assertEqual(metrics.false_alarms_per_hour, 2)
        detections[key] = [OfflineDetection(detection_id="gap", decision_seconds=1800)]
        with self.assertRaisesRegex(ValueError, "excluded"):
            evaluate_test_set(records, manifest, detections, provenance)

    def test_test_coverage_cannot_be_omitted_or_include_training_records(self):
        records = cohort()
        manifest, provenance, detections = evaluation_inputs(records)
        incomplete = dict(detections)
        del incomplete[next(iter(incomplete))]
        with self.assertRaisesRegex(ValueError, "exactly all test"):
            evaluate_test_set(records, manifest, incomplete, provenance)
        training_id = next(a.recording_id for a in manifest.assignments if a.split == "train")
        with self.assertRaisesRegex(ValueError, "exactly all test"):
            evaluate_test_set(records, manifest, {**detections, training_id: []}, provenance)

    def test_training_and_selection_provenance_cannot_include_test_patients(self):
        records = cohort()
        manifest, provenance, detections = evaluation_inputs(records)
        patient = next(a.patient_group_id for a in manifest.assignments if a.split == "test")
        for field in ("training_patient_ids", "selection_patient_ids"):
            data = provenance.model_dump(mode="json")
            data[field].append(patient)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "violates the split"):
                evaluate_test_set(records, manifest, detections, EvaluationProvenance.model_validate(data))
        data = provenance.model_dump(mode="json")
        data["split_manifest_sha256"] = sha("other_manifest")
        with self.assertRaisesRegex(ValueError, "different split"):
            evaluate_test_set(records, manifest, detections, EvaluationProvenance.model_validate(data))

    def test_invalid_or_duplicate_decisions_rejected(self):
        records = cohort(3)
        manifest, provenance, detections = evaluation_inputs(records)
        key = next(iter(detections))
        for decisions in ([{"detection_id": "bad", "decision_seconds": -1}],
                          [{"detection_id": "bad", "decision_seconds": float("nan")}],
                          [{"detection_id": "bad", "decision_seconds": 3600}],
                          [{"detection_id": "duplicate", "decision_seconds": 11}] * 2):
            with self.subTest(decisions=decisions), self.assertRaises(ValueError):
                evaluate_test_set(records, manifest, {key: decisions}, provenance)

    def test_age_subgroup_uses_recording_age_and_patient_macro_is_unweighted(self):
        records = cohort()
        manifest, _, detections = evaluation_inputs(records)
        first, second = sorted(detections)
        updated = []
        for item in records:
            if item.recording_id == first:
                data = item.model_dump(mode="json")
                data["age"] = age(17).model_dump(mode="json")
                additional = copy.deepcopy(data["annotations"][0])
                additional.update(event_id="e2", start_seconds=30, end_seconds=40)
                data["annotations"].append(additional)
                item = EEGRecording.model_validate(data)
            updated.append(item)
        manifest, provenance, detections = evaluation_inputs(updated)
        detections[first] = [OfflineDetection(detection_id="d1", decision_seconds=11),
                             OfflineDetection(detection_id="d2", decision_seconds=31)]
        report = evaluate_test_set(updated, manifest, detections, provenance)
        self.assertEqual(report.overall.event_sensitivity, 2 / 3)
        self.assertEqual(report.macro_patient_event_sensitivity, .5)
        self.assertEqual(report.per_age_group[AgeGroup.ADOLESCENT].event_sensitivity, 1)
        self.assertEqual(report.per_age_group[AgeGroup.ADULT].event_sensitivity, 0)


class IsolationTests(unittest.TestCase):
    def test_import_has_no_application_model_database_or_sensor_dependency(self):
        script = """
import sys
import seizure_research.schemas
import seizure_research.eligibility
import seizure_research.registry
import seizure_research.splitting
import seizure_research.evaluation
for prefix in ('backend', 'dashboard', 'src', 'joblib', 'sqlite3', 'streamlit', 'fastapi'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
"""
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
