"""Regression cases for excluded samples, provenance, resource limits and timebases."""

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.signal import sosfiltfilt

import test_seizure_processing as fixtures
import test_seizure_research as contracts
from seizure_research.eligibility import assess_eligibility
from seizure_research.edf import inspect_edf
from seizure_research.ingestion import ingest_cohort
from seizure_research.preprocessing import (
    ChannelMapping, PreprocessingConfig, _filter_sos, _resample_channel,
    align_reference_windows, make_age_config, preprocess_recording, process_approved_recording,
)
from seizure_research.quality import QualityInput, QualityThresholds, assess_signal_quality
from seizure_research.schemas import AgeGroup, EEGRecording, TimeInterval
from seizure_research.splitting import WindowSplitManifest, build_split_manifest
from seizure_research.waveform import read_edf


class ProcessingSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = fixtures._fixture(self.root)
        self.manifest = ingest_cohort(self.root, self.plan).manifest
        self.recording = self.manifest.records[0].recording
        self.path = self.manifest.records[0].signal.relative_path
        self.decoded = read_edf(self.root, self.path, channel_indices=(0, 1), start_seconds=0, end_seconds=60)
        self.config = make_age_config(AgeGroup.CHILD, tuple(
            ChannelMapping(source_index=i, source_label=label, target_label=label, expected_source_rate_hz=128)
            for i, label in enumerate(self.recording.montage.channels)),
            target_sampling_rate_hz=128, research_review_ref="TEST_ONLY")

    def process(self, decoded=None, config=None, **kwargs):
        return preprocess_recording(self.recording, decoded or self.decoded, config or self.config,
                                    approved_manifest=self.manifest, **kwargs)

    def test_saturation_is_assessed_before_digital_samples_are_lost(self):
        payload = bytearray((self.root / self.path).read_bytes())
        for record in range(60):
            for sample in range(0, 128, 32):
                offset = 768 + record * 512 + sample * 2
                payload[offset:offset + 2] = (32767).to_bytes(2, "little", signed=True)
        (self.root / self.path).write_bytes(payload)
        self.plan["recordings"][0]["expected_signal_sha256"] = hashlib.sha256(payload).hexdigest()
        self.manifest = ingest_cohort(self.root, self.plan).manifest
        self.recording = self.manifest.records[0].recording
        self.decoded = read_edf(self.root, self.path, channel_indices=(0, 1), start_seconds=0, end_seconds=60)
        output = self.process()
        self.assertGreater(output.report.quality.channel_reports[0].saturated_samples, 0)
        self.assertTrue(any("saturation" in r for w in output.report.quality.windows for r in w.reasons))
        self.assertEqual(output.report.status, "unusable")
        self.assertTrue(np.isnan(output.samples_uv).all())
        self.assertEqual({w.label for w in output.report.windows}, {"excluded"})

    def test_nonfinite_native_samples_are_excluded_without_polluting_other_segments(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                samples = self.decoded.channels[0].samples_uv.copy()
                samples[100] = value
                changed = replace(self.decoded, channels=(replace(self.decoded.channels[0], samples_uv=samples),
                                                         self.decoded.channels[1]))
                output = self.process(changed)
                self.assertEqual(output.report.quality.channel_reports[0].nonfinite_samples, 1)
                self.assertTrue(np.isnan(output.samples_uv[:640]).all())
                self.assertTrue(np.isfinite(output.samples_uv[25 * 128:30 * 128]).all())

    def test_resampling_and_filtering_do_not_cross_reviewed_exclusions(self):
        self.plan["recordings"][0]["quality"]["excluded_intervals"] = [
            {"start_seconds": 30.0, "end_seconds": 35.0}]
        self.manifest = ingest_cohort(self.root, self.plan).manifest
        self.recording = self.manifest.records[0].recording
        config = PreprocessingConfig.model_validate({**self.config.model_dump(),
            "target_sampling_rate_hz": 100.0, "line_noise_hz": None, "allow_resampling": True,
            "quality_thresholds": QualityThresholds(min_usable_fraction=0.1)})
        first = self.process(config=config)
        samples = self.decoded.channels[0].samples_uv.copy()
        mask = (self.decoded.channels[0].timestamps_seconds >= 30) & (self.decoded.channels[0].timestamps_seconds < 35)
        samples[mask] += 75
        altered = replace(self.decoded, channels=(replace(self.decoded.channels[0], samples_uv=samples),
                                                 self.decoded.channels[1]))
        second = self.process(altered, config)
        np.testing.assert_array_equal(first.samples_uv, second.samples_uv)
        self.assertTrue(np.isfinite(first.samples_uv).any())
        self.assertTrue(np.isnan(first.samples_uv[3000:3500]).all())

    def test_irregular_timebase_is_rejected_before_interpolation(self):
        times = self.decoded.channels[0].timestamps_seconds.copy()
        times[320:] += 0.01
        changed = replace(self.decoded, channels=(replace(self.decoded.channels[0], timestamps_seconds=times),
                                                 self.decoded.channels[1]))
        with self.assertRaisesRegex(ValueError, "timebase_or_shape"):
            self.process(changed)

    def test_threshold_override_is_part_of_effective_configuration_hash(self):
        first = self.process()
        second = self.process(thresholds=QualityThresholds(extreme_amplitude_uv=500))
        self.assertNotEqual(first.report.config_sha256, second.report.config_sha256)
        self.assertEqual(second.report.configuration.quality_thresholds.extreme_amplitude_uv, 500)
        self.assertEqual(first.report.reference_annotations, self.recording.annotations)
        self.assertEqual(first.report.annotation_review_ref, "TEST_ONLY_review")
        self.assertEqual(set(first.report.runtime_versions), {"python", "numpy", "scipy"})

    def test_file_entry_point_revalidates_annotation_bytes_and_never_logs_headers(self):
        output = process_approved_recording(self.root, self.manifest, self.recording.recording_id,
                                            self.config, start_seconds=0, end_seconds=60)
        self.assertEqual(output.report.cohort_manifest_sha256, self.manifest.sha256)
        self.assertNotIn("TEST_ONLY_HEADER_ID", output.report.model_dump_json())
        path = self.root / self.manifest.records[0].annotations.relative_path
        path.write_text(path.read_text() + "unreviewed change\n")
        with self.assertRaisesRegex(ValueError, "cohort_manifest_or_sources_changed"):
            process_approved_recording(self.root, self.manifest, self.recording.recording_id,
                                       self.config, start_seconds=0, end_seconds=60)

    def test_swapped_source_channels_cannot_keep_approved_output_labels(self):
        channels = tuple(ChannelMapping(source_index=1-i, source_label=self.config.channels[1-i].source_label,
                                        target_label=item.target_label, expected_source_rate_hz=128)
                         for i, item in enumerate(self.config.channels))
        config = PreprocessingConfig.model_validate({**self.config.model_dump(), "channels": channels})
        with self.assertRaisesRegex(ValueError, "approved_selection"):
            self.process(config=config)

    def test_unverified_annotation_review_cannot_be_processed(self):
        plan = self.manifest.review_plan.model_dump(mode="json")
        plan["recordings"][0]["annotation_review"]["verified"] = False
        altered = self.manifest.model_copy(update={"review_plan": plan})
        with self.assertRaisesRegex(ValueError, "review_gates_missing"):
            preprocess_recording(self.recording, self.decoded, self.config, approved_manifest=altered)

    def test_processing_output_memory_bound_precedes_allocation(self):
        config = PreprocessingConfig.model_validate({**self.config.model_dump(),
            "target_sampling_rate_hz": 1_000_000.0, "allow_resampling": True})
        with patch("seizure_research.preprocessing.np.arange", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(ValueError, "processed_sample_limit"):
                self.process(config=config)

    def test_filter_passband_stopband_and_phase(self):
        times = np.arange(128 * 60) / 128
        desired = np.sin(2 * np.pi * 10 * times)
        noise = np.sin(2 * np.pi * 50 * times)
        filtered = sosfiltfilt(_filter_sos(self.config), desired + noise)
        selected = slice(1280, -1280)
        self.assertGreater(np.corrcoef(filtered[selected], desired[selected])[0, 1], 0.99)
        self.assertLess(np.std(filtered[selected] - desired[selected]), 0.02)

    def test_downsampling_suppresses_above_nyquist_component(self):
        times = np.arange(128 * 60) / 128
        grid = np.arange(64 * 60) / 64
        source = np.sin(2 * np.pi * 10 * times) + np.sin(2 * np.pi * 50 * times)
        converted = _resample_channel(times, source, 128, 64, grid)
        self.assertLess(np.std((converted - np.sin(2 * np.pi * 10 * grid))[640:-640]), 0.02)

    def test_bounded_coverage_and_read_only_arrays(self):
        decoded = read_edf(self.root, self.path, channel_indices=(0, 1), start_seconds=5, end_seconds=55)
        output = self.process(decoded)
        self.assertFalse(output.report.coverage_complete)
        self.assertAlmostEqual(output.report.coverage_fraction, 50 / 60)
        self.assertGreaterEqual(output.timestamps_seconds[0], 5)
        self.assertLess(output.timestamps_seconds[-1], 55)
        with self.assertRaises(ValueError):
            output.samples_uv[1000, 0] = 0
        with self.assertRaises(ValueError):
            decoded.channels[0].samples_uv[0] = 0

    def test_no_fitted_state_can_transfer_between_recordings(self):
        first = self.process()
        bad = replace(self.decoded, channels=tuple(replace(c, samples_uv=np.zeros_like(c.samples_uv))
                                                   for c in self.decoded.channels))
        self.assertEqual(self.process(bad).report.status, "unusable")
        second = self.process()
        self.assertEqual(first.report, second.report)
        np.testing.assert_array_equal(first.samples_uv, second.samples_uv)

    def test_annotation_coverage_cannot_be_upgraded_and_events_remain_unique(self):
        partial = EEGRecording.model_validate({**self.recording.model_dump(), "annotation_coverage": "unknown"})
        with self.assertRaisesRegex(ValueError, "cannot_be_upgraded"):
            align_reference_windows(partial, ((0, 5),), annotation_coverage="complete")
        windows = align_reference_windows(self.recording, ((9, 12), (11, 14), (19, 21)))
        self.assertEqual(len({event for w in windows for event in w.event_ids}), 1)
        invalid = self.recording.model_copy(update={"annotations": (
            self.recording.annotations[0].model_copy(update={"end_seconds": 61.0}),)})
        with self.assertRaises(ValueError):
            align_reference_windows(invalid, ((0, 5),))


class QualityCoverageTests(unittest.TestCase):
    def channel(self, seconds=10):
        times = np.arange(seconds * 100) / 100
        return QualityInput("C3", 0, np.sin(2 * np.pi * 10 * times), times, sampling_rate_hz=100)

    def test_incomplete_tail_is_excluded_and_overlaps_do_not_double_count_time(self):
        item = self.channel(12)
        report = assess_signal_quality((item,), expected_channels=("C3",), start_seconds=0, end_seconds=12,
                                       thresholds=QualityThresholds(window_seconds=5, step_seconds=2))
        self.assertEqual(report.excluded_intervals, (TimeInterval(start_seconds=8, end_seconds=12),))
        self.assertAlmostEqual(report.usable_fraction, 8 / 12)
        self.assertEqual(report.status, "limited")

    def test_bad_rate_timestamps_and_digital_shape_cannot_produce_good_windows(self):
        item = self.channel()
        for changed in (replace(item, sampling_rate_hz=0), replace(item, timestamps_seconds=-item.timestamps_seconds),
                        replace(item, digital_samples=np.zeros(3), digital_min=-100, digital_max=100)):
            with self.subTest(channel=changed.source_index):
                report = assess_signal_quality((changed,), expected_channels=("C3",), start_seconds=0, end_seconds=10)
                self.assertEqual(report.status, "unusable")
                self.assertTrue(all(w.status == "excluded" for w in report.windows))

    def test_sampling_mismatch_and_unexpected_channels_exclude_full_interval(self):
        item = self.channel()
        report = assess_signal_quality((item,), expected_channels=("C3",), start_seconds=0, end_seconds=10,
                                       expected_sampling_rate_hz=128)
        self.assertEqual(report.usable_fraction, 0)
        report = assess_signal_quality((item, replace(item, label="ECG")), expected_channels=("C3",),
                                       start_seconds=0, end_seconds=10)
        self.assertEqual(report.usable_fraction, 0)

    def test_empty_expected_channels_and_unbounded_window_counts_rejected(self):
        with self.assertRaises(ValueError):
            assess_signal_quality((), expected_channels=(), start_seconds=0, end_seconds=10)
        with self.assertRaisesRegex(ValueError, "quality_window_limit"):
            assess_signal_quality((self.channel(),), expected_channels=("C3",), start_seconds=0, end_seconds=10,
                                  thresholds=QualityThresholds(step_seconds=1e-12))


class ReaderSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "signals").mkdir()
        self.path = self.root / "signals/test.edf"
        self.content = fixtures._edf(duration=2)
        self.path.write_bytes(self.content)

    def read(self, path="signals/test.edf", **kwargs):
        return read_edf(self.root, path, channel_indices=(0, 1), start_seconds=0, end_seconds=1, **kwargs)

    def test_source_replacement_between_inspection_and_decode_rejected(self):
        def inspect_and_change(path):
            metadata = inspect_edf(path)
            altered = bytearray(path.read_bytes())
            altered[-2:] = b"xx"
            path.write_bytes(altered)
            return metadata
        with patch("seizure_research.waveform.inspect_edf", side_effect=inspect_and_change):
            with self.assertRaisesRegex(ValueError, "changed_before_decode"):
                self.read()

    def test_symlinks_and_non_normalized_paths_rejected(self):
        (self.root / "alias").symlink_to(self.root / "signals", target_is_directory=True)
        (self.root / "signals/link.edf").symlink_to(self.path)
        for path in ("alias/test.edf", "signals/link.edf", "signals//test.edf", "signals/./test.edf", "signals/test.edf\x00"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.read(path)

    def test_malformed_truncated_and_unsupported_edf_never_returns_samples(self):
        for content in (self.content[:-2], self.content + b"xx", self.content[:20],
                        self.content[:192] + fixtures.field("EDF+C", 44) + self.content[236:],
                        b"\\xffBIOSEMI" + self.content[8:],
                        self.content[:236] + fixtures.field(-1, 8) + self.content[244:]):
            with self.subTest(size=len(content)):
                self.path.write_bytes(content)
                with self.assertRaises(ValueError) as error:
                    self.read()
                self.assertNotIn("TEST_ONLY_HEADER_ID", str(error.exception))

    def test_sample_limit_is_checked_before_decoding_allocation(self):
        with patch("seizure_research.waveform.np.empty", side_effect=AssertionError("allocated")):
            with self.assertRaisesRegex(ValueError, "sample_limit_exceeded"):
                self.read(max_samples=2)


class SplitSafetyTests(unittest.TestCase):
    def test_known_chb_aliases_cannot_be_split_as_different_patients(self):
        records = contracts.cohort(3)
        for index, subject in enumerate(("chb01", "chb21")):
            records[index] = contracts.modified(records[index], subject_id=subject,
                provenance={**records[index].provenance.model_dump(), "dataset_id": "chb_mit"})
        with self.assertRaisesRegex(ValueError, "Known CHB-MIT patient aliases"):
            build_split_manifest(records)

    def test_serialized_window_manifest_rejects_patient_leakage(self):
        data = {"split_manifest_sha256": "a" * 64, "assignments": [
            {"window_id": "one", "recording_id": "r", "patient_group_id": "p",
             "start_seconds": 0.0, "end_seconds": 5.0, "split": "train"},
            {"window_id": "two", "recording_id": "r", "patient_group_id": "p",
             "start_seconds": 4.0, "end_seconds": 9.0, "split": "test"}]}
        with self.assertRaisesRegex(ValueError, "Patient window leakage"):
            WindowSplitManifest.model_validate(data)

    def test_mixed_source_rates_do_not_match_single_rate_model_evidence(self):
        record = contracts.modified(contracts.recording(), channel_sampling_rates_hz=(256.0, 128.0))
        result = assess_eligibility(record, contracts.model_spec(record))
        self.assertFalse(result.offline_research_eligible)
        self.assertIn("unsupported_per_channel_sampling_rates", result.reasons)

    def test_processing_rejects_edited_patient_disjoint_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = ingest_cohort(root, fixtures.plan_fixture(root)).manifest
            records = tuple(entry.recording for entry in manifest.records)
            split = build_split_manifest(records)
            assignments = list(split.assignments)
            left, right = assignments[0].split, assignments[1].split
            assignments[0] = assignments[0].model_copy(update={"split": right})
            assignments[1] = assignments[1].model_copy(update={"split": left})
            changed = split.model_copy(update={"assignments": tuple(assignments)})
            record = records[0]
            config = make_age_config(record.age.group, tuple(
                ChannelMapping(source_index=i, source_label=label, target_label=label, expected_source_rate_hz=256)
                for i, label in enumerate(record.montage.channels)), target_sampling_rate_hz=256,
                research_review_ref="TEST_ONLY")
            output = process_approved_recording(root, manifest, record.recording_id, config,
                                                start_seconds=0, end_seconds=60, split_manifest=split)
            self.assertEqual(output.report.split_manifest_sha256, split.sha256)
            with self.assertRaisesRegex(ValueError, "deterministic assignments"):
                process_approved_recording(root, manifest, record.recording_id, config,
                                           start_seconds=0, end_seconds=60, split_manifest=changed)


if __name__ == "__main__":
    unittest.main()
