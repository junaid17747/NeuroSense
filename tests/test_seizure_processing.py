"""Synthetic EDF signal and fictitious annotation tests; never real patient data."""

import copy
import hashlib
import math
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError
from scipy.signal import sosfiltfilt

from seizure_research.ingestion import ingest_cohort
from seizure_research.preprocessing import (
    ChannelMapping, PreprocessingConfig, _filter_sos, align_reference_windows, make_age_config,
    preprocess_recording,
)
from seizure_research.quality import QualityInput, QualityThresholds, assess_signal_quality
from seizure_research.schemas import AgeGroup, TimeInterval
from seizure_research.splitting import WindowSplitInput, assign_windows_to_splits, build_split_manifest
from seizure_research.waveform import (
    MAX_DECODED_SAMPLES, channels_with_label, duplicate_channel_labels, read_edf,
)
from test_seizure_ingestion import field, plan_fixture


def _edf(labels=("FP1-F7", "F7-T7"), rates=(128, 128), *, duration=60, units=None,
         digital_ranges=None, physical_ranges=None,
         waveform=True):
    count = len(labels)
    units = tuple("uV" for _ in range(count)) if units is None else units
    digital_ranges = tuple((-32768, 32767) for _ in range(count)) if digital_ranges is None else digital_ranges
    physical_ranges = tuple((-100, 100) for _ in range(count)) if physical_ranges is None else physical_ranges
    header = b"".join((field("0", 8), field("TEST_ONLY_HEADER_ID", 80), field("TEST_ONLY", 80),
                       field("01.01.00", 8), field("00.00.00", 8), field(256 * (count + 1), 8),
                       field("", 44), field(duration, 8), field(1, 8), field(count, 4)))
    signal_fields = (
        (16, labels), (80, ("",) * count), (8, units),
        (8, tuple(item[0] for item in physical_ranges)),
        (8, tuple(item[1] for item in physical_ranges)),
        (8, tuple(item[0] for item in digital_ranges)),
        (8, tuple(item[1] for item in digital_ranges)),
        (80, ("",) * count), (8, rates), (32, ("",) * count),
    )
    variable = b"".join(field(value, width) for width, values in signal_fields for value in values)
    payload = []
    for index in range(duration):
        for channel, (rate, (dmin, dmax), (pmin, pmax)) in enumerate(zip(rates, digital_ranges, physical_ranges)):
            for sample in range(rate):
                if waveform:
                    physical = 10 * math.sin(2 * math.pi * 10 * (index + sample / rate) + channel * 0.4)
                    digital = round((physical - pmin) * (dmax - dmin) / (pmax - pmin) + dmin)
                else:
                    digital = dmax
                payload.append(max(dmin, min(dmax, digital)))
    return header + variable + struct.pack("<" + "h" * len(payload), *payload)


def _fixture(root: Path):
    plan = plan_fixture(root, subjects=("chb90",), years=(10,), events=((10, 20),))
    item = plan["recordings"][0]
    path = root / item["edf_path"]
    content = _edf()
    path.write_bytes(content)
    item["expected_signal_sha256"] = hashlib.sha256(content).hexdigest()
    summary_path = root / item["summary_path"]
    text = summary_path.read_text().replace("256 Hz", "128 Hz")
    summary_path.write_text(text)
    item["expected_summary_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    item["expected_sampling_rate_hz"] = 128.0
    plan["recordings"][0]["channels"] = [
        {"source_index": i, "source_label": label, "action": "keep", "output_label": label}
        for i, label in enumerate(("FP1-F7", "F7-T7"))
    ]
    plan["recordings"][0]["montage"] = {
        "name": "TEST_ONLY", "kind": "bipolar", "channels": ["FP1-F7", "F7-T7"],
        "reference": "pairwise", "units": "uV",
    }
    return plan


class EDFWaveformTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "subject").mkdir()

    def write(self, content):
        path = self.root / "subject" / "recording.edf"
        path.write_bytes(content)
        return path

    def test_digital_physical_calibration_and_sample_timestamps(self):
        self.write(_edf(labels=("C3",), rates=(100,), digital_ranges=((-1000, 1000),),
                        physical_ranges=((-200, 200),)))
        read = read_edf(self.root, "subject/recording.edf", channel_indices=(0,),
                        start_seconds=0, end_seconds=1)
        channel = read.channels[0]
        self.assertAlmostEqual(channel.samples_uv[3], 9.51, delta=0.11)
        self.assertEqual(channel.timestamps_seconds[3], 0.03)
        self.assertEqual(channel.metadata.physical_min, -200)
        self.assertEqual(channel.metadata.digital_max, 1000)

    def test_millivolts_are_converted_to_microvolts_explicitly(self):
        self.write(_edf(labels=("C3",), rates=(100,), units=("mV",)))
        channel = read_edf(self.root, "subject/recording.edf", channel_indices=(0,),
                           start_seconds=0, end_seconds=1).channels[0]
        self.assertAlmostEqual(channel.samples_uv[3], 9510, delta=110)

    def test_per_channel_rates_selection_order_and_bounded_read(self):
        self.write(_edf(labels=("C3", "ECG"), rates=(128, 64)))
        result = read_edf(self.root, "subject/recording.edf", channel_indices=(1, 0),
                           start_seconds=5, end_seconds=6, max_samples=200)
        self.assertEqual([item.metadata.label for item in result.channels], ["ECG", "C3"])
        self.assertEqual([len(item.samples_uv) for item in result.channels], [64, 128])
        self.assertEqual(result.channels[0].timestamps_seconds[0], 5.0)
        with self.assertRaisesRegex(ValueError, "edf_decoded_sample_limit_exceeded"):
            read_edf(self.root, "subject/recording.edf", channel_indices=(0, 1),
                     start_seconds=0, end_seconds=60, max_samples=200)
        with self.assertRaisesRegex(ValueError, "edf_sample_limit_invalid"):
            read_edf(self.root, "subject/recording.edf", channel_indices=(0,),
                     start_seconds=0, end_seconds=1, max_samples=MAX_DECODED_SAMPLES + 1)

    def test_duplicate_labels_require_index_selection_and_are_reported(self):
        result = self.write(_edf(labels=("C3", "C3")))
        decoded = read_edf(self.root, "subject/recording.edf", channel_indices=(1,),
                           start_seconds=0, end_seconds=1)
        self.assertEqual(duplicate_channel_labels(decoded.metadata), ("C3",))
        self.assertEqual(len(channels_with_label(decoded.metadata, "C3")), 2)
        self.assertEqual(decoded.channels[0].metadata.index, 1)

    def test_edf_data_record_size_limit_rejected(self):
        self.write(_edf(labels=("C3",), rates=(32_768,), duration=1))
        with self.assertRaisesRegex(ValueError, "edf_data_record_size_exceeded"):
            read_edf(self.root, "subject/recording.edf", channel_indices=(0,),
                     start_seconds=0, end_seconds=1)

    def test_unsupported_units_path_traversal_and_bounds_rejected(self):
        self.write(_edf(labels=("ECG",), rates=(128,), units=("mmHg",)))
        with self.assertRaisesRegex(ValueError, "edf_signal_unit_unsupported"):
            read_edf(self.root, "subject/recording.edf", channel_indices=(0,), start_seconds=0, end_seconds=1)
        self.write(_edf())
        for path in ("../secret.edf", "/etc/passwd", "subject/../recording.edf"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                read_edf(self.root, path, channel_indices=(0,), start_seconds=0, end_seconds=1)
        with self.assertRaisesRegex(ValueError, "edf_time_range_out_of_bounds"):
            read_edf(self.root, "subject/recording.edf", channel_indices=(0,), start_seconds=59, end_seconds=61)

    def test_non_eeg_channels_must_be_explicitly_excluded_by_index(self):
        self.write(_edf(labels=("C3", "ECG")))
        result = read_edf(self.root, "subject/recording.edf", channel_indices=(0,), start_seconds=0, end_seconds=1)
        self.assertEqual(len(result.channels), 1)
        self.assertEqual(result.channels[0].metadata.label, "C3")


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = _fixture(self.root)
        result = ingest_cohort(self.root, self.plan)
        self.assertIsNotNone(result.manifest)
        self.manifest = result.manifest
        self.entry = result.manifest.records[0]
        self.recording = self.entry.recording
        self.decoded = read_edf(self.root, self.entry.signal.relative_path,
                                channel_indices=(0, 1), start_seconds=0, end_seconds=60)
        mapping = tuple(ChannelMapping(source_index=i, source_label=label, target_label=label,
                                        expected_source_rate_hz=128.0)
                         for i, label in enumerate(("FP1-F7", "F7-T7")))
        self.config = make_age_config(AgeGroup.CHILD, mapping, target_sampling_rate_hz=128,
                                      research_review_ref="TEST_ONLY_profile_review")

    def test_population_profiles_are_explicit_and_cutoffs_differ(self):
        cutoffs = {age: make_age_config(age, self.config.channels, target_sampling_rate_hz=128,
                                        research_review_ref="TEST_ONLY").lowpass_hz for age in AgeGroup}
        self.assertEqual(set(cutoffs), set(AgeGroup))
        self.assertGreater(len(set(cutoffs.values())), 1)
        with self.assertRaises(ValidationError):
            make_age_config(AgeGroup.NEONATAL, self.config.channels, target_sampling_rate_hz=128,
                            research_review_ref=" ")

    def test_preprocessing_reproducible_shape_quality_and_annotation_alignment(self):
        first = preprocess_recording(self.recording, self.decoded, self.config, approved_manifest=self.manifest)
        second = preprocess_recording(self.recording, self.decoded, self.config, approved_manifest=self.manifest)
        self.assertEqual(first.samples_uv.shape[1], 2)
        self.assertEqual(first.channel_labels, ("FP1-F7", "F7-T7"))
        self.assertEqual(first.report.processed_sha256, second.report.processed_sha256)
        self.assertEqual(first.report.config_sha256, second.report.config_sha256)
        self.assertEqual(first.report.quality.status, "acceptable")
        self.assertTrue(np.isnan(first.samples_uv[:20]).any())  # zero-phase edge padding is excluded
        labels = {window.label for window in first.report.windows}
        self.assertIn("positive", labels)
        self.assertIn("negative", labels)
        self.assertEqual(first.report.annotation_provenance_ref, "TEST_ONLY_expert_annotation_provenance")
        self.assertTrue(first.report.coverage_complete)
        expected = np.sin(2 * np.pi * 10 * first.timestamps_seconds)
        valid = np.isfinite(first.samples_uv[:, 0])
        self.assertGreater(abs(np.corrcoef(first.samples_uv[valid, 0], expected[valid])[0, 1]), 0.98)

    def test_average_reference_and_zero_phase_filter_are_explicit(self):
        config = make_age_config(AgeGroup.CHILD, self.config.channels,
                                 target_sampling_rate_hz=128,
                                 research_review_ref="TEST_ONLY_reference_review",
                                 reference_mode="average")
        with self.assertRaisesRegex(ValueError, "reviewed_referential_montage"):
            preprocess_recording(self.recording, self.decoded, config, approved_manifest=self.manifest)
        plan = copy.deepcopy(self.plan)
        plan["recordings"][0]["montage"].update(kind="referential", reference="TEST_ONLY_common_reference")
        manifest = ingest_cohort(self.root, plan).manifest
        processed = preprocess_recording(manifest.records[0].recording, self.decoded, config,
                                         approved_manifest=manifest)
        finite_rows = np.isfinite(processed.samples_uv).all(axis=1)
        self.assertTrue(np.allclose(processed.samples_uv[finite_rows].mean(axis=1), 0, atol=1e-8))
        sos = _filter_sos(config)
        t = np.arange(128 * 20) / 128
        source = np.sin(2 * np.pi * 10 * t)
        filtered = sosfiltfilt(sos, source, padlen=3 * (2 * len(sos) + 1))
        interior = slice(500, -500)
        self.assertGreater(np.corrcoef(source[interior], filtered[interior])[0, 1], 0.99)

    def test_cross_population_channel_rate_and_config_mismatches_rejected(self):
        neonatal = make_age_config(AgeGroup.NEONATAL, self.config.channels,
                                   target_sampling_rate_hz=128,
                                   research_review_ref="TEST_ONLY_neonatal_review")
        with self.assertRaisesRegex(ValueError, "population_configuration_mismatch"):
            preprocess_recording(self.recording, self.decoded, neonatal, approved_manifest=self.manifest)
        wrong_channels = make_age_config(AgeGroup.CHILD,
                                         (ChannelMapping(source_index=0, source_label="FP1-F7",
                                                         target_label="F7-T7", expected_source_rate_hz=128),),
                                         target_sampling_rate_hz=128,
                                         research_review_ref="TEST_ONLY")
        with self.assertRaisesRegex(ValueError, "reviewed_montage_mapping_mismatch"):
            preprocess_recording(self.recording, self.decoded, wrong_channels, approved_manifest=self.manifest)
        with self.assertRaises(ValidationError):
            PreprocessingConfig(config_id="bad", age_group=AgeGroup.CHILD,
                                research_review_ref="TEST_ONLY", channels=self.config.channels,
                                target_sampling_rate_hz=80, highpass_hz=1, lowpass_hz=40,
                                line_noise_hz=50)

    def test_explicit_mixed_rate_resampling_and_source_hash_binding(self):
        path = self.root / self.entry.signal.relative_path
        content = _edf(rates=(128, 64))
        path.write_bytes(content)
        mixed = read_edf(self.root, self.entry.signal.relative_path, channel_indices=(0, 1),
                         start_seconds=0, end_seconds=60)
        with self.assertRaisesRegex(ValueError, "decoded_signal_hash"):
            preprocess_recording(self.recording, mixed, self.config, approved_manifest=self.manifest)
        mixed_plan = copy.deepcopy(self.plan)
        mixed_plan["recordings"][0]["expected_signal_sha256"] = mixed.metadata.sha256
        mixed_plan["recordings"][0]["channels"][1]["expected_sampling_rate_hz"] = 64.0
        reviewed = ingest_cohort(self.root, mixed_plan)
        self.assertIsNotNone(reviewed.manifest)
        updated_recording = reviewed.manifest.records[0].recording
        mappings = (
            ChannelMapping(source_index=0, source_label="FP1-F7", target_label="FP1-F7",
                           expected_source_rate_hz=128),
            ChannelMapping(source_index=1, source_label="F7-T7", target_label="F7-T7",
                           expected_source_rate_hz=64),
        )
        config = make_age_config(AgeGroup.CHILD, mappings, target_sampling_rate_hz=128,
                                 research_review_ref="TEST_ONLY_mixed_rate_review", allow_resampling=True)
        processed = preprocess_recording(updated_recording, mixed, config,
                                         approved_manifest=reviewed.manifest)
        self.assertEqual(processed.samples_uv.shape[1], 2)
        self.assertEqual(processed.samples_uv.shape[0], 60 * 128)

    def test_filtering_does_not_cross_quality_exclusions_and_short_segments_rejected(self):
        thresholds = QualityThresholds(window_seconds=5, step_seconds=5, flatline_std_uv=100)
        report = preprocess_recording(self.recording, self.decoded, self.config,
                                      approved_manifest=self.manifest, thresholds=thresholds).report
        self.assertIn("excluded", {item.label for item in report.windows})
        with self.assertRaisesRegex(ValueError, "filter_segment_too_short"):
            short = read_edf(self.root, self.entry.signal.relative_path,
                             channel_indices=(0, 1), start_seconds=0, end_seconds=0.1)
            preprocess_recording(self.recording, short, self.config, approved_manifest=self.manifest)


class SignalQualityTests(unittest.TestCase):
    def test_flatline_saturation_missing_duplicates_and_nonfinite_are_explicit(self):
        times = np.arange(500) / 100
        channels = (
            QualityInput("C3", 0, np.zeros(500), times, np.full(500, 32767), -32768, 32767, 100),
            QualityInput("C3", 1, np.concatenate(([np.inf], np.full(499, np.nan))), times,
                         sampling_rate_hz=100),
        )
        report = assess_signal_quality(channels, expected_channels=("C3", "C4"),
                                       start_seconds=0, end_seconds=5,
                                       expected_sampling_rate_hz=100)
        self.assertEqual(report.status, "unusable")
        self.assertEqual(report.duplicate_labels, ("C3",))
        self.assertEqual(report.missing_channels, ("C4",))
        self.assertIn("unusable", {item.status for item in report.channel_reports})
        self.assertTrue(report.excluded_intervals)

    def test_sampling_gaps_and_extreme_amplitude_exclude_windows(self):
        times = np.arange(500) / 100
        times[250:] += 0.2
        values = np.full(500, 10.0)
        values[100] = 2000
        report = assess_signal_quality((QualityInput("C3", 0, values, times, sampling_rate_hz=100),),
                                       expected_channels=("C3",), start_seconds=0,
                                       end_seconds=5.2, expected_sampling_rate_hz=100)
        self.assertTrue(report.excluded_intervals)
        self.assertTrue(any("excessive_discontinuity" in item.reasons[0] or "extreme_amplitude" in ",".join(item.reasons)
                            for item in report.windows if item.reasons))

    def test_digital_corruption_and_saturation_have_separate_reasons(self):
        times = np.arange(500) / 100
        digital = np.zeros(500, dtype=np.int64)
        digital[50] = 50_000
        digital[100] = 32767
        values = 10 * np.sin(2 * np.pi * 8 * times)
        report = assess_signal_quality((QualityInput("C3", 0, values, times, digital,
                                                     -32768, 32767, 100),),
                                       expected_channels=("C3",), start_seconds=0,
                                       end_seconds=5, expected_sampling_rate_hz=100)
        result = report.channel_reports[0]
        self.assertEqual(result.corrupt_samples, 1)
        self.assertEqual(result.saturated_samples, 1)
        self.assertIn("corrupt_digital_samples", result.reasons)
        self.assertIn("digital_rail_samples", result.reasons)


class AnnotationAlignmentTests(unittest.TestCase):
    def test_half_open_boundaries_unknown_coverage_and_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = _fixture(root)
            recording = ingest_cohort(root, plan).manifest.records[0].recording
            windows = align_reference_windows(recording, ((0, 10), (10, 15), (20, 25), (59, 60)),
                                              excluded_intervals=(TimeInterval(start_seconds=59, end_seconds=60),))
            self.assertEqual([item.label for item in windows], ["negative", "positive", "negative", "excluded"])
            self.assertEqual(windows[1].event_ids, ("chb90_01.edf:seizure:1",))
            unknown = align_reference_windows(recording, ((0, 5), (10, 15)), annotation_coverage="unknown")
            self.assertEqual([item.label for item in unknown], ["ambiguous", "positive"])
            with self.assertRaisesRegex(ValueError, "window_interval_invalid"):
                align_reference_windows(recording, ((59, 61),))

    def test_overlapping_windows_inherit_patient_disjoint_partition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = plan_fixture(root)
            records = ingest_cohort(root, plan).manifest.records
            recordings = tuple(item.recording for item in records)
            split = build_split_manifest(recordings)
            windows = tuple(WindowSplitInput(
                window_id=f"{record.recording_id}:{index}", recording_id=record.recording_id,
                patient_group_id=record.patient_group_id, start_seconds=0 if index == 0 else 4,
                end_seconds=5 if index == 0 else 9,
            ) for record in recordings for index in range(2))
            assigned = assign_windows_to_splits(windows, split)
            for recording in recordings:
                self.assertEqual(len({item.split for item in assigned.assignments
                                      if item.recording_id == recording.recording_id}), 1)
            bad_identity = windows[0].model_copy(update={"patient_group_id": "TEST_ONLY_wrong"})
            with self.assertRaisesRegex(ValueError, "identity conflicts"):
                assign_windows_to_splits((bad_identity,), split)


if __name__ == "__main__":
    unittest.main()
