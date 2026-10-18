"""Generated EDF integration for the archive cohort; no clinical data/approvals."""
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from seizure_research.adapters.chb_mit import parse_summary
from seizure_research.annotation_review import build_review_packet, seal_review_packet, parse_seizure_sidecar
from seizure_research.archive import ArchiveApproval, run_archive, read_json
from seizure_research.archive_selection import select_cohort
from seizure_research.cohort import (proposed_split, validate_proposed_split, annotation_investigation,
                                     audit_proposed_split)
from seizure_research.ingestion import normalized_plan
from seizure_research.inventory import build_inventory
from seizure_research.multipatient import (DEFAULT_RUN, run_multipatient, validate_review_inputs,
                                           load_multipatient_artifacts, read_archive_waveform)
from seizure_research.splitting import digest
from seizure_synthetic_fixture import SyntheticPilot, synthetic_environment, write_json, sha

PROJECT = Path(__file__).resolve().parents[1]


def annotation_bytes(start=30, end=40):
    def skip(samples):
        return (59 << 10).to_bytes(2, "little") + (samples >> 16).to_bytes(2, "little") + (samples & 65535).to_bytes(2, "little")
    return skip(start * 256) + (32 << 10).to_bytes(2, "little") + skip((end-start)*256) + (33 << 10).to_bytes(2, "little") + b"\0\0"


class SyntheticArchive:
    def __init__(self, base, *, sidecar_start=30):
        self.base = base
        pilot = SyntheticPilot(base)
        self.plan = copy.deepcopy(pilot.plan)
        for identity in self.plan["identities"]["entries"]:
            identity["canonical_patient_id"] = identity["subject_id"]
        self.payloads = dict(pilot.payloads)
        for path in list(self.payloads):
            if path.endswith(".seizures"):
                self.payloads[path] = annotation_bytes(start=sidecar_start)
        cases = sorted({item["subject_id"] for item in self.plan["recordings"]})
        edfs = sorted(p for p in self.payloads if p.endswith(".edf"))
        self.payloads.update({
            "ANNOTATORS": b"TEST_ONLY synthetic WFDB seizure markers\n",
            "RECORDS": ("\n".join(edfs)+"\n").encode(),
            "RECORDS-WITH-SEIZURES": ("\n".join(p for p in edfs if p+".seizures" in self.payloads)+"\n").encode(),
            "SUBJECT-INFO": ("Case Gender Age\n"+"".join(f"{case} F 10\n" for case in cases)).encode(),
        })
        self.payloads["SHA256SUMS.txt"] = "".join(f"{sha(data)}  {name}\n" for name,data in sorted(self.payloads.items()) if name!="SHA256SUMS.txt").encode()
        metadata = {p: data for p, data in self.payloads.items() if not p.endswith((".edf", ".seizures"))}
        metadata["root.html"] = ("".join(f'<a href="{case}/">{case}/</a> -\n' for case in cases)
                                 + "".join(f'<a href="{p}">{p}</a> {len(data)}\n' for p,data in self.payloads.items() if "/" not in p)).encode()
        for case in cases:
            metadata[case+"/listing.html"] = "".join(f'<a href="{Path(p).name}">{Path(p).name}</a> {len(data)}\n'
                                                       for p,data in self.payloads.items() if p.startswith(case+"/")).encode()
        self.inventory = build_inventory(metadata)
        self.selection = select_cohort(self.inventory)
        self.proposal = proposed_split(self.inventory, self.selection)
        self.packet = build_review_packet(self.inventory, self.selection)
        for row in self.packet["recordings"]:
            positive = bool(row["summary_declared_event_count"])
            row.update(recording_duration_seconds=90, review_status="reviewed", reviewer="TEST_ONLY",
                       reviewed_on="2026-10-13", evidence_ref="TEST_ONLY_review",
                       sidecar_parsed_and_reconciled=positive, reconciliation_notes="TEST_ONLY generated markers",
                       reviewed_event_intervals_seconds=[{"start_seconds":30,"end_seconds":40}] if positive else [],
                       complete_record_coverage_attested=True, verified_seizure_free=not positive,
                       quality_and_excluded_intervals_reviewed=True, eligibility_status="eligible",
                       excluded_intervals_seconds=[],
                       reviewed_seizure_free_intervals_seconds=[[0,30],[40,90]] if positive else [[0,90]])
        self.packet = seal_review_packet(self.packet)
        # Reuse the existing reviewed 18-channel/256-Hz fixture profile.
        for path, payload in self.payloads.items():
            if path.endswith((".edf", "summary.txt", ".seizures")):
                dest = pilot.root / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(payload)
        pilot.review()
        self.configs = {item["edf_path"]: pilot.config.model_dump(mode="json") for item in self.plan["recordings"]}
        self.run_config = DEFAULT_RUN.model_copy(update={"n_estimators":8,"early_stopping_rounds":3,"n_jobs":1})
        self.root,self.temporary,self.runs = (base / n for n in ("archive","temporary","runs"))
        for p in (self.root,self.temporary,self.runs):p.mkdir()
        self.approval_path = base / "TEST_ONLY_archive_approval.json"
        approval = ArchiveApproval().model_dump(mode="json")
        for key in approval:
            if key.endswith("_status"):approval[key]="approved"
            elif key.endswith(("_reviewed_on","_approved_on")):approval[key]="2026-10-13"
            elif key.endswith(("_reviewer","_approver","_ref")):approval[key]="TEST_ONLY"
        approval.update(institutional_determination="not_required",staging_root=str(self.root),
                        temporary_root=str(self.temporary),storage_persistence_confirmed=True,
                        temporary_cleanup_permitted=True,inventory_sha256=self.inventory["inventory_sha256"],
                        selection_sha256=self.selection["selection_sha256"],staging_manifest_sha256=self.selection["selection_sha256"],
                        training_runs_root=str(self.runs),
                        training_plan_sha256=digest(normalized_plan(self.plan).model_dump(mode="json")),
                        training_split_sha256=self.proposal["split_proposal_sha256"],
                        training_review_packet_sha256=self.packet["packet_sha256"],
                        training_preprocessing_sha256=digest(self.configs),
                        training_run_config_sha256=digest(self.run_config.model_dump(mode="json")))
        self.approval=approval
        write_json(self.approval_path,approval)
        self.ranges=[]

    def transport(self, request, timeout):
        prefix="https://physionet.org/files/chbmit/1.0.0/"
        relative=request.full_url.removeprefix(prefix)
        if relative not in self.payloads:raise AssertionError("Network request outside generated fixture")
        payload=self.payloads[relative]
        range_header=request.get_header("Range")
        start=int(range_header.split('=')[1].split('-')[0]) if range_header else 0
        if start:self.ranges.append(relative)
        result=io.BytesIO(payload[start:])
        result.status=206 if start else 200
        result.headers={"Content-Length":str(len(payload)-start)}
        if start:result.headers["Content-Range"]=f"bytes {start}-{len(payload)-1}/{len(payload)}"
        result.geturl=lambda:request.full_url
        return result

    def stage(self):
        with patch("seizure_research.archive.urllib.request.urlopen",side_effect=self.transport):
            return run_archive(self.root,self.temporary,self.inventory,self.selection,self.approval_path,confirmed=True)

    def run(self):
        return run_multipatient(self.root,self.temporary,self.inventory,self.selection,
                               approval_path=self.approval_path,proposal=self.proposal,review_plan=self.plan,
                               review_packet=self.packet,preprocessing=self.configs,output_dir=self.runs/"TEST_ONLY_run",
                               confirmed=True,run_config=self.run_config)


class MultipatientTests(unittest.TestCase):
    def test_official_proposal_is_deterministic_and_alias_safe(self):
        i=read_json(PROJECT/"reports/checkpoint_d7/official_inventory.json")
        s=read_json(PROJECT/"reports/checkpoint_d7/selection_proposal.json")
        p=proposed_split(i,s)
        self.assertEqual(p,proposed_split(i,s))
        self.assertEqual([len(p["distribution"][name]["patients"]) for name in ("train","validation","test")],[13,4,6])
        aliases=[r for r in p["assignments"] if r["case_id"] in ("chb01","chb21")]
        self.assertEqual({r["canonical_patient_id"] for r in aliases},{"chb01"})
        self.assertEqual(len({r["split"] for r in aliases}),1)
        changed=copy.deepcopy(p);changed["assignments"][0]["split"]="test"
        with self.assertRaisesRegex(ValueError,"split_changed"):validate_proposed_split(i,s,changed)
        report=annotation_investigation(i,s)
        self.assertEqual(len(report["issues"]),12)
        self.assertEqual(len(report["background_candidates"]),22)
        self.assertFalse(any(r["resolved_by_public_metadata"] for r in report["issues"]))
        self.assertTrue(all(r["human_review_required"] and r["human_review_evidence"] is None
                            and r["machine_interpretation"] for r in report["issues"]))
        conflict = next(r for r in report["issues"] if r["path"] == "chb07/chb07_19.edf")
        self.assertTrue(conflict["sidecar_url"].endswith("chb07_19.edf.seizures"))
        audit = audit_proposed_split(i, s, p)
        self.assertEqual(audit, audit_proposed_split(i, s, json.loads(json.dumps(p, sort_keys=True))))
        self.assertTrue(audit["candidate_classes_in_each_split"])
        self.assertEqual(audit["unknown_age_patients"], ["chb24"])
        self.assertEqual(audit["montage"]["summary_compatible_recordings"], 45)
        self.assertFalse(audit["scientific_adequacy_established"])
        self.assertEqual(audit["age_groups_missing_from_splits"],
                         {"adolescent": ["validation", "test"], "adult": ["validation"]})
        self.assertEqual([(p["distribution"][n]["positive_candidates"],
                           p["distribution"][n]["background_candidates"])
                          for n in ("train", "validation", "test")], [(13, 12), (4, 4), (6, 6)])

    def test_unverified_background_and_incompatible_montage_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="neurosense-GATES-TEST_ONLY-") as d, synthetic_environment():
            fixture = SyntheticArchive(Path(d))
            packet = copy.deepcopy(fixture.packet)
            next(r for r in packet["recordings"] if not r["summary_declared_event_count"])["verified_seizure_free"] = False
            with self.assertRaisesRegex(PermissionError, "annotation_review"):
                validate_review_inputs(fixture.inventory, fixture.selection, fixture.proposal,
                                       fixture.plan, seal_review_packet(packet), fixture.configs)
            configs = copy.deepcopy(fixture.configs)
            configs[next(iter(configs))]["channels"][0]["target_label"] = "UNREVIEWED"
            with self.assertRaisesRegex(ValueError, "montage_mismatch"):
                validate_review_inputs(fixture.inventory, fixture.selection, fixture.proposal,
                                       fixture.plan, fixture.packet, configs)

    def test_missing_published_age_is_not_imputed_from_review_plan(self):
        with tempfile.TemporaryDirectory(prefix="neurosense-AGE-TEST_ONLY-") as d, synthetic_environment():
            fixture = SyntheticArchive(Path(d))
            inventory = copy.deepcopy(fixture.inventory)
            inventory["recordings"][0]["age"] = None
            inventory["inventory_sha256"] = digest({k: v for k, v in inventory.items() if k != "inventory_sha256"})
            selection = select_cohort(inventory)
            proposal = proposed_split(inventory, selection)
            packet = {**fixture.packet, "inventory_sha256": inventory["inventory_sha256"],
                      "selection_sha256": selection["selection_sha256"]}
            with self.assertRaisesRegex(ValueError, "unknown_published_age"):
                validate_review_inputs(inventory, selection, proposal, fixture.plan,
                                       seal_review_packet(packet), fixture.configs)

    def test_checksum_valid_but_conflicting_sidecar_blocks_training(self):
        with tempfile.TemporaryDirectory(prefix="neurosense-SIDECAR-TEST_ONLY-") as d, synthetic_environment():
            fixture = SyntheticArchive(Path(d), sidecar_start=31)
            self.assertEqual(fixture.stage()["status"], "complete_verified")
            with patch("seizure_research.multipatient.train_xgboost",
                       side_effect=AssertionError("conflicting annotations reached training")):
                with self.assertRaisesRegex(ValueError, "sidecar_summary_event_disagreement"):
                    fixture.run()

    def test_summary_target_keeps_strict_annotation_checks(self):
        cache=PROJECT/"reports/checkpoint_d7/source_metadata"
        if not cache.exists():self.skipTest("public metadata cache not available")
        text=(cache/"chb09/chb09-summary.txt").read_text()
        with self.assertRaises(ValueError):parse_summary(text,"chb09")
        selected=parse_summary(text,"chb09",filename="chb09_06.edf")
        self.assertEqual(selected[0].declared_seizure_count,1)
        with self.assertRaises(ValueError):parse_summary(text,"chb09",filename="chb09_08.edf")
        for case in ("chb12","chb20"):
            self.assertTrue(parse_summary((cache/case/(case+"-summary.txt")).read_text(),case))

    def test_sidecar_parser_checks_markers_bounds_and_truncation(self):
        self.assertEqual(parse_seizure_sidecar(annotation_bytes(),sampling_rate_hz=256,duration_seconds=90),
                         [{"start_seconds":30.0,"end_seconds":40.0}])
        for bad in (annotation_bytes()[:-2], annotation_bytes()+b"\1\0", b"\0\0", annotation_bytes(end=100)):
            with self.assertRaises(ValueError):parse_seizure_sidecar(bad,sampling_rate_hz=256,duration_seconds=90)

    def test_generated_archive_to_model_and_heldout_results(self):
        with tempfile.TemporaryDirectory(prefix="neurosense-MULTIPATIENT-TEST_ONLY-") as d, synthetic_environment():
            fixture=SyntheticArchive(Path(d))
            first=fixture.selection["recordings"][0]
            (fixture.temporary/(first["sha256"]+".part")).write_bytes(fixture.payloads[first["path"]][:2048])
            self.assertEqual(fixture.stage()["status"],"complete_verified")
            self.assertIn(first["path"],fixture.ranges)
            with patch("seizure_research.multipatient.reviewed_archive_source",
                       side_effect=AssertionError("must reject before EDF extraction")):
                pending=copy.deepcopy(fixture.packet)
                pending["recordings"][0]["review_status"]="unreviewed"
                with self.assertRaisesRegex(PermissionError,"annotation_review"):
                    validate_review_inputs(fixture.inventory,fixture.selection,fixture.proposal,fixture.plan,
                                           seal_review_packet(pending),fixture.configs)
                broken=copy.deepcopy(fixture.packet)
                broken["recordings"][0]["reviewed_seizure_free_intervals_seconds"]=[]
                with self.assertRaisesRegex(ValueError,"partition"):
                    validate_review_inputs(fixture.inventory,fixture.selection,fixture.proposal,fixture.plan,
                                           seal_review_packet(broken),fixture.configs)
                approval=copy.deepcopy(fixture.approval);approval["training_status"]="pending"
                write_json(fixture.approval_path,approval)
                with self.assertRaisesRegex(PermissionError,"training_authorization"):fixture.run()
                approval=copy.deepcopy(fixture.approval);approval["processing_status"]="pending"
                write_json(fixture.approval_path,approval)
                with self.assertRaises(PermissionError):fixture.run()
                approval=copy.deepcopy(fixture.approval);approval["training_split_sha256"]="0"*64
                write_json(fixture.approval_path,approval)
                with self.assertRaisesRegex(PermissionError,"input_binding"):fixture.run()
                write_json(fixture.approval_path,fixture.approval)
            # Omitting a default must retain the authorized effective profile
            # and produce a bundle that its own loader can validate.
            for config in fixture.configs.values():
                self.assertEqual(config.pop("filter_order"), 4)
            result=fixture.run()
            self.assertEqual(result["status"],"complete")
            run=Path(result["run_dir"])
            complete=read_json(run/"complete.json")
            self.assertTrue(complete["files_sha256"])
            self.assertEqual(result["patients"],6)
            self.assertEqual(result["recordings"],12)
            self.assertFalse(result["clinical_validation"])
            metrics=result["metrics"]
            self.assertEqual(metrics["detected_events"]+metrics["missed_events"],metrics["reference_events"])
            self.assertGreater(metrics["evaluable_hours"],0)
            self.assertAlmostEqual(metrics["false_alarms_per_hour"],metrics["false_events"]/metrics["evaluable_hours"])
            self.assertEqual(list(fixture.temporary.iterdir()),[])
            frozen=read_json(run/"test_freeze.json")
            test_shards=list((run/"features/test").glob("*.parquet"))
            self.assertTrue(all(p.stat().st_mtime_ns >= (run/"test_freeze.json").stat().st_mtime_ns for p in test_shards))
            self.assertEqual(frozen["model_sha256"],hashlib.sha256((run/"model/xgboost.json").read_bytes()).hexdigest())
            review_path=fixture.base/"TEST_ONLY_review.json"
            write_json(review_path,fixture.packet)
            kwargs=dict(root=fixture.root,temporary_root=fixture.temporary,inventory=fixture.inventory,
                        selection=fixture.selection,approval_path=fixture.approval_path,review_path=review_path)
            bundle=load_multipatient_artifacts(run,**kwargs)
            self.assertEqual(bundle["evaluation"]["metrics"],metrics)
            segment=read_archive_waveform(fixture.root,fixture.temporary,fixture.inventory,fixture.selection,
                                         approval_path=fixture.approval_path,proposal=fixture.proposal,
                                         review_plan=fixture.plan,review_packet=fixture.packet,preprocessing=fixture.configs,
                                         relative_path=fixture.selection["recordings"][0]["path"],
                                         channel_labels=("FP1-F7",),start_seconds=25,end_seconds=45)
            self.assertEqual(segment["samples_uv"].shape,(20*256,1))
            self.assertEqual(list(fixture.temporary.iterdir()),[])
            path=run/"test_evaluation.json"
            payload=read_json(path);payload["metrics"]["event_sensitivity"]=0.12345
            write_json(path,payload)
            with self.assertRaisesRegex(ValueError,"checksum"):
                load_multipatient_artifacts(run,**kwargs)


if __name__=="__main__":unittest.main()
