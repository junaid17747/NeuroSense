"""Discovery placeholders only: no downloads, license grants, or approved cohorts."""

from datetime import date
from typing import Literal

from pydantic import HttpUrl

from .schemas import AgeGroup, ResearchSchema, Text


class DatasetCandidate(ResearchSchema):
    dataset_id: Text
    name: Text
    source_url: HttpUrl | None
    target_age_groups: tuple[AgeGroup, ...]
    published_population_notes: Text
    review_tasks: tuple[Text, ...]
    status: Literal["candidate_unreviewed", "unresolved_gap"]
    source_checked_on: date | None = None
    annotation_formats: tuple[Text, ...] = ()
    published_access_notes: Text = "Requires source and local access review."
    published_terms_notes: Text = "Terms have not been verified."
    terms_source_url: HttpUrl | None = None
    # These are project review results, not claims about the source's public terms.
    verified_license: Text | None = None
    verified_access: Text | None = None
    ethics_approval_ref: Text | None = None
    ready_for_use: Literal[False] = False


DATASET_REGISTRY = (
    DatasetCandidate(
        dataset_id="helsinki_neonatal_candidate",
        name="Helsinki neonatal EEG with seizure annotations",
        source_url="https://zenodo.org/records/4940267",
        target_age_groups=(AgeGroup.NEONATAL,),
        published_population_notes="Term neonatal clinical EEG with three expert annotation sets.",
        source_checked_on=date(2026, 10, 11),
        annotation_formats=("Expert-specific CSV annotations", "MATLAB annotation file"),
        published_access_notes="Zenodo lists open files; local authorization still requires review.",
        published_terms_notes=(
            "License identifier was not visible in the retrieved record. Verify release metadata and "
            "terms before use. The supplied ethics document does not authorize this project."
        ),
        terms_source_url="https://zenodo.org/records/4940267",
        review_tasks=(
            "Verify chronological age per recording; term neonates do not establish preterm validity.",
            "Pin the release and corrected annotations; preserve inter-rater disagreement.",
            "Review gestational/postmenstrual age, montage, device, and neonatal specialist protocol.",
            "Verify source terms, local access, ethics determination, and de-identification before import.",
        ),
        status="candidate_unreviewed",
    ),
    DatasetCandidate(
        dataset_id="chb_mit_candidate",
        name="CHB-MIT Scalp EEG Database",
        source_url="https://physionet.org/content/chbmit/1.0.0/",
        target_age_groups=(AgeGroup.CHILD, AgeGroup.ADOLESCENT, AgeGroup.ADULT),
        published_population_notes=(
            "Predominantly pediatric clinical EEG; published original age range extends to 22 years. "
            "No neonatal or under-one-year cohort is established by this entry."
        ),
        source_checked_on=date(2026, 10, 11),
        annotation_formats=("Per-case summary text with recording-relative seizure seconds",
                            "Separate .seizure onset/offset annotation files (not parsed by the Phase 2 adapter)"),
        published_access_notes="PhysioNet lists open access subject to the specified license.",
        published_terms_notes=(
            "PhysioNet specifies Open Data Commons Attribution License v1.0. "
            "Review attribution, notices, applicable rights, and project usage before approval."
        ),
        terms_source_url="https://physionet.org/content/chbmit/view-license/1.0.0/",
        review_tasks=(
            "Resolve chb01/chb21 as the same patient; audit every alias before splitting.",
            "Verify recording-specific ages; chb24 is absent from the published SUBJECT-INFO file.",
            "Inspect per-record channel layouts, gaps, device metadata, and event annotations.",
            "Verify release, license obligations, local access, and ethics determination before import.",
        ),
        status="candidate_unreviewed",
    ),
    DatasetCandidate(
        dataset_id="tusz_candidate",
        name="Temple University Hospital EEG Seizure Corpus (TUSZ)",
        source_url="https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
        target_age_groups=(AgeGroup.ADULT, AgeGroup.OLDER_ADULT),
        published_population_notes=(
            "Clinical EEG with manual seizure event annotations. Age targets are research review "
            "priorities, not a claim of verified coverage or adequate subgroup sample sizes."
        ),
        source_checked_on=date(2026, 10, 11),
        annotation_formats=(
            "Manual event start/stop, channel, and seizure type; release-specific serialization requires review",
        ),
        published_access_notes="Custodian registration and accepted access request are required; local access unverified.",
        published_terms_notes="Review the current custodian agreement and release terms; no project license grant is asserted.",
        terms_source_url="https://isip.piconepress.com/projects/nedc/html/tuh_eeg/",
        review_tasks=(
            "Verify the requested release, patient ages, annotation guidelines, devices, and montages.",
            "Review the custodian's access process and current terms; local access is unverified.",
            "Audit identity overlap across releases, official partitions, and other cohorts.",
            "Determine local ethics requirements and age-specific suitability before import.",
        ),
        status="candidate_unreviewed",
    ),
    DatasetCandidate(
        dataset_id="infant_coverage_gap",
        name="Infant clinical seizure EEG cohort: unresolved",
        source_url=None,
        target_age_groups=(AgeGroup.INFANT,),
        published_population_notes="No suitable 29-day to under-one-year cohort verified for this project.",
        review_tasks=("Identify a clinically annotated cohort and independently verify all permissions and metadata.",),
        status="unresolved_gap",
    ),
)


def candidates_for_age(age_group: AgeGroup) -> tuple[DatasetCandidate, ...]:
    """Discovery only; never infer recording age or model eligibility from this."""
    age_group = AgeGroup(age_group)
    return tuple(item for item in DATASET_REGISTRY if age_group in item.target_age_groups)
