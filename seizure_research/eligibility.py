"""Reject-by-default compatibility checks. This module never runs a model."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import Field, StrictBool, ValidationError, model_validator

from .schemas import (
    AgeGroup, ChannelMontage, EEGRecording, PositiveInt, PositiveNumber,
    RecordingDevice, ResearchSchema, SHA256, Text, require_research_ready,
)


class AcquisitionProfile(ResearchSchema):
    profile_id: Text
    montage: ChannelMontage
    sampling_rate_hz: PositiveNumber
    device: RecordingDevice
    preprocessing_id: Text


class ValidationEvidence(ResearchSchema):
    """One reviewed age x acquisition cell, tied to exact model bytes."""

    age_group: AgeGroup
    acquisition_profile_id: Text
    model_sha256: SHA256
    dataset_refs: Annotated[tuple[Text, ...], Field(min_length=1)]
    split_manifest_sha256: SHA256
    evaluation_protocol_ref: Text
    metrics_report_ref: Text
    review_ref: Text
    reviewed: StrictBool = False
    patient_independent: StrictBool = False
    clinical_annotations: StrictBool = False
    patient_count: PositiveInt
    seizure_event_count: PositiveInt
    evaluated_hours: PositiveNumber
    neonatal_specialist_review_ref: Text | None = None


class SeizureModelSpec(ResearchSchema):
    model_id: Text
    version: Text
    model_sha256: SHA256
    purpose: Literal["offline_seizure_event_research"]
    supported_age_groups: Annotated[tuple[AgeGroup, ...], Field(min_length=1)]
    acquisition_profiles: Annotated[tuple[AcquisitionProfile, ...], Field(min_length=1)]
    validation_evidence: tuple[ValidationEvidence, ...] = ()

    @model_validator(mode="after")
    def check_profiles(self):
        ids = [profile.profile_id for profile in self.acquisition_profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("Acquisition profile IDs must be unique.")
        if len(set(self.supported_age_groups)) != len(self.supported_age_groups):
            raise ValueError("Supported age groups must be unique.")
        for evidence in self.validation_evidence:
            if evidence.acquisition_profile_id not in ids:
                raise ValueError("Evidence names an unknown acquisition profile.")
            if evidence.age_group not in self.supported_age_groups:
                raise ValueError("Evidence age group is not declared by the model.")
        return self


class EligibilityDecision(ResearchSchema):
    offline_research_eligible: StrictBool
    reasons: tuple[Text, ...]
    live_prediction_allowed: Literal[False] = False
    medical_alerts_allowed: Literal[False] = False


# No clinically validated seizure models are bundled or registered in Phase 1.
REGISTERED_MODELS: tuple[SeizureModelSpec, ...] = ()


def assess_eligibility(
    recording: EEGRecording | Mapping[str, Any],
    model: SeizureModelSpec | Mapping[str, Any] | None,
) -> EligibilityDecision:
    """Check a single explicit candidate. There is no fallback or model loader.

    Revalidate objects at this boundary, including objects made with Pydantic's
    unchecked model_copy/model_construct APIs. Never echo sensitive raw metadata.
    """
    try:
        recording = EEGRecording.model_validate(recording)
    except (ValidationError, TypeError, ValueError):
        return EligibilityDecision(offline_research_eligible=False, reasons=("invalid_recording_metadata",))
    if model is None:
        return EligibilityDecision(offline_research_eligible=False, reasons=("no_model_registered",))
    try:
        model = SeizureModelSpec.model_validate(model)
    except (ValidationError, TypeError, ValueError):
        return EligibilityDecision(offline_research_eligible=False, reasons=("invalid_model_metadata",))

    reasons = []
    try:
        require_research_ready(recording)
    except ValueError:
        reasons.append("recording_not_research_ready")
    age_group = recording.age.group
    if age_group not in model.supported_age_groups:
        reasons.append("unsupported_age_group")
    profiles = model.acquisition_profiles
    if not any(profile.montage == recording.montage for profile in profiles):
        reasons.append("incompatible_montage")
    if not any(profile.device == recording.device for profile in profiles):
        reasons.append("unsupported_device")
    if not any(profile.sampling_rate_hz == recording.sampling_rate_hz for profile in profiles):
        reasons.append("unsupported_sampling_rate")
    if not any(profile.preprocessing_id == recording.preprocessing_id for profile in profiles):
        reasons.append("unsupported_preprocessing")
    matching_profiles = {
        profile.profile_id for profile in profiles
        if profile.montage == recording.montage and profile.device == recording.device
        and profile.sampling_rate_hz == recording.sampling_rate_hz
        and profile.preprocessing_id == recording.preprocessing_id
    }
    if not matching_profiles:
        reasons.append("unsupported_acquisition_combination")
    # Evidence for adults never authorizes an infant, child, adolescent, neonate,
    # or older adult, even if someone adds that group to supported_age_groups.
    evidence = [item for item in model.validation_evidence
                if item.age_group == age_group
                and item.acquisition_profile_id in matching_profiles
                and item.model_sha256 == model.model_sha256
                and item.reviewed and item.patient_independent and item.clinical_annotations]
    if not evidence:
        reasons.append("missing_age_specific_validation")
    elif age_group == AgeGroup.NEONATAL and not any(
        item.neonatal_specialist_review_ref for item in evidence
    ):
        reasons.append("missing_neonatal_specialist_review")
    return EligibilityDecision(offline_research_eligible=not reasons, reasons=tuple(reasons))
