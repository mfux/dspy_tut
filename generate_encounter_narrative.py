#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""generate_encounter_narrative.py

Generate narrative summaries for patient encounters from Synthea EHR data.

This script reads structured patient and encounter profiles from CSV files
and uses DSPy to generate human-readable narrative summaries that medical
professionals can use in clinical settings.

Author(s): mfux
"""

import argparse
from pathlib import Path
from typing import Optional
import random

import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import mlflow
import dspy


#############
# Constants #
#############

DEFAULT_DATA_DIR: str = "data/synthea"
DEFAULT_SEED: int = 313
DEFAULT_TEMPERATURE: float = 1.0
DEFAULT_MAX_TOKENS: int = 16000
ARGS = None  # global variable to store command line arguments

#########
# Input #
#########


def parse_args(args=None) -> argparse.Namespace:
    """Runtime args parser."""
    parser = argparse.ArgumentParser(
        description="Generate narrative summaries for patient encounters from Synthea EHR data"
    )

    parser.add_argument(
        "--data-dir",
        help="Path to directory containing Synthea CSV files",
        type=str,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--encounter-id",
        help="Specific encounter ID to process (if not provided, a random encounter will be selected)",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--model",
        help="LLM model to use",
        type=str,
        default="openai/gpt-4o",
    )

    parser.add_argument(
        "--temperature",
        help="Temperature for LLM generation",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )

    parser.add_argument(
        "--max-tokens",
        help="Maximum tokens for LLM generation",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )

    parser.add_argument(
        "--mlflow-uri",
        help="MLflow tracking URI",
        type=str,
        default="http://localhost:5000",
    )

    parser.add_argument(
        "--no-mlflow",
        help="Disable MLflow logging",
        action="store_true",
    )

    return parser.parse_args(args)


#################
# Data Models   #
#################


class Date(BaseModel):
    """Represents a date."""

    year: int
    month: int
    day: int


class DateTime(BaseModel):
    """Represents a date and time."""

    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int


class PatientProfile(BaseModel):
    """Patient profile data model from EHR system."""

    # required fields
    id: str = Field(description="Unique patient identifier (UUID)")
    birthdate: Date
    ssn: str = Field(description="Patient's social security number")
    first: str = Field(description="Patient's first name")
    last: str = Field(description="Patient's last name")
    race: str = Field(description="Patient's race")
    ethnicity: str = Field(description="Patient's ethnicity (hispanic/nonhispanic)")
    gender: str = Field(description="Patient's gender (M/F)")
    birthplace: str = Field(description="Patient's birthplace")
    address: str = Field(description="Patient's street address")
    city: str = Field(description="Patient's city")
    state: str = Field(description="Patient's state")
    county: str = Field(description="Patient's county")

    # optional fields
    deathdate: Optional[Date] = Field(
        None, description="Patient's death date, or None if alive"
    )
    drivers: Optional[str] = Field(
        None,
        description="Patient's driver's license number or None if patient does not have one",
    )
    passport: Optional[str] = Field(
        None, description="Patient's passport number or None if not applicable"
    )
    prefix: Optional[str] = Field(
        None,
        description="Patient's prefix (e.g., Mr., Mrs., Dr.) or None if not applicable",
    )
    suffix: Optional[str] = Field(
        None,
        description="Patient's suffix (e.g., Jr., Sr., PhD) or None if not applicable",
    )
    maiden: Optional[str] = Field(
        None, description="Patient's maiden name or None if not applicable"
    )
    zip: Optional[str] = Field(
        None, description="Patient's ZIP code or None if not available"
    )
    marital: Optional[str] = Field(
        None,
        description="Marital status (M=Married, S=Single) or None if not specified",
    )

    # numeric fields
    lat: float = Field(description="Patient's address latitude")
    lon: float = Field(description="Patient's address longitude")
    healthcare_expenses: float = Field(description="Total healthcare expenses")
    healthcare_coverage: float = Field(description="Total healthcare coverage amount")

    @classmethod
    def from_row(cls, row) -> "PatientProfile":
        """Convert a DataFrame row to a PatientProfile object."""
        return cls(
            id=row["Id"],
            birthdate=parse_date(row["BIRTHDATE"]),
            ssn=row["SSN"],
            first=row["FIRST"],
            last=row["LAST"],
            marital=None if pd.isna(row["MARITAL"]) else row["MARITAL"],
            race=row["RACE"],
            ethnicity=row["ETHNICITY"],
            gender=row["GENDER"],
            birthplace=row["BIRTHPLACE"],
            address=row["ADDRESS"],
            city=row["CITY"],
            state=row["STATE"],
            county=row["COUNTY"],
            deathdate=parse_date(row["DEATHDATE"]),
            drivers=None if pd.isna(row["DRIVERS"]) else row["DRIVERS"],
            passport=None if pd.isna(row["PASSPORT"]) else row["PASSPORT"],
            prefix=None if pd.isna(row["PREFIX"]) else row["PREFIX"],
            suffix=None if pd.isna(row["SUFFIX"]) else row["SUFFIX"],
            maiden=None if pd.isna(row["MAIDEN"]) else row["MAIDEN"],
            zip=None if pd.isna(row["ZIP"]) else str(int(row["ZIP"])),
            lat=float(row["LAT"]),
            lon=float(row["LON"]),
            healthcare_expenses=float(row["HEALTHCARE_EXPENSES"]),
            healthcare_coverage=float(row["HEALTHCARE_COVERAGE"]),
        )


class EncounterProfile(BaseModel):
    """Encounter profile data model from EHR system."""

    # required fields
    id: str = Field(
        description="Unique encounter identifier (UUID) used to join with other encounter-related tables"
    )
    start: DateTime = Field(
        description="UTC timestamp (date + time) when the encounter began; sourced from START column"
    )
    stop: DateTime = Field(
        description="UTC timestamp (date + time) when the encounter ended; sourced from STOP column"
    )
    patient: str = Field(
        description="Foreign key (UUID) referencing the patient involved in this encounter (PatientProfile.id)"
    )
    organization: str = Field(
        description="UUID of the facility or organizational entity where services were delivered (e.g., clinic, hospital department)"
    )
    provider: str = Field(
        description="UUID of the primary clinician or provider responsible for the encounter"
    )
    payer: str = Field(
        description="UUID of the payer (insurance or funding entity) associated with claim adjudication for this encounter"
    )
    encounterclass: str = Field(
        description="High-level setting/category of the encounter (e.g., 'ambulatory', 'wellness'); useful for stratifying utilization"
    )
    code: str = Field(
        description="SNOMED CT concept code representing the primary encounter type or procedure focus"
    )
    description: str = Field(
        description="Human-readable categorical label tied to the encounter SNOMED code (e.g., 'Encounter for symptom', 'General examination of patient (procedure)', 'Consultation for treatment')"
    )

    # numeric fields
    base_encounter_cost: float = Field(
        description="Baseline assessed cost for the encounter before claim adjustments or payer calculations"
    )
    total_claim_cost: float = Field(
        description="Final total cost submitted on the claim after adjustments (patient + payer portions combined)"
    )
    payer_coverage: float = Field(
        description="Monetary amount covered by the payer for this encounter; total_claim_cost - payer_coverage approximates patient or other responsibility"
    )

    # optional fields
    reasoncode: Optional[float] = Field(
        None,
        description="SNOMED CT code for the clinical reason/chief complaint or diagnosis motivating the encounter; None if not recorded",
    )
    reasondescription: Optional[str] = Field(
        None,
        description="Human-readable label for reasoncode (e.g., 'Acute bronchitis (disorder)'); None if no reason supplied",
    )

    @classmethod
    def from_row(cls, row) -> "EncounterProfile":
        """Convert a DataFrame row to an EncounterProfile object."""
        return cls(
            id=row["Id"],
            start=parse_datetime(row["START"]),
            stop=parse_datetime(row["STOP"]),
            patient=row["PATIENT"],
            organization=row["ORGANIZATION"],
            provider=row["PROVIDER"],
            payer=row["PAYER"],
            encounterclass=row["ENCOUNTERCLASS"],
            code=str(row["CODE"]),
            description=row["DESCRIPTION"],
            base_encounter_cost=float(row["BASE_ENCOUNTER_COST"]),
            total_claim_cost=float(row["TOTAL_CLAIM_COST"]),
            payer_coverage=float(row["PAYER_COVERAGE"]),
            reasoncode=None if pd.isna(row["REASONCODE"]) else float(row["REASONCODE"]),
            reasondescription=None
            if pd.isna(row["REASONDESCRIPTION"])
            else row["REASONDESCRIPTION"],
        )


class ObservationProfile(BaseModel):
    """Observation profile data model from EHR system."""

    # required fields
    date: DateTime = Field(
        description="UTC timestamp (date + time) when the observation was recorded"
    )
    patient: str = Field(
        description="Foreign key (UUID) referencing the patient for whom this observation was recorded (PatientProfile.id)"
    )
    code: str = Field(
        description="LOINC code identifying the type of observation (e.g., '8302-2' for Body Height)"
    )
    description: str = Field(
        description="Human-readable label describing what was observed (e.g., 'Body Height', 'Blood Pressure', 'Heart Rate')"
    )
    value: str = Field(
        description="The measured or observed value; may be numeric (e.g., '193.3'), text, or coded depending on observation type"
    )
    type: str = Field(
        description="Data type classification of the observation value (e.g., 'numeric', 'text', 'code') indicating how to interpret the value field"
    )

    # optional fields
    encounter: Optional[str] = Field(
        None,
        description="Foreign key (UUID) referencing the encounter during which this observation was made (EncounterProfile.id); None if observation was recorded outside an encounter context (e.g., patient-reported data, ongoing monitoring)",
    )
    units: Optional[str] = Field(
        None,
        description="Unit of measurement for the observation value (e.g., 'cm', 'mmHg', 'beats/min'); None for non-quantitative observations or observations without applicable units",
    )

    @classmethod
    def from_row(cls, row) -> "ObservationProfile":
        """Convert a DataFrame row to an ObservationProfile object."""
        return cls(
            date=parse_datetime(row["DATE"]),
            patient=row["PATIENT"],
            encounter=None if pd.isna(row["ENCOUNTER"]) else row["ENCOUNTER"],
            code=str(row["CODE"]),
            description=row["DESCRIPTION"],
            value=str(row["VALUE"]),
            units=None if pd.isna(row["UNITS"]) else row["UNITS"],
            type=row["TYPE"],
        )


class ImmunizationProfile(BaseModel):
    """Immunization profile data model from EHR system."""

    # required fields
    date: DateTime = Field(
        description="UTC timestamp (date + time) when the immunization was administered"
    )
    patient: str = Field(
        description="Foreign key (UUID) referencing the patient who received the immunization (PatientProfile.id)"
    )
    code: int = Field(
        description="CVX (Vaccine Administered) code identifying the specific vaccine administered"
    )
    description: str = Field(
        description="Human-readable label describing the vaccine (e.g., 'Influenza seasonal injectable preservative free', 'Hepatitis B adult')"
    )
    base_cost: float = Field(description="Base cost of the immunization administration")

    # optional fields
    encounter: Optional[str] = Field(
        None,
        description="Foreign key (UUID) referencing the encounter during which this immunization was administered (EncounterProfile.id); None if immunization was recorded outside an encounter context",
    )

    @classmethod
    def from_row(cls, row) -> "ImmunizationProfile":
        """Convert a DataFrame row to an ImmunizationProfile object."""
        return cls(
            date=parse_datetime(row["DATE"]),
            patient=row["PATIENT"],
            encounter=None if pd.isna(row["ENCOUNTER"]) else row["ENCOUNTER"],
            code=int(row["CODE"]),
            description=row["DESCRIPTION"],
            base_cost=float(row["BASE_COST"]),
        )


class MedicationProfile(BaseModel):
    """Medication profile data model from EHR system."""

    # required fields
    start: DateTime = Field(
        description="UTC timestamp (date + time) when the medication was started"
    )
    patient: str = Field(
        description="Foreign key (UUID) referencing the patient who was prescribed the medication (PatientProfile.id)"
    )
    payer: str = Field(
        description="UUID of the payer (insurance or funding entity) covering the medication costs"
    )
    code: int = Field(
        description="RxNorm code identifying the specific medication prescribed"
    )
    description: str = Field(
        description="Human-readable label describing the medication (e.g., 'Etonogestrel 68 MG Drug Implant', 'Amoxicillin 250 MG')"
    )
    base_cost: float = Field(
        description="Base cost per dispense of the medication before insurance coverage"
    )
    payer_coverage: float = Field(
        description="Amount covered by the payer for the medication"
    )
    dispenses: int = Field(
        description="Number of times the medication was dispensed during the prescription period"
    )
    totalcost: float = Field(
        description="Total cost for all dispenses of the medication (base_cost * dispenses)"
    )

    # optional fields
    stop: Optional[DateTime] = Field(
        None,
        description="UTC timestamp (date + time) when the medication was stopped; None if medication is ongoing or stop date not recorded",
    )
    encounter: Optional[str] = Field(
        None,
        description="Foreign key (UUID) referencing the encounter during which this medication was prescribed (EncounterProfile.id); None if prescribed outside an encounter context",
    )
    reasoncode: Optional[float] = Field(
        None,
        description="SNOMED CT code for the clinical reason/diagnosis for which the medication was prescribed; None if not recorded",
    )
    reasondescription: Optional[str] = Field(
        None,
        description="Human-readable label for reasoncode (e.g., 'Hypertension', 'Type 2 diabetes'); None if no reason supplied",
    )

    @classmethod
    def from_row(cls, row) -> "MedicationProfile":
        """Convert a DataFrame row to a MedicationProfile object."""
        return cls(
            start=parse_datetime(row["START"]),
            stop=parse_datetime(row["STOP"]),
            patient=row["PATIENT"],
            payer=row["PAYER"],
            encounter=None if pd.isna(row["ENCOUNTER"]) else row["ENCOUNTER"],
            code=int(row["CODE"]),
            description=row["DESCRIPTION"],
            base_cost=float(row["BASE_COST"]),
            payer_coverage=float(row["PAYER_COVERAGE"]),
            dispenses=int(row["DISPENSES"]),
            totalcost=float(row["TOTALCOST"]),
            reasoncode=None if pd.isna(row["REASONCODE"]) else float(row["REASONCODE"]),
            reasondescription=None
            if pd.isna(row["REASONDESCRIPTION"])
            else row["REASONDESCRIPTION"],
        )


class ProcedureProfile(BaseModel):
    """Procedure profile data model from EHR system."""

    # required fields
    date: DateTime = Field(
        description="UTC timestamp (date + time) when the procedure was performed"
    )
    patient: str = Field(
        description="Foreign key (UUID) referencing the patient who received the procedure (PatientProfile.id)"
    )
    code: int = Field(
        description="SNOMED CT code identifying the specific procedure performed"
    )
    description: str = Field(
        description="Human-readable label describing the procedure (e.g., 'Insertion of subcutaneous contraceptive', 'Appendectomy')"
    )
    base_cost: float = Field(
        description="Base cost of the procedure before insurance coverage"
    )

    # optional fields
    encounter: Optional[str] = Field(
        None,
        description="Foreign key (UUID) referencing the encounter during which this procedure was performed (EncounterProfile.id); None if procedure was recorded outside an encounter context",
    )
    reasoncode: Optional[float] = Field(
        None,
        description="SNOMED CT code for the clinical reason/diagnosis for which the procedure was performed; None if not recorded",
    )
    reasondescription: Optional[str] = Field(
        None,
        description="Human-readable label for reasoncode (e.g., 'Contraception', 'Acute appendicitis'); None if no reason supplied",
    )

    @classmethod
    def from_row(cls, row) -> "ProcedureProfile":
        """Convert a DataFrame row to a ProcedureProfile object."""
        return cls(
            date=parse_datetime(row["DATE"]),
            patient=row["PATIENT"],
            encounter=None if pd.isna(row["ENCOUNTER"]) else row["ENCOUNTER"],
            code=int(row["CODE"]),
            description=row["DESCRIPTION"],
            base_cost=float(row["BASE_COST"]),
            reasoncode=None if pd.isna(row["REASONCODE"]) else float(row["REASONCODE"]),
            reasondescription=None
            if pd.isna(row["REASONDESCRIPTION"])
            else row["REASONDESCRIPTION"],
        )


class CarePlanProfile(BaseModel):
    """Care plan profile data model from EHR system."""

    # required fields
    id: str = Field(description="Unique care plan identifier (UUID)")
    start: Date = Field(description="Date when the care plan was initiated")
    patient: str = Field(
        description="Foreign key (UUID) referencing the patient for whom this care plan was created (PatientProfile.id)"
    )
    code: int = Field(
        description="SNOMED CT code identifying the specific care plan type"
    )
    description: str = Field(
        description="Human-readable label describing the care plan (e.g., 'Respiratory therapy', 'Diabetes self management plan')"
    )

    # optional fields
    stop: Optional[Date] = Field(
        None,
        description="Date when the care plan was stopped or completed; None if care plan is ongoing",
    )
    encounter: Optional[str] = Field(
        None,
        description="Foreign key (UUID) referencing the encounter during which this care plan was created (EncounterProfile.id); None if created outside an encounter context",
    )
    reasoncode: Optional[float] = Field(
        None,
        description="SNOMED CT code for the clinical reason/diagnosis for which the care plan was created; None if not recorded",
    )
    reasondescription: Optional[str] = Field(
        None,
        description="Human-readable label for reasoncode (e.g., 'Acute bronchitis (disorder)', 'Type 2 diabetes'); None if no reason supplied",
    )

    @classmethod
    def from_row(cls, row) -> "CarePlanProfile":
        """Convert a DataFrame row to a CarePlanProfile object."""
        return cls(
            id=row["Id"],
            start=parse_date(row["START"]),
            stop=parse_date(row["STOP"]),
            patient=row["PATIENT"],
            encounter=None if pd.isna(row["ENCOUNTER"]) else row["ENCOUNTER"],
            code=int(row["CODE"]),
            description=row["DESCRIPTION"],
            reasoncode=None if pd.isna(row["REASONCODE"]) else float(row["REASONCODE"]),
            reasondescription=None
            if pd.isna(row["REASONDESCRIPTION"])
            else row["REASONDESCRIPTION"],
        )


class ConditionProfile(BaseModel):
    """Condition profile data model from EHR system."""

    # required fields
    start: Date = Field(
        description="Date when the condition was first diagnosed or recorded"
    )
    patient: str = Field(
        description="Foreign key (UUID) referencing the patient with this condition (PatientProfile.id)"
    )
    code: int = Field(description="SNOMED CT code identifying the specific condition")
    description: str = Field(
        description="Human-readable label describing the condition (e.g., 'Chronic sinusitis (disorder)', 'Hypertension')"
    )

    # optional fields
    stop: Optional[Date] = Field(
        None,
        description="Date when the condition was resolved or no longer active; None if condition is ongoing",
    )
    encounter: Optional[str] = Field(
        None,
        description="Foreign key (UUID) referencing the encounter during which this condition was diagnosed (EncounterProfile.id); None if diagnosed outside an encounter context",
    )

    @classmethod
    def from_row(cls, row) -> "ConditionProfile":
        """Convert a DataFrame row to a ConditionProfile object."""
        return cls(
            start=parse_date(row["START"]),
            stop=parse_date(row["STOP"]),
            patient=row["PATIENT"],
            encounter=None if pd.isna(row["ENCOUNTER"]) else row["ENCOUNTER"],
            code=int(row["CODE"]),
            description=row["DESCRIPTION"],
        )


class DeviceProfile(BaseModel):
    """Device profile data model from EHR system."""

    # required fields
    start: DateTime = Field(
        description="UTC timestamp (date + time) when the device was implanted or assigned"
    )
    patient: str = Field(
        description="Foreign key (UUID) referencing the patient who received the device (PatientProfile.id)"
    )
    code: int = Field(description="SNOMED CT code identifying the specific device type")
    description: str = Field(
        description="Human-readable label describing the device (e.g., 'Implantable defibrillator device (physical object)', 'Insulin pump')"
    )
    udi: str = Field(
        description="Unique Device Identifier (UDI) for the specific device instance"
    )

    # optional fields
    stop: Optional[DateTime] = Field(
        None,
        description="UTC timestamp (date + time) when the device was removed or deactivated; None if device is still active",
    )
    encounter: Optional[str] = Field(
        None,
        description="Foreign key (UUID) referencing the encounter during which this device was implanted or assigned (EncounterProfile.id); None if assigned outside an encounter context",
    )

    @classmethod
    def from_row(cls, row) -> "DeviceProfile":
        """Convert a DataFrame row to a DeviceProfile object."""
        return cls(
            start=parse_datetime(row["START"]),
            stop=parse_datetime(row["STOP"]),
            patient=row["PATIENT"],
            encounter=None if pd.isna(row["ENCOUNTER"]) else row["ENCOUNTER"],
            code=int(row["CODE"]),
            description=row["DESCRIPTION"],
            udi=row["UDI"],
        )


class ImagingStudyProfile(BaseModel):
    """Imaging study profile data model from EHR system."""

    # required fields
    id: str = Field(description="Unique imaging study identifier (UUID)")
    date: DateTime = Field(
        description="UTC timestamp (date + time) when the imaging study was performed"
    )
    patient: str = Field(
        description="Foreign key (UUID) referencing the patient who received the imaging study (PatientProfile.id)"
    )
    bodysite_code: int = Field(
        description="SNOMED CT code identifying the body site that was imaged"
    )
    bodysite_description: str = Field(
        description="Human-readable label describing the body site (e.g., 'Arm', 'Chest', 'Head')"
    )
    modality_code: str = Field(
        description="DICOM modality code identifying the imaging technique (e.g., 'DX' for Digital Radiography, 'CT', 'MR')"
    )
    modality_description: str = Field(
        description="Human-readable label describing the imaging modality (e.g., 'Digital Radiography', 'Computed Tomography')"
    )
    sop_code: str = Field(
        description="DICOM SOP (Service-Object Pair) class UID identifying the type of image storage"
    )
    sop_description: str = Field(
        description="Human-readable label describing the SOP class (e.g., 'Digital X-Ray Image Storage')"
    )

    # optional fields
    encounter: Optional[str] = Field(
        None,
        description="Foreign key (UUID) referencing the encounter during which this imaging study was performed (EncounterProfile.id); None if performed outside an encounter context",
    )

    @classmethod
    def from_row(cls, row) -> "ImagingStudyProfile":
        """Convert a DataFrame row to an ImagingStudyProfile object."""
        return cls(
            id=row["Id"],
            date=parse_datetime(row["DATE"]),
            patient=row["PATIENT"],
            encounter=None if pd.isna(row["ENCOUNTER"]) else row["ENCOUNTER"],
            bodysite_code=int(row["BODYSITE_CODE"]),
            bodysite_description=row["BODYSITE_DESCRIPTION"],
            modality_code=row["MODALITY_CODE"],
            modality_description=row["MODALITY_DESCRIPTION"],
            sop_code=row["SOP_CODE"],
            sop_description=row["SOP_DESCRIPTION"],
        )


###################
# DSPy Signatures #
###################


class PatientProfileNarrator(dspy.Signature):
    """Reads a structured patient profile and narrates it as a medical professional would do in the context of a conversation with a colleague. The narrator wants to give a concise summary of the patient profile, focusing on the most relevant attributes for a medical professional in a clinical setting. The colleague should be able to use the information for generating a medical report or for further analysis of the patient profile."""

    patient_profile: PatientProfile = dspy.InputField(
        desc="A structured patient profile from the EHR system. The profile includes relevant attributes like name, gender, birthdate etc. but also attributes that might not be relevant for the conversation like social security number. Some attributes like Address may only be relevant if it is something unusual like homelessness or if its in a tropical country for example."
    )

    narrative: str = dspy.OutputField(
        desc="The narrative summary of the patient profile."
    )


class EncounterProfileNarrator(dspy.Signature):
    """Reads a structured encounter profile and narrates it as a medical professional would do in the context of a conversation with a colleague. The narrator wants to give a concise summary of the encounter, focusing on the most relevant attributes for a medical professional in a clinical setting. The colleague should be able to use the information for generating a medical report or for further analysis of the encounter profile."""

    encounter_profile: EncounterProfile = dspy.InputField(
        desc="A structured encounter profile from the EHR system. The profile includes relevant attributes like encounterclass, description, reasondescription etc. but also attributes that might not be relevant for the conversation like code or id. Some attributes like start and end may only be relevant in combination if it is something unusual like a very long stay."
    )

    patient_profile_narrative: str = dspy.InputField(
        desc="A narrative summary of the patient profile associated with the encounter."
    )

    observations_narrative: Optional[str] = dspy.InputField(
        default=None,
        desc="A narrative summary of the observations recorded during the encounter.",
    )

    immunizations_narrative: Optional[str] = dspy.InputField(
        default=None,
        desc="A narrative summary of the immunizations administered. Compare the date of the immunizations with the encounter date to infer if they were given during the encounter.",
    )

    medications_narrative: Optional[str] = dspy.InputField(
        default=None,
        desc="A narrative summary of the medications prescribed. Compare the start date of the medications with the encounter date to infer if they were prescribed during the encounter.",
    )

    procedures_narrative: Optional[str] = dspy.InputField(
        default=None,
        desc="A narrative summary of the procedures performed.",
    )

    careplans_narrative: Optional[str] = dspy.InputField(
        default=None,
        desc="A narrative summary of the care plans created or updated. Compare the start date of the care plans with the encounter date to infer if they were created during the encounter.",
    )

    conditions_narrative: Optional[str] = dspy.InputField(
        default=None,
        desc="A narrative summary of the conditions diagnosed or documented. Compare the start date of the conditions with the encounter date to infer if they were diagnosed during the encounter.",
    )

    devices_narrative: Optional[str] = dspy.InputField(
        default=None,
        desc="A narrative summary of the devices implanted or assigned. Compare the start date of the devices with the encounter date to infer if they were implanted during the encounter.",
    )

    imaging_studies_narrative: Optional[str] = dspy.InputField(
        default=None,
        desc="A narrative summary of the imaging studies performed. Compare the date of the imaging studies with the encounter date to infer if they were performed during the encounter.",
    )

    narrative: str = dspy.OutputField(
        desc="The narrative summary of the encounter profile."
    )


class ObservationNarrator(dspy.Signature):
    """Reads a list of structured observation profiles from a clinical EHR system belonging to a specific encounter. The output is a narrative summary of the observations. The purpose is to give a concise overview of the observations for the clinical documentation specialist who is trying to get all relevant information to create a complete overview of the encounter."""

    observation_profiles: list[ObservationProfile] = dspy.InputField(
        desc="A list of structured observation profiles from the EHR system. Each profile includes relevant attributes like description, value, units, and type. Some attributes like code or patient ID might not be relevant for the narrative conversation. Focus on grouping related observations (e.g., vital signs, lab results) and highlighting any abnormal or clinically significant values."
    )

    narrative: str = dspy.OutputField(
        desc="The narrative summary of the observations, organized by clinical relevance and highlighting key findings."
    )


class ImmunizationNarrator(dspy.Signature):
    """Reads a list of structured immunization profiles from a clinical EHR system belonging to a specific encounter. The output is a narrative summary of the immunizations. The purpose is to give a concise overview of the immunizations administered for the clinical documentation specialist who is trying to get all relevant information to create a complete overview of the encounter."""

    immunization_profiles: list[ImmunizationProfile] = dspy.InputField(
        desc="A list of structured immunization profiles from the EHR system. Each profile includes relevant attributes like description, date, and base_cost. Some attributes like code or patient ID might not be relevant for the narrative conversation. Focus on listing the vaccines administered and any notable clinical context."
    )

    narrative: str = dspy.OutputField(
        desc="The narrative summary of the immunizations, organized chronologically and highlighting key vaccines administered and the dates they were given."
    )


class MedicationNarrator(dspy.Signature):
    """Reads a list of structured medication profiles from a clinical EHR system belonging to a specific encounter. The output is a narrative summary of the medications. The purpose is to give a concise overview of the medications prescribed for the clinical documentation specialist who is trying to get all relevant information to create a complete overview of the encounter."""

    medication_profiles: list[MedicationProfile] = dspy.InputField(
        desc="A list of structured medication profiles from the EHR system. Each profile includes relevant attributes like description, start/stop dates, reason for prescription, dispenses, and costs. Some attributes like code or patient ID might not be relevant for the narrative conversation. Focus on listing the medications prescribed, their clinical indications, duration, and any notable information such as ongoing medications or reasons for prescription."
    )

    narrative: str = dspy.OutputField(
        desc="The narrative summary of the medications, organized by clinical relevance and highlighting key prescriptions, their indications, and treatment duration."
    )


class ProcedureNarrator(dspy.Signature):
    """Reads a list of structured procedure profiles from a clinical EHR system belonging to a specific encounter. The output is a narrative summary of the procedures. The purpose is to give a concise overview of the procedures performed for the clinical documentation specialist who is trying to get all relevant information to create a complete overview of the encounter."""

    procedure_profiles: list[ProcedureProfile] = dspy.InputField(
        desc="A list of structured procedure profiles from the EHR system. Each profile includes relevant attributes like description, date, reason for procedure, and base_cost. Some attributes like code or patient ID might not be relevant for the narrative. Focus on listing the procedures performed, their clinical indications, and any notable clinical context."
    )

    narrative: str = dspy.OutputField(desc="The narrative summary of the procedures.")


class CarePlanNarrator(dspy.Signature):
    """Reads a list of structured care plan profiles from a clinical EHR system belonging to a specific encounter. The output is a narrative summary of the care plans. The purpose is to give a concise overview of the care plans created or updated for the clinical documentation specialist who is trying to get all relevant information to create a complete overview of the encounter."""

    careplan_profiles: list[CarePlanProfile] = dspy.InputField(
        desc="A list of structured care plan profiles from the EHR system. Each profile includes relevant attributes like description, start/stop dates, and reason for the care plan. Some attributes like code or patient ID might not be relevant for the narrative. Focus on listing the care plans created, their clinical indications, duration, and any notable information such as ongoing care plans."
    )

    narrative: str = dspy.OutputField(
        desc="The narrative summary of the care plans, organized by clinical relevance and highlighting key care plans, their indications, and treatment duration."
    )


class ConditionNarrator(dspy.Signature):
    """Reads a list of structured condition profiles from a clinical EHR system belonging to a specific encounter. The output is a narrative summary of the conditions. The purpose is to give a concise overview of the conditions diagnosed or documented for the clinical documentation specialist who is trying to get all relevant information to create a complete overview of the encounter."""

    condition_profiles: list[ConditionProfile] = dspy.InputField(
        desc="A list of structured condition profiles from the EHR system. Each profile includes relevant attributes like description and start/stop dates. Some attributes like code or patient ID might not be relevant for the narrative. Focus on listing the conditions diagnosed, their onset dates, and whether they are ongoing or resolved."
    )

    narrative: str = dspy.OutputField(
        desc="The narrative summary of the conditions, organized by clinical significance and highlighting key diagnoses and their current status."
    )


class DeviceNarrator(dspy.Signature):
    """Reads a list of structured device profiles from a clinical EHR system belonging to a specific encounter. The output is a narrative summary of the devices. The purpose is to give a concise overview of the devices implanted or assigned for the clinical documentation specialist who is trying to get all relevant information to create a complete overview of the encounter."""

    device_profiles: list[DeviceProfile] = dspy.InputField(
        desc="A list of structured device profiles from the EHR system. Each profile includes relevant attributes like description, start/stop dates, and UDI. Some attributes like code or patient ID might not be relevant for the narrative. Focus on listing the devices implanted or assigned, their dates, and whether they are still active."
    )

    narrative: str = dspy.OutputField(
        desc="The narrative summary of the devices, organized chronologically and highlighting key devices implanted or assigned and their current status."
    )


class ImagingStudyNarrator(dspy.Signature):
    """Reads a list of structured imaging study profiles from a clinical EHR system belonging to a specific encounter. The output is a narrative summary of the imaging studies. The purpose is to give a concise overview of the imaging studies performed for the clinical documentation specialist who is trying to get all relevant information to create a complete overview of the encounter."""

    imaging_study_profiles: list[ImagingStudyProfile] = dspy.InputField(
        desc="A list of structured imaging study profiles from the EHR system. Each profile includes relevant attributes like body site, modality description, and date. Some attributes like codes or patient ID might not be relevant for the narrative. Focus on listing the imaging studies performed, the body sites examined, and the imaging modalities used."
    )

    narrative: str = dspy.OutputField(
        desc="The narrative summary of the imaging studies, organized by body site or chronologically and highlighting key studies performed."
    )


class DocumentationGuru(dspy.Signature):
    """Reads a narration of a clinical encounter. Is able to predict from his extensive experience as a clinical documentation specialist, which kind of clinical documentation was likely created during the encounter. He will output a complete list of all documents that were created during the encounter, including the type of document and a brief description of its content."""

    encounter_narration: str = dspy.InputField(
        desc="A narration of a clinical encounter, including all relevant information about the encounter, such as patient demographics, clinical findings, procedures performed, medications prescribed, and any other relevant information if applicable."
    )

    documents: list[dict[str, str]] = dspy.OutputField(
        desc="A list of dictionaries representing the documents created during the encounter. Each dictionary contains 'type' (e.g., 'Clinical Note', 'Discharge Summary') and 'description' (a brief summary of the document's content)."
    )


###########
# Helpers #
###########


def parse_date(date_str) -> Optional[Date]:
    """Parse a date string or NaN value into a Date object or None."""
    if pd.isna(date_str) or date_str == "":
        return None
    # parse date string in format YYYY-MM-DD
    year, month, day = date_str.split("-")
    return Date(year=int(year), month=int(month), day=int(day))


def parse_datetime(datetime_str) -> Optional[DateTime]:
    """Parse a datetime string or NaN value into a DateTime object or None."""
    if pd.isna(datetime_str) or datetime_str == "":
        return None
    # parse datetime string in format YYYY-MM-DDTHH:MM:SSZ
    date_part, time_part = datetime_str.rstrip("Z").split("T")
    year, month, day = date_part.split("-")
    hour, minute, second = time_part.split(":")
    return DateTime(
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(hour),
        minute=int(minute),
        second=int(second),
    )


def load_synthea_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all CSV files from the Synthea data directory."""
    dataframes = {}
    for csv_file in data_dir.glob("*.csv"):
        dataframes[csv_file.stem] = pd.read_csv(csv_file, header=0)
    return dataframes


def get_encounter(
    dataframes: dict[str, pd.DataFrame], encounter_id: Optional[str] = None
):
    """Get a specific encounter or a random one if no ID is provided."""
    encounters_df = dataframes["encounters"]

    if encounter_id:
        # filter for specific encounter
        encounter_rows = encounters_df[encounters_df["Id"] == encounter_id]
        if encounter_rows.empty:
            raise ValueError(f"Encounter ID '{encounter_id}' not found in the data")
        encounter_row = encounter_rows.iloc[0]
    else:
        # sample random encounter
        encounter_row = encounters_df.sample().iloc[0]

    return encounter_row


def get_encounter_related_data(
    dataframes: dict[str, pd.DataFrame], encounter_id: str
) -> dict[str, pd.DataFrame]:
    """Get all dataframes that have encounter-related data for a specific encounter.

    Args:
        dataframes: Dictionary of all dataframes indexed by table name
        encounter_id: The encounter ID to filter by

    Returns:
        Dictionary of filtered dataframes (indexed by table name) that contain
        rows related to the specified encounter. Only includes dataframes that
        have an 'ENCOUNTER' column and have at least one matching row.
    """
    encounter_dataframes = {}

    for df_name, df in dataframes.items():
        if "ENCOUNTER" in df.columns:
            filtered_df = df[df["ENCOUNTER"] == encounter_id]
            if not filtered_df.empty:
                encounter_dataframes[df_name] = filtered_df

    return encounter_dataframes


def generate_patient_narrative(patient_profile: PatientProfile) -> str:
    """Generate a narrative summary for a patient profile."""
    narrator = dspy.ChainOfThought(PatientProfileNarrator)
    narrative = narrator(patient_profile=patient_profile).narrative
    return narrative


def generate_encounter_narrative(
    encounter_profile: EncounterProfile,
    patient_profile_narrative: str,
    observations_narrative: str = None,
    immunizations_narrative: str = None,
    medications_narrative: str = None,
    procedures_narrative: str = None,
    careplans_narrative: str = None,
    conditions_narrative: str = None,
    devices_narrative: str = None,
    imaging_studies_narrative: str = None,
) -> str:
    """Generate a narrative summary for an encounter profile."""
    narrator = dspy.ChainOfThought(EncounterProfileNarrator)
    narrative = narrator(
        encounter_profile=encounter_profile,
        patient_profile_narrative=patient_profile_narrative,
        observations_narrative=observations_narrative,
        immunizations_narrative=immunizations_narrative,
        medications_narrative=medications_narrative,
        procedures_narrative=procedures_narrative,
        careplans_narrative=careplans_narrative,
        conditions_narrative=conditions_narrative,
        devices_narrative=devices_narrative,
        imaging_studies_narrative=imaging_studies_narrative,
    ).narrative
    return narrative


def generate_observations_narrative(
    observation_profiles: list[ObservationProfile],
) -> str:
    """Generate a narrative summary for a list of observation profiles."""
    narrator = dspy.ChainOfThought(ObservationNarrator)
    narrative = narrator(observation_profiles=observation_profiles).narrative
    return narrative


def generate_immunizations_narrative(
    immunization_profiles: list[ImmunizationProfile],
) -> str:
    """Generate a narrative summary for a list of immunization profiles."""
    narrator = dspy.ChainOfThought(ImmunizationNarrator)
    narrative = narrator(immunization_profiles=immunization_profiles).narrative
    return narrative


def generate_medications_narrative(
    medication_profiles: list[MedicationProfile],
) -> str:
    """Generate a narrative summary for a list of medication profiles."""
    narrator = dspy.ChainOfThought(MedicationNarrator)
    narrative = narrator(medication_profiles=medication_profiles).narrative
    return narrative


def generate_procedures_narrative(
    procedure_profiles: list[ProcedureProfile],
) -> str:
    """Generate a narrative summary for a list of procedure profiles."""
    narrator = dspy.ChainOfThought(ProcedureNarrator)
    narrative = narrator(procedure_profiles=procedure_profiles).narrative
    return narrative


def generate_careplans_narrative(
    careplan_profiles: list[CarePlanProfile],
) -> str:
    """Generate a narrative summary for a list of care plan profiles."""
    narrator = dspy.ChainOfThought(CarePlanNarrator)
    narrative = narrator(careplan_profiles=careplan_profiles).narrative
    return narrative


def generate_conditions_narrative(
    condition_profiles: list[ConditionProfile],
) -> str:
    """Generate a narrative summary for a list of condition profiles."""
    narrator = dspy.ChainOfThought(ConditionNarrator)
    narrative = narrator(condition_profiles=condition_profiles).narrative
    return narrative


def generate_devices_narrative(
    device_profiles: list[DeviceProfile],
) -> str:
    """Generate a narrative summary for a list of device profiles."""
    narrator = dspy.ChainOfThought(DeviceNarrator)
    narrative = narrator(device_profiles=device_profiles).narrative
    return narrative


def generate_imaging_studies_narrative(
    imaging_study_profiles: list[ImagingStudyProfile],
) -> str:
    """Generate a narrative summary for a list of imaging study profiles."""
    narrator = dspy.ChainOfThought(ImagingStudyNarrator)
    narrative = narrator(imaging_study_profiles=imaging_study_profiles).narrative
    return narrative


def generate_documents_list(
    encounter_narration: str,
) -> list[dict[str, str]]:
    """Generate a list of documents likely created during the encounter."""
    guru = dspy.ChainOfThought(DocumentationGuru)
    documents = guru(encounter_narration=encounter_narration).documents
    return documents


def setup_dspy(model: str, temperature: float, max_tokens: int) -> None:
    """Configure DSPy with the specified LLM."""
    dspy.configure(lm=dspy.LM(model, temperature=temperature, max_tokens=max_tokens))


def setup_mlflow(mlflow_uri: str, experiment_name: str = "Astrik Doc Gen") -> None:
    """Configure MLflow tracking."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.dspy.autolog(log_evals=True, log_compiles=True, log_traces_from_compile=True)


#################
#     MAIN      #
#################


def main(args=None) -> None:
    """Main entry point for encounter narrative generation."""
    # parse args
    global ARGS
    ARGS = parse_args(args)

    # load environment variables
    load_dotenv()

    # set random seed
    random.seed(ARGS.seed)

    # setup dspy
    setup_dspy(ARGS.model, ARGS.temperature, ARGS.max_tokens)

    # setup mlflow
    if not ARGS.no_mlflow:
        setup_mlflow(ARGS.mlflow_uri)

    # load data
    data_dir = Path(ARGS.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")

    print(f"Loading data from {data_dir}...")
    dataframes = load_synthea_data(data_dir)
    print(f"Loaded {len(dataframes)} tables")

    # get encounter
    print("\nSelecting encounter...")
    encounter_row = get_encounter(dataframes, ARGS.encounter_id)
    encounter_profile = EncounterProfile.from_row(encounter_row)
    print(f"Encounter ID: {encounter_profile.id}")
    print(f"Description: {encounter_profile.description}")

    # get associated patient
    print("\nRetrieving patient information...")
    patients_df = dataframes["patients"]
    patient_row = patients_df.loc[patients_df["Id"] == encounter_profile.patient].iloc[
        0
    ]
    patient_profile = PatientProfile.from_row(patient_row)
    print(
        f"Patient: {patient_profile.prefix} {patient_profile.first} {patient_profile.last}"
    )

    # get all encounter-related data
    print("\nRetrieving encounter-related data...")
    encounter_dataframes = get_encounter_related_data(dataframes, encounter_profile.id)
    for df_name, df in encounter_dataframes.items():
        print(f"  - {df_name}: {len(df)} row(s)")

    # generate observations narrative
    observation_narrative = None
    if "observations" in encounter_dataframes.keys():
        observation_profiles = []
        for _, row in encounter_dataframes["observations"].iterrows():
            observation_profiles.append(ObservationProfile.from_row(row))
        observation_narrative = generate_observations_narrative(observation_profiles)

    # generate immunizations narrative
    immunization_narrative = None
    if "immunizations" in encounter_dataframes.keys():
        immunization_profiles = []
        for _, row in encounter_dataframes["immunizations"].iterrows():
            immunization_profiles.append(ImmunizationProfile.from_row(row))
        immunization_narrative = generate_immunizations_narrative(immunization_profiles)

    # generate medications narrative
    medication_narrative = None
    if "medications" in encounter_dataframes.keys():
        medication_profiles = []
        for _, row in encounter_dataframes["medications"].iterrows():
            medication_profiles.append(MedicationProfile.from_row(row))
        medication_narrative = generate_medications_narrative(medication_profiles)

    # generate procedures narrative
    procedure_narrative = None
    if "procedures" in encounter_dataframes.keys():
        procedure_profiles = []
        for _, row in encounter_dataframes["procedures"].iterrows():
            procedure_profiles.append(ProcedureProfile.from_row(row))
        procedure_narrative = generate_procedures_narrative(procedure_profiles)

    # generate careplans narrative
    careplan_narrative = None
    if "careplans" in encounter_dataframes.keys():
        careplan_profiles = []
        for _, row in encounter_dataframes["careplans"].iterrows():
            careplan_profiles.append(CarePlanProfile.from_row(row))
        careplan_narrative = generate_careplans_narrative(careplan_profiles)

    # generate conditions narrative
    condition_narrative = None
    if "conditions" in encounter_dataframes.keys():
        condition_profiles = []
        for _, row in encounter_dataframes["conditions"].iterrows():
            condition_profiles.append(ConditionProfile.from_row(row))
        condition_narrative = generate_conditions_narrative(condition_profiles)

    # generate devices narrative
    device_narrative = None
    if "devices" in encounter_dataframes.keys():
        device_profiles = []
        for _, row in encounter_dataframes["devices"].iterrows():
            device_profiles.append(DeviceProfile.from_row(row))
        device_narrative = generate_devices_narrative(device_profiles)

    # generate imaging studies narrative
    imaging_study_narrative = None
    if "imaging_studies" in encounter_dataframes.keys():
        imaging_study_profiles = []
        for _, row in encounter_dataframes["imaging_studies"].iterrows():
            imaging_study_profiles.append(ImagingStudyProfile.from_row(row))
        imaging_study_narrative = generate_imaging_studies_narrative(
            imaging_study_profiles
        )

    # generate patient narrative
    print("\nGenerating patient narrative...")
    patient_narrative = generate_patient_narrative(patient_profile)
    print(f"\nPatient Narrative:\n{patient_narrative}")

    # generate encounter narrative
    print("\n" + "=" * 80)
    print("Generating encounter narrative...")
    encounter_narrative = generate_encounter_narrative(
        encounter_profile,
        patient_narrative,
        observations_narrative=observation_narrative,
        immunizations_narrative=immunization_narrative,
        medications_narrative=medication_narrative,
        procedures_narrative=procedure_narrative,
        careplans_narrative=careplan_narrative,
        conditions_narrative=condition_narrative,
        devices_narrative=device_narrative,
        imaging_studies_narrative=imaging_study_narrative,
    )

    print("\n" + "=" * 80)
    print("ENCOUNTER NARRATIVE:")
    print("=" * 80)
    print(encounter_narrative)
    print("=" * 80)

    # generate documents list
    print("\nGenerating list of documents likely created during the encounter...")
    documents_list = generate_documents_list(encounter_narration=encounter_narrative)
    print("\nDocuments List:")
    for doc in documents_list:
        print(f"  - Type: {doc['type']}, Description: {doc['description']}")


if __name__ == "__main__":
    main()
