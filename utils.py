from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd


def hello_world():
    return "Hello, world!"


class Date(BaseModel):
    # Somehow LLM is bad at specifying `datetime.datetime`
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
    # Required fields
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

    # Optional fields (can be empty/null in CSV)
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

    # Numeric fields
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
    # Required fields
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

    # Numeric fields
    base_encounter_cost: float = Field(
        description="Baseline assessed cost for the encounter before claim adjustments or payer calculations"
    )
    total_claim_cost: float = Field(
        description="Final total cost submitted on the claim after adjustments (patient + payer portions combined)"
    )
    payer_coverage: float = Field(
        description="Monetary amount covered by the payer for this encounter; total_claim_cost - payer_coverage approximates patient or other responsibility"
    )

    # Optional fields
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


def parse_date(date_str) -> Optional[Date]:
    """Parse a date string or NaN value into a Date object or None."""
    if pd.isna(date_str) or date_str == "":
        return None
    # Parse date string in format YYYY-MM-DD
    year, month, day = date_str.split("-")
    return Date(year=int(year), month=int(month), day=int(day))


def parse_datetime(datetime_str) -> Optional[DateTime]:
    """Parse a datetime string or NaN value into a DateTime object or None."""
    if pd.isna(datetime_str) or datetime_str == "":
        return None
    # Parse datetime string in format YYYY-MM-DDTHH:MM:SSZ
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
