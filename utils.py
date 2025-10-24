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


class PatientProfile(BaseModel):
    # Required fields
    id: str = Field(description="Unique patient identifier (UUID)")
    birthdate: Date
    ssn: str = Field(description="Patient's social security number")
    first: str = Field(description="Patient's first name")
    last: str = Field(description="Patient's last name")
    marital: str = Field(description="Marital status (M=Married, S=Single)")
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

    # Numeric fields
    lat: float = Field(description="Patient's address latitude")
    lon: float = Field(description="Patient's address longitude")
    healthcare_expenses: float = Field(description="Total healthcare expenses")
    healthcare_coverage: float = Field(description="Total healthcare coverage amount")


def parse_date(date_str) -> Optional[Date]:
    """Parse a date string or NaN value into a Date object or None."""
    if pd.isna(date_str) or date_str == "":
        return None
    # Parse date string in format YYYY-MM-DD
    year, month, day = date_str.split("-")
    return Date(year=int(year), month=int(month), day=int(day))


def row_to_patient_profile(row) -> PatientProfile:
    """Convert a DataFrame row to a PatientProfile object."""
    return PatientProfile(
        id=row["Id"],
        birthdate=parse_date(row["BIRTHDATE"]),
        ssn=row["SSN"],
        first=row["FIRST"],
        last=row["LAST"],
        marital=row["MARITAL"],
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
