## allergies

```json
{
  "START": "1982-10-25",
  "STOP": NaN,
  "PATIENT": "76982e06-f8b8-4509-9ca3-65a99c8650fe",
  "ENCOUNTER": "b896bf40-8b72-42b7-b205-142ee3a56b55",
  "CODE": 300916003,
  "DESCRIPTION": "Latex allergy"
}
```

## careplans

```json
{
  "Id": "d2500b8c-e830-433a-8b9d-368d30741520",
  "START": "2010-01-23",
  "STOP": "2012-01-23",
  "PATIENT": "034e9e3b-2def-4559-bb2a-7850888ae060",
  "ENCOUNTER": "d0c40d10-8d87-447e-836e-99d26ad52ea5",
  "CODE": 53950000,
  "DESCRIPTION": "Respiratory therapy",
  "REASONCODE": 10509002.0,
  "REASONDESCRIPTION": "Acute bronchitis (disorder)"
}
```

## conditions

```json
{
  "START": "2001-05-01",
  "STOP": NaN,
  "PATIENT": "1d604da9-9a81-4ba9-80c2-de3375d59b40",
  "ENCOUNTER": "8f104aa7-4ca9-4473-885a-bba2437df588",
  "CODE": 40055000,
  "DESCRIPTION": "Chronic sinusitis (disorder)"
}
```

## devices

```json
{
  "START": "2001-07-04T08:42:44Z",
  "STOP": NaN,
  "PATIENT": "d49f748f-928d-40e8-92c8-73e4c5679711",
  "ENCOUNTER": "2500b8bd-dc98-44ef-a252-22dc4f81d61b",
  "CODE": 72506001,
  "DESCRIPTION": "Implantable defibrillator  device (physical object)",
  "UDI": "(01)67677988606464(11)010613(17)260628(10)2882441934(21)7849600052"
}
```

## encounters

```json
{
  "Id": "d0c40d10-8d87-447e-836e-99d26ad52ea5",
  "START": "2010-01-23T17:45:28Z",
  "STOP": "2010-01-23T18:10:28Z",
  "PATIENT": "034e9e3b-2def-4559-bb2a-7850888ae060",
  "ORGANIZATION": "e002090d-4e92-300e-b41e-7d1f21dee4c6",
  "PROVIDER": "e6283e46-fd81-3611-9459-0edb1c3da357",
  "PAYER": "6e2f1a2d-27bd-3701-8d08-dae202c58632",
  "ENCOUNTERCLASS": "ambulatory",
  "CODE": 185345009,
  "DESCRIPTION": "Encounter for symptom",
  "BASE_ENCOUNTER_COST": 129.16,
  "TOTAL_CLAIM_COST": 129.16,
  "PAYER_COVERAGE": 54.16,
  "REASONCODE": 10509002.0,
  "REASONDESCRIPTION": "Acute bronchitis (disorder)"
}
```

## imaging_studies

```json
{
  "Id": "d3e49b38-7634-4416-879d-7bc68bf3e7df",
  "DATE": "2014-07-08T15:35:36Z",
  "PATIENT": "b58731cc-2d8b-4c2d-b327-4cab771af3ef",
  "ENCOUNTER": "3a36836d-da25-4e73-808b-972b669b7e4e",
  "BODYSITE_CODE": 40983000,
  "BODYSITE_DESCRIPTION": "Arm",
  "MODALITY_CODE": "DX",
  "MODALITY_DESCRIPTION": "Digital Radiography",
  "SOP_CODE": "1.2.840.10008.5.1.4.1.1.1.1",
  "SOP_DESCRIPTION": "Digital X-Ray Image Storage"
}
```

## immunizations

```json
{
  "DATE": "2010-07-27T12:58:08Z",
  "PATIENT": "10339b10-3cd1-4ac3-ac13-ec26728cb592",
  "ENCOUNTER": "dae2b7cb-1316-4b78-954f-fa610a6c6d0e",
  "CODE": 140,
  "DESCRIPTION": "Influenza  seasonal  injectable  preservative free",
  "BASE_COST": 140.52
}
```

## medications

```json
{
  "START": "2010-05-05T00:26:23Z",
  "STOP": "2011-04-30T00:26:23Z",
  "PATIENT": "8d4c4326-e9de-4f45-9a4c-f8c36bff89ae",
  "PAYER": "b1c428d6-4f07-31e0-90f0-68ffa6ff8c76",
  "ENCOUNTER": "1e0d6b0e-1711-4a25-99f9-b1c700c9b260",
  "CODE": 389221,
  "DESCRIPTION": "Etonogestrel 68 MG Drug Implant",
  "BASE_COST": 677.08,
  "PAYER_COVERAGE": 0.0,
  "DISPENSES": 12,
  "TOTALCOST": 8124.96,
  "REASONCODE": NaN,
  "REASONDESCRIPTION": NaN
}
```

## observations

```json
{
  "DATE": "2012-01-23T17:45:28Z",
  "PATIENT": "034e9e3b-2def-4559-bb2a-7850888ae060",
  "ENCOUNTER": "e88bc3a9-007c-405e-aabc-792a38f4aa2b",
  "CODE": "8302-2",
  "DESCRIPTION": "Body Height",
  "VALUE": "193.3",
  "UNITS": "cm",
  "TYPE": "numeric"
}
```

## organizations

```json
{
  "Id": "ef58ea08-d883-3957-8300-150554edc8fb",
  "NAME": "HEALTHALLIANCE HOSPITALS  INC",
  "ADDRESS": "60 HOSPITAL ROAD",
  "CITY": "LEOMINSTER",
  "STATE": "MA",
  "ZIP": "01453",
  "LAT": 42.520838,
  "LON": -71.770876,
  "PHONE": "9784662000",
  "REVENUE": 198002.2800000044,
  "UTILIZATION": 1557
}
```

## patients

```json
{
  "Id": "1d604da9-9a81-4ba9-80c2-de3375d59b40",
  "BIRTHDATE": "1989-05-25",
  "DEATHDATE": NaN,
  "SSN": "999-76-6866",
  "DRIVERS": "S99984236",
  "PASSPORT": "X19277260X",
  "PREFIX": "Mr.",
  "FIRST": "José Eduardo181",
  "LAST": "Gómez206",
  "SUFFIX": NaN,
  "MAIDEN": NaN,
  "MARITAL": "M",
  "RACE": "white",
  "ETHNICITY": "hispanic",
  "GENDER": "M",
  "BIRTHPLACE": "Marigot  Saint Andrew Parish  DM",
  "ADDRESS": "427 Balistreri Way Unit 19",
  "CITY": "Chicopee",
  "STATE": "Massachusetts",
  "COUNTY": "Hampden County",
  "ZIP": 1013.0,
  "LAT": 42.22835382315942,
  "LON": -72.56295055096882,
  "HEALTHCARE_EXPENSES": 271227.08,
  "HEALTHCARE_COVERAGE": 1334.88
}
```

## payers

```json
{
  "Id": "b3221cfc-24fb-339e-823d-bc4136cbc4ed",
  "NAME": "Dual Eligible",
  "ADDRESS": "7500 Security Blvd",
  "CITY": "Baltimore",
  "STATE_HEADQUARTERED": "MD",
  "ZIP": 21244.0,
  "PHONE": "1-877-267-2323",
  "AMOUNT_COVERED": 141676.87,
  "AMOUNT_UNCOVERED": 119449.83,
  "REVENUE": 1305000.0,
  "COVERED_ENCOUNTERS": 907,
  "UNCOVERED_ENCOUNTERS": 0,
  "COVERED_MEDICATIONS": 556,
  "UNCOVERED_MEDICATIONS": 0,
  "COVERED_PROCEDURES": 280,
  "UNCOVERED_PROCEDURES": 0,
  "COVERED_IMMUNIZATIONS": 223,
  "UNCOVERED_IMMUNIZATIONS": 0,
  "UNIQUE_CUSTOMERS": 25,
  "QOLS_AVG": 0.3628096741903461,
  "MEMBER_MONTHS": 3348
}
```

## payer_transitions

```json
{
  "PATIENT": "1d604da9-9a81-4ba9-80c2-de3375d59b40",
  "START_YEAR": 1989,
  "END_YEAR": 1998,
  "PAYER": "b1c428d6-4f07-31e0-90f0-68ffa6ff8c76",
  "OWNERSHIP": "Guardian"
}
```

## procedures

```json
{
  "DATE": "2011-04-30T00:26:23Z",
  "PATIENT": "8d4c4326-e9de-4f45-9a4c-f8c36bff89ae",
  "ENCOUNTER": "6aa37300-d1b4-48e7-a2f8-5e0f70f48f38",
  "CODE": 169553002,
  "DESCRIPTION": "Insertion of subcutaneous contraceptive",
  "BASE_COST": 14896.56,
  "REASONCODE": NaN,
  "REASONDESCRIPTION": NaN
}
```

## providers

```json
{
  "Id": "3421aa75-dec7-378d-a9e0-0bc764e4cb0d",
  "ORGANIZATION": "ef58ea08-d883-3957-8300-150554edc8fb",
  "NAME": "Tomas436 Sauer652",
  "GENDER": "M",
  "SPECIALITY": "GENERAL PRACTICE",
  "ADDRESS": "60 HOSPITAL ROAD",
  "CITY": "LEOMINSTER",
  "STATE": "MA",
  "ZIP": "01453",
  "LAT": 42.520838,
  "LON": -71.770876,
  "UTILIZATION": 1557
}
```

