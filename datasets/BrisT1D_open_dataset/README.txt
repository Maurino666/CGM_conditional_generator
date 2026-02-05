# BrisT1D-Open Dataset

## Overview
The BrisT1D Dataset features the data collected in a longitudinal study in which 24 young adults with Type 1 Diabetes (T1D) in the UK were given smartwatches to use alongside their T1D self-management. During the six-month study, participants donated the data collected by their T1D devices and smartwatches and were involved in monthly interviews or focus groups. The anonymised transcripts of these study sessions are included along with the device data, in both an anonymised raw state and a processed state.

This is the Open-Access part of the BrisT1D Dataset. It includes the processed state device data, study forms, transcripts, and demographic data. The raw state device data can be found in the BrisT1D-Restricted Dataset, available through data.bris (University of Bristol Data Repository) under Restricted Access.

**Creators:** Sam Gordon James (0000-0002-9238-1693), Miranda E. G. Armstrong (0000-0001-8946-5776), Aisling A. O'Kane (0000-0001-8219-8126), Harry Emerson (0000-0002-5829-0261), and Zahraa S. Abdallah (0000-0002-1291-2918)


## Data Structure

### Folder Structure:
brist1d_dataset/
├── device_data/
│   └── processed_state/
│       ├── P01.csv
│       ├── P02.csv
│       ├── ...
│       └── P24.csv
├── study_forms/
│   ├── consent_form.pdf
│   └── participant_information_sheet.pdf
├── transcripts/
│   ├── 0_introductory_interviews/
│   │   ├── 0_Int_P01.pdf
│   │   └── ...
│   ├── 1_july_interviews/
│   │   ├── 1_Int_P01.pdf
│   │   └── ...
│   ├── 2_august_focus_groups/
│   │   ├── 2_FG_FG1.pdf
│   │   └── ...
│   ├── 3_september_focus_groups/
│   │   ├── 3_FG_FG1.pdf
│   │   └── ...
│   ├── 4_october_focus_groups/
│   │   ├── 4_FG_FG1.pdf
│   │   └── ...
│   ├── 5_novemeber_focus_groups/
│   │   ├── 5_FG_FG1.pdf
│   │   └── ...
│   └── 6_decemeber_interviews/
│       ├── 6_Int_P01.pdf
│       └── ...
├── demographic_data.csv
├── LICENCE.txt
└── README.txt


### File Formats:
- `device_data/processed_data/PXX.csv` (Processed device data from participant XX)
- `consent_form.pdf` (Blank version of consent form, converted to online format and signed by participants)
- `participant_information_sheet.pdf` (Participant informaiton sheet provided to participants)
- `X_Int_PXX.pdf` (Anonymised interview transcipt e.g. 0_Int_P01.csv is the interview with P01 from study round 0)
- `X_FG_FGX.pdf` (Anonymised focus group transcipt e.g. 2_FG_FG1 is the first focus group from study round 2)
- `demographic_data.csv` (Demographic informaiton from the particpants collected in open text boxes)
- `LICENCE.txt` (Dataset licence details)
- `README.txt` (This file)


### Processed Data Schema (`PXX.csv`) 
| Column Name | Data Type  | Unit    | Description                                                                                                                           |  
|-------------|------------|---------|---------------------------------------------------------------------------------------------------------------------------------------|  
| `timestamp` | `datetime` | -       | The time and date the reading was taken. For some variables, this corresponds to the end of the interval the data is aggregated over. |  
| `bg`        | `float`    | mmol/L  | Blood glucose level recorded by the continuous glucose monitor.                                                                       |  
| `insulin`   | `float`    | units   | Total insulin dose received in the previous five minutes from the insulin pump.                                                       |  
| `carbs`     | `float`    | g       | Carbohydrate intake recorded by the user in the insulin pump or reader.                                                               |  
| `hr`        | `integer`  | bpm     | Mean heart rate for the previous five minutes as recorded by the smartwatch.                                                          |  
| `dist`      | `float`    | m       | Total distance travelled in the previous five minutes as recorded by the smartwatch.                                                  |  
| `steps`     | `integer`  | count   | Total steps taken in the previous five minutes as recorded by the smartwatch.                                                         |  
| `cals`      | `float`    | kcal    | Total calories burned in the previous five minutes as recorded by the smartwatch.                                                     |  
| `activity`  | `string`   | -       | Labelled activity events, declared by the user.                                                                                       |  
| `device`    | `string`   | -       | Name of the device that was used to collect the data.                                                                                 |  


## Ethics & Licence
Ethical approval for the study was recived from the University of Bristol Engineering Faculty Research Ethics Committee (Ref: 13065).

The participant information sheet and consent form can be found in the `study_forms/` directory.

Licence: Creative Commons Attribution 4.0 (see `LICENCE.txt` for details)