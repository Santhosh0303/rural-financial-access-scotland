# Dissertation Development Decisions Log

## Purpose of this file
This file records the key development decisions I have locked for my dissertation project. It is being maintained so that the coding phase follows one clear structure and so that important technical decisions are documented consistently.



## Locked project decisions
I have locked the following decisions for the development phase:

- My dissertation will be developed in two versions.
- Version 1 will be the validated current-state backbone.
- Version 2 will be the temporal extension using historical bank-closure data.
- The final reporting geography for the project will be Data Zone 2022.
- The support/input geography for population and SIMD data will be Data Zone 2011.
- No final zone-level analytical output will be treated as valid until the harmonisation step has been completed and checked.
- One-zone-one-row logic must be preserved in every final zone-level output.
- Dashboard outputs will only be created from validated analytical tables.
- The machine-learning stage must remain interpretable and carefully explained.
- Historical bank-closure work will use Geolytix as the primary source, with Which? used for validation and cross-checking.
- Version 2 temporal logic will only begin after Version 1 has passed quality checks.

## Locked project structure
I will follow the folder structure already created in my dissertation project directory. The active structure for development is:

- data/raw
- data/interim
- data/processed
- docs
- notebooks
- outputs/dashboard
- outputs/figures
- outputs/tables
- src
- config

## Locked raw-data structure
Inside `data/raw`, I will use the following folder names exactly as saved in the project:

- bank_closures
- dz_boundaries_2011
- dz_boundaries_2022
- population
- post_office
- rurality
- simd

## Current setup status
At this stage, the development environment has been prepared as follows:

- VS Code installed
- Python working correctly
- Virtual environment created for the project
- Project interpreter selected in VS Code
- Core Python packages installed successfully

## Development rule
During coding, I will complete and validate one layer before moving to the next. The development order I will follow is:

geography -> harmonisation -> context -> current services -> accessibility -> labels -> machine learning -> scenario simulation -> dashboard outputs -> temporal extension

## Note
This file will be updated only when a development decision is formally changed, so that the project remains consistent throughout the dissertation build.
