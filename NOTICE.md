# Internal sharing notice

This package was extracted from the private repository
`panhainan2020/VITAL_Lab-Omni` at commit
`8662fc729fb2a7becc63c4275b236371d346ba59`.

Included source lineage:

- `deploy/omni_29dof/t12_serial.py` -> `src/t12_remote/serial.py`
- `deploy/omni_29dof/t12_mapping_detector.py` ->
  `src/t12_remote/mapping_detector.py`
- generic tests and a sanitized configuration template derived from the same
  source tree

Explicitly excluded:

- Omni policy-selection and motor-control implementations
- model and policy binaries
- robot-specific udev serial numbers
- the measured lab calibration report and machine-specific serial path
- vendor Omni SDK, joystick, and remote-controller copies
- passwords, control tokens, private keys, and robot addresses

The owner authorized public repository visibility on 2026-10-16. No
open-source license has been selected for this extracted unit. Public
availability does not by itself grant permission to copy, modify, or
redistribute the code. Confirm the owner's sharing terms and third-party
obligations before reuse.

