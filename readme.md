# Code and data for
---
*Step-by-step toward understanding ultranonlocality: Wavevector and frequency dependence of exchange-correlation model kernels*

## Maintainers:
Aaron Kaplan (kaplan@temple.edu) and Niraj Nepal
---

### Directories:

**code** contains all analysis code needed for VASP. To extract the charge density from CHGCAR,
  https://gitlab.com/dhamil/vasp-utilities
was used with options:
*python3 main.py -dens -full*

**metal_data** contains jellium data and pseudopotential perturbative data for Na and Al

**Si** contains VASP data for Si. For copyright reasons, POTCAR files cannot be included. The pseudopotential used was
*PAW_PBE Si 05Jan2001*
