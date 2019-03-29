# ns-struc

Neutron-star structure code, including TOV, slow-rotation and tidal deformation equations.

---

### Scripts

###### NS properties (M-R relation, M-Lambda relation, etc.)

* getnsprops eos.csv

###### NS stability (extract stable branches of M-rhoc relation)

* splitbranches macro-eos.csv

###### Macroscopic NS observables linked to EoS (Mmax, R1.4, etc.)

* getmacro macro-eos.csv
* makemacro ./in/dir/ ./out/dir/ 50 (batch version direct from EoS)

---

### Tools

###### NS properties plot (M-R relation, M-Lambda relation, etc.)

* plotprops macro-eos.csv
* compareprops macro-eos1.csv,macro-eos2.csv (show fractional difference)

###### Macroscopic NS observables plot (Mmax, R1.4, etc.)

* comparemacro macro1.csv,macro2.csv (show fractional difference)

