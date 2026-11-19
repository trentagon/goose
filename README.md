```
                                                        _...--.
                                        _____......----'     .'
                                  _..-''                   .'
                                .'                       ./
                        _.--._.'                       .' |
                     .-'                           .-.'  /
                   .'   _.-.                     .  \   '
                 .'  .'   .'    _    .-.        / `./  :
               .'  .'   .'  .--' `.  |  \  |`. |     .'
            _.'  .'   .' `.'       `-'   \ / |.'   .'
         _.'  .-'   .'     `-.            `      .'
       .'   .'    .'          `-.._ _ _ _ .-.    :
      /     o _.-'               .--'   .'   \   |
    .'-.__..-'                  /..    .`    / .'
  .'   . '                       /.'/.'     /  |
 `---'                                   _.'   '
                                       /.'    .'
                                        /.'/.'
 ,adPPYb,d8  ,adPPYba,   ,adPPYba,  ,adPPYba,  ,adPPYba,  
a8"    `Y88 a8"     "8a a8"     "8a I8[    "" a8P_____88  
8b       88 8b       d8 8b       d8  `"Y8ba,  8PP"""""""  
"8a,   ,d88 "8a,   ,a8" "8a,   ,a8" aa    ]8I "8b,   ,aa  
 `"YbbdP"Y8  `"YbbdP"'   `"YbbdP"'  `"YbbdP"'  `"Ybbd8"'  
 aa,    ,88                                               
  "Y8bbdP"    
```
A model of the [G]eologic [C]arbon [C]ycle during [S]nowball [E]arth (GOOSE).
Developed by Trent Thomas and David Catling.

This is a directory of python codes that track how the carbon and alkalinity in the Earth system evolve over time according to the geologic carbon cycle during global glaciation events in Earth's history known as "snowball Earth" events. A full model description is found in the associated manuscript:

"Three-stage Formation of Cap Carbonates after Marinoan Snowball Glaciation Consistent with Depositional Timescales and Geochemistry" by Trent B. Thomas and David C. Catling. Nature Communications.

As a matter of courtesy, we request that people using this code please cite the above manuscript. In the interest of an "open source" approach, we also request that authors who use and modify the code, please send a copy of papers and modified code to the corresponding author (Trent Thomas; tbthomas@uw.edu).

This code requires only some standard python libraries to run and was developed in python 3. There are 2 jupyter notebooks designed to act as a tutorial to the code. We recommend starting with "TutorialSolo".

## Bug fixes (November 2026)

The original GOOSE code was written by hand before publication in 2024. After publication, we analyzed the code with an LLM and found two implementation errors. The code at the top level of this repository now has fixes for these errors. The fixes do not add new physics. They make the code calculate the model equations correctly.

* Bug 1 was related to calcium conservation as the ocean volume changes when the ice sheet forms and melts.
* Bug 2 was related to a unit conversion in the seafloor weathering parameterization.

Further tests show that these bugs did not significantly alter the conclusions from the 2024 Nature Communications paper or the 2026 Geology paper, but future work should incorporate the fixes for accuracy. The Geology paper used a modified version of this code, archived at Zenodo (https://doi.org/10.5281/zenodo.17567190).

The previous, uncorrected version of the code is in `Legacy_NatComms2024` with the same files and structure. Use this version to reproduce the published results.
