FRAMED
======

This module implements a complete 2D finite element solver for frame structures.
The solver uses Euler-Bernoulli beam elements with 3 degrees of freedom per node:
- u: horizontal displacement
- v: vertical displacement
- θ: rotation

Key Features:
- 2D frame elements with bending and axial stiffness
- Proper coordinate transformation from local to global coordinates
- Sparse matrix assembly and solving
- Visualization of original and deformed configurations
- Multiple load case analysis
- Linear elastic material behavior
- Penalty method for boundary conditions

Implementation Highlights:
- Element stiffness matrices based on beam theory
- Global assembly using standard finite element procedures
- Efficient sparse matrix solving for large systems
- Comprehensive result visualization and analysis
- Verification of linear elastic behavior through parametric studies

Example Usage:
    python3 framed.py

This will analyze a portal frame under lateral loading and display:
- Original and deformed frame configurations
- Displacement results at key nodes
- Load-displacement relationships
- Structural assessment against serviceability limits
