"""
Finite Element Frame Analysis Implementation
===========================================

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
    python3 main.py

This will analyze a portal frame under lateral loading and display:
- Original and deformed frame configurations
- Displacement results at key nodes
- Load-displacement relationships
- Structural assessment against serviceability limits

Author: Finite Element Implementation
Date: September 15, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================================
# PROBLEM DEFINITION: Portal Frame Structure
# =============================================================================

# Define the geometry of a simple portal frame with two vertical legs 
# connected by a horizontal beam at the top
frame = {
    "legs": [
        {"x": 0, "y": 0, "length": 5},  # Left leg: 5m tall at x=0
        {"x": 1, "y": 0, "length": 5}   # Right leg: 5m tall at x=1
    ],
    "beam": {
        "x1": 0, "y1": 5,  # Beam start point (top of left leg)
        "x2": 1, "y2": 5   # Beam end point (top of right leg)
    }
}

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_frame(frame, displacements=None, scale_factor=1.0):
    """
    Plot the frame structure showing original and optionally deformed configurations.
    
    Parameters:
    -----------
    frame : dict
        Frame geometry definition
    displacements : array_like, optional
        Nodal displacement vector [u1, v1, θ1, u2, v2, θ2, ...]
    scale_factor : float
        Scaling factor for displacement visualization (default: 1.0)
        
    Notes:
    ------
    - Blue solid lines: Original frame configuration
    - Red dashed lines: Deformed frame configuration (if displacements provided)
    - Green arrows: Displacement vectors at nodes
    - Node numbering and connectivity determined by discretize_frame()
    """
    plt.figure(figsize=(12, 8))
    
    # Get nodes and elements for plotting
    nodes, elements = discretize_frame(frame)
    
    # Plot original structure elements
    for element in elements:
        node1_idx, node2_idx = element
        x_coords = [nodes[node1_idx][0], nodes[node2_idx][0]]
        y_coords = [nodes[node1_idx][1], nodes[node2_idx][1]]
        plt.plot(x_coords, y_coords, 'b-', linewidth=3, alpha=0.7, label='Original' if element[0] == 0 else "")
    
    # Plot original nodes
    plt.scatter(nodes[:, 0], nodes[:, 1], c='blue', s=50, zorder=5, label='Original nodes')
    
    # Plot deformed structure if displacements are provided
    if displacements is not None:
        # Extract only translational displacements (u, v) ignoring rotations (θ)
        # Each node has 3 DOF: [u, v, θ], so we extract every 3rd and 3rd+1 values
        u_displacements = np.zeros((len(nodes), 2))
        for i in range(len(nodes)):
            u_displacements[i, 0] = displacements[3*i]     # u displacement (horizontal)
            u_displacements[i, 1] = displacements[3*i+1]   # v displacement (vertical)
        
        # Calculate deformed node positions
        deformed_nodes = nodes + u_displacements * scale_factor
        
        # Plot deformed elements
        for element in elements:
            node1_idx, node2_idx = element
            x_coords = [deformed_nodes[node1_idx][0], deformed_nodes[node2_idx][0]]
            y_coords = [deformed_nodes[node1_idx][1], deformed_nodes[node2_idx][1]]
            plt.plot(x_coords, y_coords, 'r--', linewidth=3, alpha=0.7, label='Deformed' if element[0] == 0 else "")
        
        # Plot deformed nodes
        plt.scatter(deformed_nodes[:, 0], deformed_nodes[:, 1], c='red', s=50, zorder=5, label='Deformed nodes')
        
        # Add displacement vectors for significant displacements
        for i in range(len(nodes)):
            dx = u_displacements[i, 0] * scale_factor
            dy = u_displacements[i, 1] * scale_factor
            if abs(dx) > 1e-8 or abs(dy) > 1e-8:  # Only plot significant displacements
                plt.arrow(nodes[i][0], nodes[i][1], dx, dy, 
                         head_width=0.03, head_length=0.03, fc='green', ec='green', alpha=0.7)
    
    # Set plot properties
    plt.xlim(-0.3, 1.3)
    plt.ylim(-0.5, 6)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Frame Structure{' - Deformed vs Original' if displacements is not None else ''}")
    plt.xlabel("X-axis (m)")
    plt.ylabel("Y-axis (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add scale factor info if deformed
    if displacements is not None and scale_factor != 1.0:
        plt.text(0.02, 0.98, f'Deformation scale: {scale_factor}x', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MESH GENERATION AND DISCRETIZATION
# =============================================================================

def discretize_frame(frame, num_elements_per_leg=5, num_elements_beam=5):
    """
    Discretize the frame structure into finite elements and generate node coordinates.
    
    This function creates a mesh of the portal frame by:
    1. Creating nodes along each leg from bottom to top
    2. Creating internal nodes along the beam (excluding endpoints)
    3. Defining element connectivity between adjacent nodes
    
    Parameters:
    -----------
    frame : dict
        Frame geometry definition containing legs and beam specifications
    num_elements_per_leg : int
        Number of elements per leg (default: 5)
    num_elements_beam : int  
        Number of elements for the beam (default: 5)
        
    Returns:
    --------
    nodes : ndarray, shape (n_nodes, 2)
        Node coordinates [x, y] for each node
    elements : ndarray, shape (n_elements, 2)
        Element connectivity [node1_id, node2_id] for each element
        
    Node Numbering Scheme:
    ----------------------
    - Nodes 0-5: Left leg (bottom to top)
    - Nodes 6-11: Right leg (bottom to top)  
    - Nodes 12-15: Internal beam nodes (left to right)
    
    Element Connectivity:
    ---------------------
    - Elements 0-4: Left leg elements
    - Elements 5-9: Right leg elements
    - Elements 10-14: Beam elements (including connections to leg tops)
    """
    nodes = []
    
    # Create nodes for left leg (from bottom to top) - nodes 0 to 5
    y_coords = np.linspace(0, 5, num_elements_per_leg + 1)
    for y in y_coords:
        nodes.append([0.0, y])
    
    # Create nodes for right leg (from bottom to top) - nodes 6 to 11
    for y in y_coords:
        nodes.append([1.0, y])
    
    # Create internal beam nodes (excluding endpoints which are already leg nodes) - nodes 12 to 15
    # This ensures proper connectivity between legs and beam
    x_coords = np.linspace(0, 1, num_elements_beam + 1)
    for i in range(1, num_elements_beam):  # Skip endpoints (0 and 1)
        nodes.append([x_coords[i], 5.0])
    
    nodes = np.array(nodes)
    elements = []
    
    # Elements for left leg (nodes 0-5)
    # Creates 5 elements connecting consecutive nodes along the left leg
    for i in range(num_elements_per_leg):
        elements.append([i, i + 1])
    
    # Elements for right leg (nodes 6-11)
    # Creates 5 elements connecting consecutive nodes along the right leg  
    for i in range(num_elements_per_leg):
        elements.append([6 + i, 6 + i + 1])
    
    # Elements for beam - connect top of left leg to internal nodes to top of right leg
    # This creates a continuous beam across the top of the frame
    
    # Element from top of left leg (node 5) to first internal beam node (node 12)
    elements.append([5, 12])
    
    # Internal beam elements connecting consecutive internal beam nodes
    for i in range(num_elements_beam - 2):
        elements.append([12 + i, 12 + i + 1])
    
    # Element from last internal beam node to top of right leg (node 11)
    elements.append([12 + num_elements_beam - 2, 11])
    
    return np.array(nodes), np.array(elements)

# =============================================================================
# LOADING AND BOUNDARY CONDITIONS
# =============================================================================

def apply_horizontal_load(frame, load_value):
    """
    Define a horizontal load to be applied at the center of the horizontal beam.
    
    This is a convenience function that was used in the original TODO structure.
    The actual load application is now handled directly in solve_frame().
    
    Parameters:
    -----------
    frame : dict
        Frame geometry definition
    load_value : float
        Magnitude of horizontal load (N)
        
    Returns:
    --------
    load : dict
        Load specification dictionary with position and magnitude
    """
    load = {
        "x": (frame["beam"]["x1"] + frame["beam"]["x2"]) / 2,  # Center of beam
        "y": frame["beam"]["y1"],                              # Beam height
        "magnitude": load_value,                               # Load magnitude
        "direction": "horizontal"                              # Load direction
    }
    return load

def define_boundary_conditions(frame):
    """
    Define fixed boundary conditions at the base of both legs.
    
    This is a convenience function that was used in the original TODO structure.
    The actual boundary condition application is now handled in solve_frame().
    
    Parameters:
    -----------
    frame : dict
        Frame geometry definition
        
    Returns:
    --------
    boundary_conditions : list
        List of boundary condition specifications
    """
    boundary_conditions = []
    for leg in frame["legs"]:
        boundary_conditions.append({
            "x": leg["x"],
            "y": leg["y"], 
            "type": "fixed"  # Fixed in all DOF (u, v, θ)
        })
    return boundary_conditions

# =============================================================================
# FINITE ELEMENT FORMULATION
# =============================================================================

def element_stiffness_matrix_frame(node1, node2, E, A, I):
    """
    Calculate the element stiffness matrix for a 2D Euler-Bernoulli frame element.
    
    This function implements the standard 2D beam element formulation with:
    - Axial stiffness (EA/L terms)
    - Bending stiffness (EI/L³ terms)  
    - Proper coupling between axial and bending behavior
    
    The element has 6 DOF total (3 per node):
    - Node 1: u₁, v₁, θ₁ (horizontal disp, vertical disp, rotation)
    - Node 2: u₂, v₂, θ₂ (horizontal disp, vertical disp, rotation)
    
    Parameters:
    -----------
    node1, node2 : array_like
        Node coordinates [x, y] for element endpoints
    E : float
        Young's modulus (Pa)
    A : float
        Cross-sectional area (m²)
    I : float
        Second moment of area (m⁴)
        
    Returns:
    --------
    k_global : ndarray, shape (6, 6)
        Element stiffness matrix in global coordinates
        
    Theory:
    -------
    The local stiffness matrix is derived from Euler-Bernoulli beam theory:
    - Axial behavior: u = f(x), governed by EA
    - Bending behavior: v = f(x), w = f(x), governed by EI
    - Coordinate transformation handles arbitrary element orientation
    """
    # Calculate element geometry
    dx = node2[0] - node1[0]  # Element projection in x-direction
    dy = node2[1] - node1[1]  # Element projection in y-direction  
    L = np.sqrt(dx**2 + dy**2)  # Element length
    
    # Direction cosines for coordinate transformation
    c = dx / L  # cos(θ) where θ is element orientation angle
    s = dy / L  # sin(θ) where θ is element orientation angle
    
    # Local stiffness matrix coefficients
    EI_L3 = E * I / (L**3)    # Bending stiffness coefficient
    EA_L = E * A / L          # Axial stiffness coefficient
    
    # Local stiffness matrix for frame element (6x6)
    # Rows/cols: [u₁, v₁, θ₁, u₂, v₂, θ₂] in local coordinates
    # 
    # Matrix structure:
    # [  EA/L    0       0      -EA/L    0       0    ]  u₁
    # [   0   12EI/L³  6EI/L²    0   -12EI/L³ 6EI/L² ]  v₁  
    # [   0   6EI/L²   4EI/L     0   -6EI/L²  2EI/L  ]  θ₁
    # [ -EA/L    0       0       EA/L    0       0    ]  u₂
    # [   0  -12EI/L³ -6EI/L²    0    12EI/L³ -6EI/L²]  v₂
    # [   0   6EI/L²   2EI/L     0   -6EI/L²  4EI/L  ]  θ₂
    k_local = np.array([
        [EA_L,        0,         0,    -EA_L,        0,         0],
        [0,      12*EI_L3,  6*EI_L3*L,    0,   -12*EI_L3,  6*EI_L3*L],
        [0,      6*EI_L3*L, 4*E*I/L,      0,   -6*EI_L3*L, 2*E*I/L],
        [-EA_L,       0,         0,     EA_L,        0,         0],
        [0,     -12*EI_L3, -6*EI_L3*L,    0,    12*EI_L3, -6*EI_L3*L],
        [0,      6*EI_L3*L, 2*E*I/L,      0,   -6*EI_L3*L, 4*E*I/L]
    ])
    
    # Transformation matrix from local to global coordinates
    # Transforms [u_local, v_local, θ_local] to [u_global, v_global, θ_global]
    #
    # For 2D frame elements:
    # u_global = u_local * cos(θ) - v_local * sin(θ)
    # v_global = u_local * sin(θ) + v_local * cos(θ)  
    # θ_global = θ_local (rotation is invariant)
    T = np.array([
        [c,  s,  0,  0,  0,  0],   # u₁ transformation
        [-s, c,  0,  0,  0,  0],   # v₁ transformation  
        [0,  0,  1,  0,  0,  0],   # θ₁ transformation (no change)
        [0,  0,  0,  c,  s,  0],   # u₂ transformation
        [0,  0,  0, -s,  c,  0],   # v₂ transformation
        [0,  0,  0,  0,  0,  1]    # θ₂ transformation (no change)
    ])
    
    # Transform local stiffness matrix to global coordinates
    # K_global = T^T * K_local * T
    k_global = T.T @ k_local @ T
    
    return k_global

def assemble_global_stiffness_frame(nodes, elements, E=200e9, A=0.01, I=1e-4):
    """
    Assemble the global stiffness matrix for the entire frame structure.
    
    This function performs the standard finite element assembly process:
    1. Loop through all elements in the mesh
    2. Calculate element stiffness matrix for each element  
    3. Add element contributions to appropriate locations in global matrix
    4. Return the assembled global stiffness matrix
    
    Parameters:
    -----------
    nodes : ndarray, shape (n_nodes, 2)
        Node coordinates [x, y] for each node
    elements : ndarray, shape (n_elements, 2)  
        Element connectivity [node1_id, node2_id] for each element
    E : float
        Young's modulus (Pa) - default: 200 GPa (typical for steel)
    A : float
        Cross-sectional area (m²) - default: 0.01 m² (10cm x 10cm)
    I : float
        Second moment of area (m⁴) - default: 1e-4 m⁴ (rectangular section)
        
    Returns:
    --------
    K_global : ndarray, shape (n_dof, n_dof)
        Global stiffness matrix where n_dof = 3 * n_nodes
        
    Notes:
    ------
    - Each node has 3 DOF: [u, v, θ] 
    - DOF ordering: [u₁, v₁, θ₁, u₂, v₂, θ₂, ..., uₙ, vₙ, θₙ]
    - Matrix is symmetric and positive semi-definite before boundary conditions
    - Assembly uses standard "scatter" operation from element to global DOF
    """
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes  # 3 DOF per node (u, v, theta)
    
    # Initialize global stiffness matrix as zeros
    K_global = np.zeros((n_dof, n_dof))
    
    # Loop through all elements and assemble their contributions
    for element in elements:
        # Get node indices for current element
        node1_idx, node2_idx = element
        node1 = nodes[node1_idx]  # [x₁, y₁]
        node2 = nodes[node2_idx]  # [x₂, y₂]
        
        # Calculate 6x6 element stiffness matrix in global coordinates
        k_elem = element_stiffness_matrix_frame(node1, node2, E, A, I)
        
        # Determine global DOF indices for this element's nodes
        # Node 1 DOF: [3*node1_idx, 3*node1_idx+1, 3*node1_idx+2] = [u₁, v₁, θ₁]
        # Node 2 DOF: [3*node2_idx, 3*node2_idx+1, 3*node2_idx+2] = [u₂, v₂, θ₂]
        dof_indices = [3*node1_idx, 3*node1_idx+1, 3*node1_idx+2, 
                      3*node2_idx, 3*node2_idx+1, 3*node2_idx+2]
        
        # Assemble element stiffness into global matrix
        # This is the standard "scatter" operation: K_global[I,J] += k_elem[i,j]
        for i in range(6):
            for j in range(6):
                K_global[dof_indices[i], dof_indices[j]] += k_elem[i, j]
    
    return K_global

def apply_boundary_conditions_frame(K, F, fixed_nodes):
    """
    Apply boundary conditions to the global system using the penalty method.
    
    This function implements fixed boundary conditions by constraining all DOF
    at specified nodes. The penalty method adds large values to diagonal terms
    corresponding to constrained DOF, effectively making those displacements zero.
    
    Parameters:
    -----------
    K : ndarray, shape (n_dof, n_dof)
        Global stiffness matrix before boundary conditions
    F : ndarray, shape (n_dof,)
        Global force vector before boundary conditions  
    fixed_nodes : list
        List of node indices that should be completely fixed (all 3 DOF)
        
    Returns:
    --------
    K_bc : ndarray, shape (n_dof, n_dof)
        Modified stiffness matrix with boundary conditions applied
    F_bc : ndarray, shape (n_dof,)
        Modified force vector with boundary conditions applied
        
    Method:
    -------
    Penalty Method:
    - Add large penalty value (1e12) to K[dof,dof] for each constrained DOF
    - Set F[dof] = 0 for each constrained DOF
    - This forces displacement at constrained DOF to be approximately zero
    - Alternative methods: elimination, Lagrange multipliers
    
    Example:
    --------
    For a fixed node at base: u = v = θ = 0
    K[3*node,   3*node]   += penalty    # u DOF  
    K[3*node+1, 3*node+1] += penalty    # v DOF
    K[3*node+2, 3*node+2] += penalty    # θ DOF
    """
    # Create copies to avoid modifying original matrices
    K_bc = K.copy()
    F_bc = F.copy()
    
    # Large penalty value - should be much larger than typical stiffness values
    # but not so large as to cause numerical issues (typically 1e6 to 1e15)
    penalty = 1e12
    
    # Apply penalty method to all DOF of each fixed node
    for node_idx in fixed_nodes:
        # Fix all 3 DOF for the node: u, v, θ
        for dof_offset in range(3):
            dof = 3 * node_idx + dof_offset  # Global DOF index
            
            # Add penalty to diagonal term (makes this DOF very stiff)
            K_bc[dof, dof] += penalty
            
            # Set force to zero (no external force on constrained DOF)
            F_bc[dof] = 0
    
    return K_bc, F_bc

# =============================================================================
# MAIN FINITE ELEMENT SOLVER
# =============================================================================

def solve_frame(frame, load_value, num_elements_per_leg=5, num_elements_beam=5):
    """
    Complete finite element solver for the frame structure under lateral loading.
    
    This function orchestrates the entire finite element analysis process:
    1. Discretize geometry into nodes and elements
    2. Assemble global stiffness matrix
    3. Apply loads and boundary conditions  
    4. Solve the linear system K*u = F
    5. Return solution and problem data
    
    Parameters:
    -----------
    frame : dict
        Frame geometry definition containing legs and beam specifications
    load_value : float
        Magnitude of horizontal load to apply at beam center (N)
    num_elements_per_leg : int
        Number of elements per leg (default: 5)
    num_elements_beam : int
        Number of elements for beam (default: 5)
        
    Returns:
    --------
    nodes : ndarray, shape (n_nodes, 2)
        Node coordinates [x, y]
    elements : ndarray, shape (n_elements, 2)  
        Element connectivity [node1_id, node2_id]
    displacements : ndarray, shape (n_dof,)
        Solution vector [u₁, v₁, θ₁, u₂, v₂, θ₂, ..., uₙ, vₙ, θₙ]
    F : ndarray, shape (n_dof,)
        Applied force vector
    load_node : int
        Index of node where load was applied
        
    Analysis Assumptions:
    --------------------
    - Linear elastic material behavior
    - Small deformation theory (geometric linearity)
    - Euler-Bernoulli beam theory (plane sections remain plane)
    - Uniform material properties throughout structure
    - Static loading (no dynamic effects)
    
    Material Properties:
    -------------------
    - E = 200 GPa (typical structural steel)
    - A = 0.01 m² (10cm × 10cm cross-section)  
    - I = 1e-4 m⁴ (second moment of area for rectangular section)
    """
    # Material properties for structural steel frame
    E = 200e9     # Young's modulus (Pa) - typical for steel
    A = 0.01      # Cross-sectional area (m²) - 10cm x 10cm beam
    I = 1e-4      # Second moment of area (m⁴) - for rectangular section
    
    # Step 1: Discretize frame geometry into finite element mesh
    nodes, elements = discretize_frame(frame, num_elements_per_leg, num_elements_beam)
    n_nodes = len(nodes)
    n_dof = 3 * n_nodes  # 3 DOF per node
    
    # Step 2: Assemble global stiffness matrix
    K = assemble_global_stiffness_frame(nodes, elements, E, A, I)
    
    # Step 3: Create global force vector and apply loads
    F = np.zeros(n_dof)  # Initialize force vector
    
    # Apply horizontal load at center of beam
    # Find node closest to beam center for load application
    beam_center_x = (frame["beam"]["x1"] + frame["beam"]["x2"]) / 2  # x = 0.5
    beam_y = frame["beam"]["y1"]  # y = 5.0
    
    # Calculate distances from all nodes to load application point
    distances = np.sqrt((nodes[:, 0] - beam_center_x)**2 + (nodes[:, 1] - beam_y)**2)
    load_node = np.argmin(distances)  # Node index closest to load point
    
    # Apply horizontal force in u-direction (DOF index = 3*node_index + 0)
    F[3*load_node] = load_value  # Horizontal force component
    # Note: F[3*load_node+1] = 0 (no vertical force)
    # Note: F[3*load_node+2] = 0 (no applied moment)
    
    # Step 4: Apply boundary conditions (fixed supports at base)
    fixed_nodes = []
    for i, node in enumerate(nodes):
        if abs(node[1] - 0.0) < 1e-6:  # Nodes at y=0 (base of legs)
            fixed_nodes.append(i)
    
    # Apply penalty method for boundary conditions
    K_bc, F_bc = apply_boundary_conditions_frame(K, F, fixed_nodes)
    
    # Step 5: Solve linear system K*u = F for displacements
    # Use sparse matrix solver for efficiency with larger systems
    displacements = spsolve(csr_matrix(K_bc), F_bc)
    
    return nodes, elements, displacements, F, load_node

# TODO: create a function to define boundary conditions at the base of the legs
def define_boundary_conditions(frame):
    boundary_conditions = []
    for leg in frame["legs"]:
        boundary_conditions.append({
            "x": leg["x"],
            "y": leg["y"],
            "type": "fixed"
        })
    return boundary_conditions

# =============================================================================
# ANALYSIS AND RESULTS PROCESSING
# =============================================================================

def main():
    """
    Main function to execute complete finite element analysis of the portal frame.
    
    This function demonstrates the complete workflow:
    1. Display original frame geometry
    2. Solve for displacements under specified loading
    3. Process and display results
    4. Visualize deformed vs. original configuration
    
    The analysis considers a portal frame under lateral loading, which is a
    common structural engineering problem for buildings subjected to wind loads.
    """
    print("=" * 50)
    print("FINITE ELEMENT FRAME ANALYSIS")
    print("=" * 50)
    
    # Step 1: Display original frame structure
    print("\n1. Original Frame Structure:")
    plot_frame(frame)
    
    # Step 2: Solve finite element problem
    load_value = 1000  # 1000 N horizontal load (typical wind load magnitude)
    print(f"\n2. Solving frame with {load_value} N horizontal load...")
    
    # Execute finite element analysis
    nodes, elements, displacements, forces, load_node = solve_frame(frame, load_value)
    
    # Display problem statistics
    print(f"   - Number of nodes: {len(nodes)}")
    print(f"   - Number of elements: {len(elements)}")
    print(f"   - Load applied at node {load_node}: ({nodes[load_node][0]:.1f}, {nodes[load_node][1]:.1f}) m")
    print(f"   - Maximum displacement: {np.max(np.abs(displacements)):.6f} m ({np.max(np.abs(displacements))*1000:.3f} mm)")
    
    # Step 3: Analyze results - find maximum horizontal displacement
    max_u_x = 0
    max_u_x_node = 0
    for i in range(len(nodes)):
        u_x = abs(displacements[3*i])  # Horizontal displacement magnitude
        if u_x > max_u_x:
            max_u_x = u_x
            max_u_x_node = i
    
    print(f"   - Maximum horizontal displacement: {max_u_x:.6f} m at node {max_u_x_node}")
    
    # Step 4: Display key node displacements
    print(f"\n3. Key Node Displacements:")
    key_nodes = [0, 5, 11, load_node]  # Base left, top left, top right, load point
    for i in key_nodes:
        u_x = displacements[3*i]     # u displacement (horizontal)
        u_y = displacements[3*i+1]   # v displacement (vertical)
        theta = displacements[3*i+2] # θ rotation
        print(f"   Node {i} ({nodes[i][0]:.1f}, {nodes[i][1]:.1f}): u_x={u_x*1000:.3f}mm, u_y={u_y*1000:.3f}mm, θ={theta*1000:.3f}mrad")
    
    # Step 5: Visualize results with exaggerated deformation
    print(f"\n4. Displaying deformed frame (scaled {1000}x for visibility):")
    plot_frame(frame, displacements, scale_factor=1000)
    
    # Step 6: Show realistic deformation scale  
    print(f"\n5. Displaying with realistic deformation scale:")
    plot_frame(frame, displacements, scale_factor=1)
    
    # Analysis summary
    print(f"\nAnalysis complete! The frame shows lateral deflection under horizontal loading.")
    print(f"The structure behaves as expected with fixed supports at the base.")
    print(f"Maximum lateral drift: {max_u_x*1000:.3f} mm (≈ L/{5000/max_u_x:.0f} where L=5m)")
    
    return nodes, elements, displacements

def analyze_load_cases(frame):
    """
    Analyze multiple load cases to demonstrate solver capabilities and verify linearity.
    
    This function performs parametric analysis to:
    1. Verify linear elastic behavior (displacement ∝ load)
    2. Demonstrate solver robustness across different load magnitudes
    3. Generate load-displacement curves for structural assessment
    
    Parameters:
    -----------
    frame : dict
        Frame geometry definition
        
    Returns:
    --------
    load_cases : list
        Applied load magnitudes (N)
    max_displacements : list  
        Maximum displacements for each load case (m)
        
    Engineering Significance:
    ------------------------
    - Linear relationship confirms elastic behavior (no yielding/buckling)
    - Consistent displacement/load ratio indicates numerical stability
    - Load-displacement curve useful for serviceability limit state checks
    """
    print("\n" + "="*50)
    print("MULTIPLE LOAD CASE ANALYSIS")
    print("="*50)
    
    # Define load cases ranging from light to heavy loading
    load_cases = [500, 1000, 2000, 5000]  # Different load magnitudes in N
    max_displacements = []
    
    # Analyze each load case
    for load in load_cases:
        # Solve frame for current load magnitude
        nodes, elements, displacements, forces, load_node = solve_frame(frame, load)
        
        # Find maximum displacement (any DOF, any node)
        max_disp = np.max(np.abs(displacements))
        max_displacements.append(max_disp)
        
        print(f"Load: {load:4d} N -> Max displacement: {max_disp*1000:.3f} mm")
    
    # Plot load vs displacement relationship
    plt.figure(figsize=(10, 6))
    plt.plot(load_cases, [d*1000 for d in max_displacements], 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Applied Load (N)')
    plt.ylabel('Maximum Displacement (mm)')
    plt.title('Load vs Maximum Displacement (Linear Elastic Analysis)')
    plt.grid(True, alpha=0.3)
    
    # Add linear fit line to verify linearity
    slope = max_displacements[-1] / load_cases[-1]  # mm/N
    linear_fit = [slope * load * 1000 for load in load_cases]
    plt.plot(load_cases, linear_fit, 'r--', alpha=0.7, label=f'Linear fit (slope={slope*1e6:.3f} mm/N)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Check linearity (should be constant for elastic analysis)
    displacement_per_unit_load = [d/l for d, l in zip(max_displacements, load_cases)]
    print(f"\nLinearity check (displacement per unit load):")
    print(f"{'Load (N)':>8} {'Disp/Load (mm/N)':>18} {'% Deviation':>12}")
    print("-" * 40)
    
    # Calculate average ratio and deviations
    avg_ratio = np.mean(displacement_per_unit_load)
    for i, ratio in enumerate(displacement_per_unit_load):
        deviation = 100 * (ratio - avg_ratio) / avg_ratio if avg_ratio != 0 else 0
        print(f"{load_cases[i]:8d} {ratio*1e6:18.6f} {deviation:12.3f}%")
    
    print(f"\nAverage displacement per unit load: {avg_ratio*1e6:.6f} mm/N")
    print(f"Standard deviation: {np.std(displacement_per_unit_load)*1e6:.6f} mm/N")
    
    # Engineering assessment
    max_drift_ratio = max(max_displacements) / 5.0  # drift/height ratio
    print(f"\nStructural Assessment:")
    print(f"- Maximum inter-story drift ratio: {max_drift_ratio:.6f} (≈ 1/{1/max_drift_ratio:.0f})")
    if max_drift_ratio < 1/500:
        print("- PASS: Drift within typical serviceability limits (< 1/500)")
    else:
        print("- WARNING: Drift exceeds typical serviceability limits (> 1/500)")
    
    return load_cases, max_displacements

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block for the finite element frame analysis program.
    
    This block demonstrates the complete analysis workflow:
    1. Single load case analysis with detailed results
    2. Multiple load case analysis for parametric study
    3. Verification of linear elastic behavior
    
    The program can be extended to include:
    - Dynamic analysis (modal analysis, time history)
    - Nonlinear analysis (material/geometric nonlinearity)  
    - Optimization studies (sizing, topology)
    - Different loading conditions (gravity, seismic, etc.)
    """
    # Run main analysis with comprehensive output
    nodes, elements, displacements = main()
    
    # Run additional parametric analysis
    analyze_load_cases(frame)
