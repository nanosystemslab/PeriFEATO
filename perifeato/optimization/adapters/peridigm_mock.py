#!/usr/bin/env python3
"""
Mock Peridigm Module for Local Development
===========================================

This module simulates Peridigm's damage output without running the actual code.
Used for rapid local development before integrating with HPC Peridigm.

Key features:
1. Realistic damage behavior (density-dependent)
2. Same interface as real Peridigm module
3. Fast (runs in < 1 second)
4. Configurable noise and damage parameters

Author: PhD Candidate
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Optional

# Flag to identify this is a mock
IS_MOCK = True
VERSION = "1.0-mock"


def run_peridigm_mock(
    mesh,
    density: np.ndarray,
    fracture_zone_ids: List[int],
    shell_body_ids: List[int],
    impact_velocity: float = 25.0,
    config: Optional[Dict] = None
) -> Dict:
    """
    Mock Peridigm simulation that generates realistic damage field.
    
    Behavior:
    - Fracture zone: damage inversely proportional to density
      (low density → high damage, which is what we want)
    - Shell body: damage proportional to stress/density ratio
      (high stress + low density → damage, which is bad)
    
    Args:
        mesh: Mesh object with n_cells, cell_centers, etc.
        density: Density field (0-1 for each cell)
        fracture_zone_ids: Region IDs for fracture zone
        shell_body_ids: Region IDs for main shell
        impact_velocity: Simulated impact speed (m/s)
        config: Optional config dict for tuning behavior
    
    Returns:
        dict with:
            'damage_field': np.ndarray (0-1 for each cell)
            'D_fracture_zone': float (mean damage in fracture zone)
            'D_shell_body': float (mean damage in shell body)
            'exodus_file': None (mock doesn't write files)
            'is_mock': True
    """
    
    # Default config
    defaults = {
        'noise_level': 0.05,
        'fracture_sensitivity': 1.5,
        'shell_damage_factor': 0.3,
        'impact_scaling': 0.02
    }
    if config is None:
        config = defaults
    else:
        merged = defaults.copy()
        merged.update(config)
        config = merged

    n_cells = len(density)
    damage = np.zeros(n_cells)
    
    # Get region masks
    fracture_mask = np.isin(mesh.cell_regions, fracture_zone_ids)
    shell_mask = np.isin(mesh.cell_regions, shell_body_ids)
    
    # ========== FRACTURE ZONE DAMAGE ==========
    # Logic: We WANT fracture here
    # Low density → high damage (good for fracture)
    # High density → low damage (prevents fracture)
    
    for cell_id in np.where(fracture_mask)[0]:
        rho = density[cell_id]
        
        # Inverse relationship with density
        # At ρ=0.1: damage ≈ 0.9 (very damaged, will fracture)
        # At ρ=0.5: damage ≈ 0.5 (partially damaged)
        # At ρ=0.9: damage ≈ 0.1 (minimal damage, won't fracture)
        
        base_damage = (1.0 - rho) * config['fracture_sensitivity']
        
        # Add impact velocity effect
        impact_effect = impact_velocity * config['impact_scaling']
        
        # Add spatial variation (simulate heterogeneous fracture)
        spatial_factor = 1.0 + 0.2 * np.sin(cell_id * 0.1)
        
        # Combine
        damage[cell_id] = base_damage * impact_effect * spatial_factor
    
    # ========== SHELL BODY DAMAGE ==========
    # Logic: We DON'T want damage here
    # High density → low damage (protects shell)
    # Low density → potentially high damage (shell fails)
    # Also depends on stress level (estimate based on position)
    
    for cell_id in np.where(shell_mask)[0]:
        rho = density[cell_id]
        
        # Estimate stress based on position
        # (In real FEA, we'd use actual stress field)
        # For mock: assume stress higher near impact points
        z_position = mesh.cell_centers[cell_id, 2]  # Axial position
        z_normalized = z_position / mesh.height
        
        # Higher stress near mid-height (impact zone)
        stress_estimate = 1.0 - 2.0 * abs(z_normalized - 0.5)
        stress_estimate = max(0.1, stress_estimate)  # Minimum stress
        
        # Damage depends on stress/density ratio
        # High stress + low density → high damage
        # Low stress or high density → low damage
        
        base_damage = stress_estimate * (1.0 - rho) * config['shell_damage_factor']
        
        # Add impact velocity effect (smaller than fracture zone)
        impact_effect = impact_velocity * config['impact_scaling'] * 0.5
        
        damage[cell_id] = base_damage * impact_effect
    
    # ========== ADD NOISE ==========
    # Simulate material heterogeneity and computational noise
    noise = np.random.normal(0, config['noise_level'], n_cells)
    damage += noise
    
    # ========== CLIP TO VALID RANGE ==========
    damage = np.clip(damage, 0.0, 1.0)
    
    # ========== COMPUTE METRICS ==========
    D_fracture = np.mean(damage[fracture_mask]) if fracture_mask.any() else 0.0
    D_shell = np.mean(damage[shell_mask]) if shell_mask.any() else 0.0
    
    return {
        'damage_field': damage,
        'D_fracture_zone': D_fracture,
        'D_shell_body': D_shell,
        'max_damage': np.max(damage),
        'exodus_file': None,  # Mock doesn't write files
        'is_mock': True,
        'impact_velocity_used': impact_velocity,
        'config_used': config
    }


def validate_mock_behavior():
    """
    Test that mock Peridigm behaves correctly.
    
    Checks:
    1. Low density in fracture zone → high damage
    2. High density in fracture zone → low damage
    3. Damage values in valid range [0, 1]
    """
    
    print("Validating mock Peridigm behavior...")
    
    # Create dummy mesh
    class DummyMesh:
        def __init__(self):
            self.n_cells = 100
            self.cell_regions = np.concatenate([
                np.ones(30, dtype=int),    # Region 1: fracture zone
                np.ones(70, dtype=int) * 2  # Region 2: shell body
            ])
            self.cell_centers = np.random.rand(100, 3) * 0.01
            self.height = 0.028
    
    mesh = DummyMesh()
    
    # Test 1: Low density in fracture zone
    density_low = np.ones(100) * 0.2
    result_low = run_peridigm_mock(
        mesh, density_low,
        fracture_zone_ids=[1],
        shell_body_ids=[2],
        impact_velocity=25.0
    )
    
    # Test 2: High density in fracture zone
    density_high = np.ones(100) * 0.8
    result_high = run_peridigm_mock(
        mesh, density_high,
        fracture_zone_ids=[1],
        shell_body_ids=[2],
        impact_velocity=25.0
    )
    
    print(f"  Low density case:")
    print(f"    D_fracture = {result_low['D_fracture_zone']:.3f} (should be high)")
    print(f"    D_shell = {result_low['D_shell_body']:.3f}")
    
    print(f"  High density case:")
    print(f"    D_fracture = {result_high['D_fracture_zone']:.3f} (should be low)")
    print(f"    D_shell = {result_high['D_shell_body']:.3f}")
    
    # Assertions
    assert result_low['D_fracture_zone'] > result_high['D_fracture_zone'], \
        "Low density should produce higher damage in fracture zone"
    
    assert 0 <= result_low['D_fracture_zone'] <= 1, \
        "Damage must be in [0, 1] range"
    
    assert 0 <= result_high['D_shell_body'] <= 1, \
        "Damage must be in [0, 1] range"
    
    print("✓ Mock Peridigm validation passed!")
    
    return True


# ============================================================================
# Interface for Real Peridigm (to be implemented later on HPC)
# ============================================================================

def run_peridigm_real(
    mesh,
    density: np.ndarray,
    fracture_zone_ids: List[int],
    shell_body_ids: List[int],
    impact_velocity: float = 25.0,
    config: Optional[Dict] = None
) -> Dict:
    """
    PLACEHOLDER for real Peridigm integration.
    
    This function has the SAME SIGNATURE as run_peridigm_mock()
    so they can be swapped seamlessly.
    
    Implementation steps (on HPC):
    1. Write Peridigm input deck (.yaml)
    2. Convert mesh to Peridigm format (.g file)
    3. Submit SLURM job
    4. Wait for completion
    5. Read Exodus output (.e file)
    6. Extract damage field
    7. Return in same dict format as mock
    
    Returns:
        Same dict structure as mock:
        {
            'damage_field': np.ndarray,
            'D_fracture_zone': float,
            'D_shell_body': float,
            'exodus_file': str (path to .e file),
            'is_mock': False
        }
    """
    raise NotImplementedError(
        "Real Peridigm integration not yet implemented. "
        "This function will be completed on HPC with access to Peridigm."
    )


# ============================================================================
# Convenience function for automatic mock/real selection
# ============================================================================

def run_peridigm(
    mesh,
    density: np.ndarray,
    fracture_zone_ids: List[int],
    shell_body_ids: List[int],
    impact_velocity: float = 25.0,
    config: Optional[Dict] = None,
    use_mock: bool = True
) -> Dict:
    """
    Run Peridigm (mock or real depending on flag).
    
    Args:
        ... (same as run_peridigm_mock)
        use_mock: If True, use mock. If False, use real Peridigm.
    
    Returns:
        Damage results dict
    """
    
    if use_mock:
        return run_peridigm_mock(
            mesh, density, fracture_zone_ids, shell_body_ids,
            impact_velocity, config
        )
    else:
        return run_peridigm_real(
            mesh, density, fracture_zone_ids, shell_body_ids,
            impact_velocity, config
        )


# ============================================================================
# Main (for testing)
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Mock Peridigm Module Test")
    print("="*60)
    
    validate_mock_behavior()
    
    print("\n✓ Mock module ready for use!")
    print("\nTo use in your code:")
    print("  from peridigm_mock import run_peridigm_mock")
    print("  results = run_peridigm_mock(mesh, density, ...)")
