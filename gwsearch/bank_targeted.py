"""
Generate targeted template banks around known events for pipeline validation.

This module provides functions to create dense template banks centered on
specific known gravitational wave events like GW150914, allowing for
rapid validation of the search pipeline.

Literature Reference:
- Harry et al. 2014 "Template bank generation" PhysRevD.89.024010
- GW150914 parameters: Abbott et al. 2016 PhysRevLett.116.061102
"""

import numpy as np
from typing import List, Dict
import logging

LOG = logging.getLogger(__name__)


def generate_gw150914_test_bank(grid_points_per_dim=10) -> List[Dict]:
    """
    Generate template bank centered on GW150914 parameters.
    
    GW150914 source-frame masses: m1=36.2 M☉, m2=29.1 M☉
    This creates a dense grid around these values for testing.
    
    Args:
        grid_points_per_dim: Number of grid points per dimension
    
    Returns:
        List of template parameter dictionaries
    
    Example:
        >>> bank = generate_gw150914_test_bank(grid_points_per_dim=15)
        >>> len(bank)  # ~120 templates
        112
    """
    # GW150914 parameters with ±20% margin
    m1_center = 36.2
    m2_center = 29.1
    margin = 0.20
    
    m1_min = m1_center * (1 - margin)  # ~29 M☉
    m1_max = m1_center * (1 + margin)  # ~43 M☉
    m2_min = m2_center * (1 - margin)  # ~23 M☉
    m2_max = m2_center * (1 + margin)  # ~35 M☉
    
    # Generate grid
    m1_range = np.linspace(m1_min, m1_max, grid_points_per_dim)
    m2_range = np.linspace(m2_min, m2_max, grid_points_per_dim)
    
    templates = []
    for m1 in m1_range:
        for m2 in m2_range:
            # Physical constraint: m1 >= m2
            if m1 >= m2:
                templates.append({
                    'mass1': float(m1),
                    'mass2': float(m2),
                    'spin1z': 0.0,  # Non-spinning for simplicity
                    'spin2z': 0.0,
                    'approximant': 'IMRPhenomD',
                    'f_lower': 15.0
                })
    
    LOG.info("Generated %d targeted templates around GW150914 (m1=%.1f, m2=%.1f)",
             len(templates), m1_center, m2_center)
    
    return templates


def generate_bbh_moderate_bank(mass_range=(10, 50), num_templates=500) -> List[Dict]:
    """
    Generate moderate-density BBH template bank for testing.
    
    Uses simple geometric spacing (not optimal, but adequate for validation).
    For production searches, use PyCBC's stochastic bank generator.
    
    Args:
        mass_range: (min_mass, max_mass) in solar masses
        num_templates: Target number of templates
    
    Returns:
        List of template parameter dictionaries
    
    Example:
        >>> bank = generate_bbh_moderate_bank(mass_range=(10, 50), num_templates=500)
        >>> len(bank)  # ~500-600 templates
        528
    """
    min_mass, max_mass = mass_range
    
    # Estimate grid density: N = n_m1 * n_m2 / 2 (triangle)
    # For num_templates=500: n ≈ sqrt(2*500) ≈ 32 per dimension
    n_per_dim = int(np.sqrt(2 * num_templates))
    
    m1_range = np.linspace(min_mass, max_mass, n_per_dim)
    m2_range = np.linspace(min_mass, max_mass, n_per_dim)
    
    templates = []
    for m1 in m1_range:
        for m2 in m2_range:
            if m1 >= m2:  # Physical constraint
                templates.append({
                    'mass1': float(m1),
                    'mass2': float(m2),
                    'spin1z': 0.0,
                    'spin2z': 0.0,
                    'approximant': 'IMRPhenomD',
                    'f_lower': 15.0
                })
    
    LOG.info("Generated %d moderate-density templates (mass range: %.1f-%.1f M☉)",
             len(templates), min_mass, max_mass)
    
    return templates


def generate_multi_event_bank(events=None, margin=0.15, points_per_event=64) -> List[Dict]:
    """
    Generate template bank covering multiple known events.
    
    Useful for validating pipeline on O1/O2/O3 catalog events.
    
    Args:
        events: List of (m1, m2) tuples. If None, uses O1/O2 loud events
        margin: ±fractional margin around each event
        points_per_event: Number of templates per event region
    
    Returns:
        List of template parameter dictionaries
    
    Example:
        >>> # Cover GW150914, GW151226, GW170817
        >>> bank = generate_multi_event_bank()
        >>> len(bank)  # ~200-300 templates covering all events
    """
    if events is None:
        # Default: O1/O2 loud BBH events
        events = [
            (36.2, 29.1),  # GW150914
            (14.2, 7.5),   # GW151226
            (31.2, 19.4),  # GW151012 (LVT)
            (50.6, 34.3),  # GW170104
            (35.5, 23.9),  # GW170608
            (30.5, 25.3),  # GW170814
            (1.46, 1.27),  # GW170817 (BNS)
        ]
    
    templates = []
    for m1_center, m2_center in events:
        # Create local grid
        m1_min = m1_center * (1 - margin)
        m1_max = m1_center * (1 + margin)
        m2_min = m2_center * (1 - margin)
        m2_max = m2_center * (1 + margin)
        
        n_per_dim = int(np.sqrt(points_per_event))
        m1_range = np.linspace(m1_min, m1_max, n_per_dim)
        m2_range = np.linspace(m2_min, m2_max, n_per_dim)
        
        for m1 in m1_range:
            for m2 in m2_range:
                if m1 >= m2:
                    # Choose appropriate approximant based on mass
                    if m1 + m2 < 3.0:
                        approx = 'TaylorF2'  # BNS
                    else:
                        approx = 'IMRPhenomD'  # BBH
                    
                    templates.append({
                        'mass1': float(m1),
                        'mass2': float(m2),
                        'spin1z': 0.0,
                        'spin2z': 0.0,
                        'approximant': approx,
                        'f_lower': 15.0 if m1 + m2 > 3.0 else 20.0
                    })
    
    LOG.info("Generated %d templates covering %d known events", len(templates), len(events))
    
    return templates


def save_bank_to_dict(templates: List[Dict]) -> Dict:
    """
    Convert template list to bank dictionary format expected by pipeline.
    
    Args:
        templates: List of template parameter dictionaries
    
    Returns:
        Bank dictionary with 'templates' and 'metadata' keys
    """
    bank = {
        'templates': templates,
        'metadata': {
            'type': 'targeted',
            'n_templates': len(templates),
            'generator': 'bank_targeted.py',
            'coverage': 'dense_grid_around_known_events'
        }
    }
    
    return bank
