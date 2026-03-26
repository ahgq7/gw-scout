"""
Advanced Background Estimation using Time-Slides

This module implements production-quality background estimation for
calculating False Alarm Rate (FAR) and Inverse FAR (IFAR) for gravitational
wave candidate events.

References:
- Abbott et al. (2016) "Observing gravitational-wave transient GW150914"
  Phys. Rev. Lett. 116, 061102
- Usman et al. (2016) "The PyCBC search for gravitational waves from compact
  binary coalescence" Class. Quantum Grav. 33, 215004
- Nitz et al. (2018) "1-OGC: The first open gravitational-wave catalog of
  binary mergers from analysis of public Advanced LIGO data"
  Astrophys. J. 872, 195
"""

from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

LOG = logging.getLogger(__name__)


@dataclass
class BackgroundConfig:
    """Configuration for background estimation."""
    
    # Time-slide parameters
    num_slides: int = 100  # Number of time-slides (production: 100-1000+)
    slide_step: float = 0.1  # Time-slide step size (seconds)
    max_slide_offset: float = 100.0  # Maximum time offset (seconds)
    circular_slides: bool = True  # Use circular vs linear slides
    
    # Statistical parameters
    min_background_triggers: int = 100  # Minimum background triggers for reliable FAR
    confidence_level: float = 0.95  # Confidence level for error estimation
    
    # Detector configuration
    reference_detector: str = "H1"  # Reference detector (not shifted)
    shifted_detectors: List[str] = None  # Detectors to shift (default: all others)
    
    # Coincidence parameters  
    coincidence_window: float = 0.015  # Coincidence window (seconds) for H1-L1
    
    def __post_init__(self):
        if self.shifted_detectors is None:
            self.shifted_detectors = ["L1"]  # Default: shift L1 relative to H1


class TimeSlideBackgroundEstimator:
    """
    Time-slide background estimator for gravitational wave searches.
    
    Implements the time-slide method used in LIGO/Virgo analyses to estimate
    the background noise distribution and calculate False Alarm Rates (FAR).
    
    The method works by:
    1. Take triggers from multiple detectors
    2. Apply time-shifts to non-reference detectors
    3. Re-calculate coincidences with shifted times
    4. Shifted coincidences are PURE BACKGROUND (no real signals)
    5. Build background distribution from shifted coincidences
    6. Calculate FAR for foreground (zero-lag) candidates
    """
    
    def __init__(self, cfg: BackgroundConfig):
        """
        Initialize background estimator.
        
        Args:
            cfg: Background configuration
        """
        self.cfg = cfg
        self.slide_offsets: List[float] = []
        self.background_triggers: List[Dict] = []
        self.foreground_triggers: List[Dict] = []
        
        # Generate time-slide offsets
        self._generate_slide_offsets()
    
    def _generate_slide_offsets(self):
        """
        Generate time-slide offset values.
        
        For H1-L1:
        - H1 (reference): offset = 0
        - L1: offsets = [-100s to +100s] in steps of 0.1s
        
        This gives ~2000 slides total, which is standard for production.
        """
        LOG.info("Generating time-slide offsets")
        LOG.info("  Reference detector: %s (no shift)", self.cfg.reference_detector)
        LOG.info("  Shifted detectors: %s", self.cfg.shifted_detectors)
        LOG.info("  Slide step: %.3f s", self.cfg.slide_step)
        LOG.info("  Max offset: %.1f s", self.cfg.max_slide_offset)
        
        # Generate offsets symmetrically around zero
        # Don't include zero (that's foreground/zero-lag)
        offsets = []
        
        offset = self.cfg.slide_step
        while offset <= self.cfg.max_slide_offset:
            offsets.append(offset)
            offsets.append(-offset)
            offset += self.cfg.slide_step
        
        # Limit to requested number of slides
        if len(offsets) > self.cfg.num_slides:
            # Take uniform subset
            indices = np.linspace(0, len(offsets)-1, self.cfg.num_slides, dtype=int)
            offsets = [offsets[i] for i in indices]
        
        self.slide_offsets = sorted(offsets)
        
        LOG.info("Generated %d time-slide offsets", len(self.slide_offsets))
        if len(self.slide_offsets) > 0:
            LOG.info("  Range: [%.2f, %.2f] seconds", 
                    min(self.slide_offsets), max(self.slide_offsets))
    
    def apply_time_slide(self, triggers: Dict[str, List[Dict]], 
                        slide_offset: float) -> Dict[str, List[Dict]]:
        """
        Apply time-slide to trigger list.
        
        Args:
            triggers: Dictionary of {ifo: [trigger_list]}
            slide_offset: Time offset to apply (seconds)
        
        Returns:
            New trigger dictionary with shifted times
        """
        shifted_triggers = {}
        
        for ifo, trig_list in triggers.items():
            if ifo == self.cfg.reference_detector:
                # Reference detector: no shift
                shifted_triggers[ifo] = trig_list
            elif ifo in self.cfg.shifted_detectors:
                # Shifted detector: apply offset
                shifted_list = []
                for trig in trig_list:
                    shifted_trig = trig.copy()
                    shifted_trig['gps'] = trig['gps'] + slide_offset
                    shifted_trig['time_slide_offset'] = slide_offset
                    shifted_list.append(shifted_trig)
                shifted_triggers[ifo] = shifted_list
            else:
                # Unknown detector: no shift
                shifted_triggers[ifo] = trig_list
        
        return shifted_triggers
    
    def find_coincidences_in_slide(self, triggers: Dict[str, List[Dict]], 
                                   slide_id: int) -> List[Dict]:
        """
        Find coincident triggers in time-slid data.
        
        Args:
            triggers: Dictionary of {ifo: [trigger_list]} (already shifted)
            slide_id: Slide identifier
        
        Returns:
            List of coincident events
        """
        from .coincidence import find_coincidences
        
        # Find coincidences with configured window
        coincs = find_coincidences(
            triggers,
            window=self.cfg.coincidence_window,
            require_same_template=True,
        )
        
        # Tag with slide_id
        for coinc in coincs:
            coinc['slide_id'] = slide_id
            coinc['is_background'] = True
        
        return coincs
    
    def compute_background(self, per_ifo_triggers: Dict[str, List[Dict]]) -> Dict:
        """
        Compute background distribution from time-slides.
        
        Args:
            per_ifo_triggers: Dictionary of {ifo: [trigger_list]}
        
        Returns:
            Background statistics dictionary
        """
        LOG.info("Computing background distribution with time-slides")
        LOG.info("  Input triggers: %s", 
                {ifo: len(trigs) for ifo, trigs in per_ifo_triggers.items()})
        
        background_coincs = []
        
        # Perform time-slides
        for slide_idx, offset in enumerate(self.slide_offsets):
            # Apply slide
            shifted_triggers = self.apply_time_slide(per_ifo_triggers, offset)
            
            # Find coincidences
            coincs = self.find_coincidences_in_slide(shifted_triggers, slide_idx)
            background_coincs.extend(coincs)
            
            if (slide_idx + 1) % 10 == 0 or slide_idx == 0:
                LOG.info("  Slide %d/%d (offset=%.2fs): %d background coincidences",
                        slide_idx + 1, len(self.slide_offsets), offset, len(coincs))
        
        self.background_triggers = background_coincs
        
        LOG.info("Total background coincidences: %d", len(background_coincs))
        
        # Extract ranking statistics
        if len(background_coincs) > 0:
            network_stats = [c['network_stat'] for c in background_coincs]
            network_stats_sorted = sorted(network_stats, reverse=True)
        else:
            network_stats_sorted = []
        
        # Calculate background statistics
        total_background_time = len(self.slide_offsets) * self._estimate_observing_time(per_ifo_triggers)
        
        background_stats = {
            'num_slides': len(self.slide_offsets),
            'num_background_triggers': len(background_coincs),
            'background_time_years': total_background_time / (365.25 * 86400),
            'network_stat_distribution': network_stats_sorted,
            'loudest_background_stat': network_stats_sorted[0] if network_stats_sorted else 0.0,
        }
        
        LOG.info("Background statistics:")
        LOG.info("  Slides: %d", background_stats['num_slides'])
        LOG.info("  Background triggers: %d", background_stats['num_background_triggers'])
        LOG.info("  Effective background time: %.2f years", 
                background_stats['background_time_years'])
        if len(network_stats_sorted) > 0:
            LOG.info("  Loudest background: %.2f", background_stats['loudest_background_stat'])
        
        return background_stats
    
    def _estimate_observing_time(self, per_ifo_triggers: Dict[str, List[Dict]]) -> float:
        """
        Estimate observing time from trigger distribution.
        
        Args:
            per_ifo_triggers: Dictionary of {ifo: [trigger_list]}
        
        Returns:
            Estimated observing time in seconds
        """
        # Simple estimate: time span of triggers
        all_times = []
        for trig_list in per_ifo_triggers.values():
            for trig in trig_list:
                all_times.append(trig['gps'])
        
        if len(all_times) < 2:
            return 0.0
        
        time_span = max(all_times) - min(all_times)
        return time_span
    
    def calculate_far(self, candidate_stat: float, background_stats: Dict) -> Tuple[float, float]:
        """
        Calculate False Alarm Rate (FAR) for a candidate event.
        
        FAR = (number of background events louder than candidate) / (total background time)
        
        Args:
            candidate_stat: Network statistic of candidate
            background_stats: Background statistics from compute_background()
        
        Returns:
            (FAR in Hz, IFAR in years)
        """
        network_stat_dist = background_stats['network_stat_distribution']
        background_time_sec = background_stats['background_time_years'] * 365.25 * 86400
        
        if background_time_sec == 0:
            # No background time
            return float('inf'), 0.0
        
        # Count background events louder than candidate
        n_louder = sum(1 for stat in network_stat_dist if stat >= candidate_stat)
        
        # Add +1 to avoid zero FAR (statistical treatment)
        # This is conservative - assumes we might have missed one
        n_louder_conservative = n_louder + 1
        
        # FAR = count / time (in Hz)
        far_hz = n_louder_conservative / background_time_sec
        
        # IFAR = 1/FAR (in years)
        if far_hz > 0:
            ifar_years = 1.0 / (far_hz * 365.25 * 86400)
        else:
            ifar_years = float('inf')
        
        return far_hz, ifar_years
    
    def calculate_p_value(self, candidate_stat: float, background_stats: Dict) -> float:
        """
        Calculate p-value (statistical significance) for a candidate.
        
        p-value = probability of background producing event at least this loud
        
        Args:
            candidate_stat: Network statistic of candidate
            background_stats: Background statistics
        
        Returns:
            p-value (0 to 1)
        """
        network_stat_dist = background_stats['network_stat_distribution']
        
        if len(network_stat_dist) == 0:
            return 1.0  # No background data = not significant
        
        # Fraction of background louder or equal
        n_total = len(network_stat_dist)
        n_louder = sum(1 for stat in network_stat_dist if stat >= candidate_stat)
        
        p_value = (n_louder + 1) / (n_total + 1)  # +1 for statistical smoothing
        
        return p_value
    
    def calculate_significance(self, candidate_stat: float, background_stats: Dict) -> float:
        """
        Calculate significance in sigma (Gaussian equivalent).
        
        Converts p-value to Gaussian sigma using inverse error function.
        
        Args:
            candidate_stat: Network statistic of candidate
            background_stats: Background statistics
        
        Returns:
            Significance in sigma (e.g., 5.0 = 5-sigma detection)
        """
        from scipy.special import erfinv
        
        p_value = self.calculate_p_value(candidate_stat, background_stats)
        
        if p_value >= 1.0:
            return 0.0
        
        # Convert p-value to sigma
        # p = 0.5 * erfc(sigma / sqrt(2))
        # sigma = sqrt(2) * erfinv(1 - 2*p)
        
        # Clamp p_value to avoid numerical issues
        p_clamped = max(1e-15, min(1.0 - 1e-15, p_value))
        
        sigma = np.sqrt(2) * erfinv(1 - 2 * p_clamped)
        
        return abs(sigma)
    
    def rank_candidates(self, candidates: List[Dict], background_stats: Dict) -> List[Dict]:
        """
        Rank candidates by significance and calculate FAR/IFAR for each.
        
        Args:
            candidates: List of candidate coincidences
            background_stats: Background statistics
        
        Returns:
            Sorted list of candidates with FAR/IFAR/significance
        """
        LOG.info("Ranking %d candidates against background", len(candidates))
        
        ranked_candidates = []
        
        for cand in candidates:
            network_stat = cand['network_stat']
            
            # Calculate FAR and IFAR
            far_hz, ifar_years = self.calculate_far(network_stat, background_stats)
            
            # Calculate p-value and significance
            p_value = self.calculate_p_value(network_stat, background_stats)
            sigma = self.calculate_significance(network_stat, background_stats)
            
            # Annotate candidate
            cand_annotated = cand.copy()
            cand_annotated.update({
                'far_hz': far_hz,
                'far_per_year': far_hz * 365.25 * 86400,
                'ifar_years': ifar_years,
                'p_value': p_value,
                'significance_sigma': sigma,
            })
            
            ranked_candidates.append(cand_annotated)
        
        # Sort by significance (highest first)
        ranked_candidates.sort(key=lambda x: x['significance_sigma'], reverse=True)
        
        # Log top candidates
        LOG.info("Top candidates by significance:")
        for i, cand in enumerate(ranked_candidates[:5]):
            LOG.info("  #%d: network_stat=%.2f, FAR=%.2e/yr, IFAR=%.2f yr, sigma=%.2f",
                    i+1, cand['network_stat'], cand['far_per_year'], 
                    cand['ifar_years'], cand['significance_sigma'])
        
        return ranked_candidates
    
    def set_detection_threshold(self, background_stats: Dict, 
                               target_far_per_year: float = 1.0) -> float:
        """
        Set detection threshold based on target FAR.
        
        Args:
            background_stats: Background statistics
            target_far_per_year: Target FAR (events per year), e.g., 1.0 = 1 per year
        
        Returns:
            Network statistic threshold
        """
        network_stat_dist = background_stats['network_stat_distribution']
        background_time_years = background_stats['background_time_years']
        
        if len(network_stat_dist) == 0 or background_time_years == 0:
            LOG.warning("No background data, cannot set threshold")
            return 0.0
        
        # Calculate how many background events correspond to target FAR
        n_allowed = int(target_far_per_year * background_time_years)
        n_allowed = max(1, n_allowed)  # At least 1
        
        # Threshold is the n_allowed'th loudest background event
        if n_allowed <= len(network_stat_dist):
            threshold = network_stat_dist[n_allowed - 1]
        else:
            # Target FAR is too high, use minimum background stat
            threshold = min(network_stat_dist) if network_stat_dist else 0.0
        
        LOG.info("Detection threshold for FAR=%.2f/yr: network_stat >= %.2f",
                target_far_per_year, threshold)
        
        return threshold


def estimate_background_with_timeslides(
    per_ifo_triggers: Dict[str, List[Dict]],
    foreground_candidates: List[Dict],
    cfg: BackgroundConfig = None,
) -> Tuple[Dict, List[Dict]]:
    """
    Convenience function for background estimation.
    
    Args:
        per_ifo_triggers: Per-IFO trigger lists
        foreground_candidates: Foreground (zero-lag) coincidences
        cfg: Background configuration (default if None)
    
    Returns:
        (background_stats, ranked_candidates)
    """
    if cfg is None:
        cfg = BackgroundConfig()
    
    estimator = TimeSlideBackgroundEstimator(cfg)
    
    # Compute background
    background_stats = estimator.compute_background(per_ifo_triggers)
    
    # Rank candidates
    ranked_candidates = estimator.rank_candidates(foreground_candidates, background_stats)
    
    return background_stats, ranked_candidates
