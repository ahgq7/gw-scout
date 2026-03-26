"""
GWOSC segment database interface for data quality validation.
Uses official GWOSC timeline APIs to query science-mode segments.
"""
from __future__ import annotations

import logging
from typing import List, Tuple, Optional

LOG = logging.getLogger(__name__)


def get_science_segments(ifo: str, start_gps: float, end_gps: float) -> List[Tuple[float, float]]:
    """
    Query GWOSC for science-mode segments for specified IFO and time range.
    
    Args:
        ifo: Detector name (e.g., 'H1', 'L1')
        start_gps: Start GPS time
        end_gps: End GPS time
    
    Returns:
        List of (start, end) tuples representing valid science segments
    """
    try:
        from gwosc.timeline import get_segments
        
        # Query for DATA segments (basic science mode)
        timeline_name = f"{ifo}_DATA"
        segments = get_segments(timeline_name, start_gps, end_gps)
        
        if segments:
            LOG.info("Found %d science segments for %s in GPS %.1f-%.1f", 
                    len(segments), ifo, start_gps, end_gps)
            return segments
        else:
            LOG.warning("No science segments found for %s in GPS %.1f-%.1f", 
                       ifo, start_gps, end_gps)
            return []
            
    except ImportError:
        LOG.warning("gwosc.timeline not available; cannot validate segments")
        return []
    except Exception as exc:
        LOG.warning("Failed to query segments for %s: %s", ifo, exc)
        return []


def is_science_time(ifo: str, gps_time: float, tolerance: float = 1.0) -> bool:
    """
    Check if a specific GPS time is in a science segment.
    
    Args:
        ifo: Detector name
        gps_time: GPS time to check
        tolerance: Time window around gps_time to query (seconds)
    
    Returns:
        True if time is in a science segment
    """
    segments = get_science_segments(ifo, gps_time - tolerance, gps_time + tolerance)
    
    for seg_start, seg_end in segments:
        if seg_start <= gps_time <= seg_end:
            return True
    
    return False


def get_overlap_duration(ifo: str, start_gps: float, end_gps: float) -> float:
    """
    Calculate total duration of science-mode overlap in requested interval.
    
    Args:
        ifo: Detector name
        start_gps: Start of requested interval
        end_gps: End of requested interval
    
    Returns:
        Total duration (seconds) of science-mode data in interval
    """
    segments = get_science_segments(ifo, start_gps, end_gps)
    
    total_duration = 0.0
    for seg_start, seg_end in segments:
        # Clip segment to requested interval
        overlap_start = max(seg_start, start_gps)
        overlap_end = min(seg_end, end_gps)
        
        if overlap_end > overlap_start:
            total_duration += (overlap_end - overlap_start)
    
    return total_duration


def find_good_time_range(
    ifo: str, 
    center_gps: float, 
    min_duration: float = 256.0,
    max_search: float = 3600.0
) -> Optional[Tuple[float, float]]:
    """
    Find a continuous science-mode interval near a reference time.
    
    Args:
        ifo: Detector name
        center_gps: Center GPS time to search around
        min_duration: Minimum required duration (seconds)
        max_search: Maximum search window around center (seconds)
    
    Returns:
        (start, end) tuple of good interval, or None if not found
    """
    search_start = center_gps - max_search / 2
    search_end = center_gps + max_search / 2
    
    segments = get_science_segments(ifo, search_start, search_end)
    
    # Find longest segment that meets minimum duration
    best_segment = None
    best_duration = 0.0
    
    for seg_start, seg_end in segments:
        duration = seg_end - seg_start
        if duration >= min_duration and duration > best_duration:
            best_segment = (seg_start, seg_end)
            best_duration = duration
    
    if best_segment:
        LOG.info("Found %s science segment: GPS %.1f-%.1f (%.1f sec)", 
                ifo, best_segment[0], best_segment[1], best_duration)
    else:
        LOG.warning("No science segment >= %.1f sec found for %s near GPS %.1f", 
                   min_duration, ifo, center_gps)
    
    return best_segment


def get_duty_cycle(ifo: str, start_gps: float, end_gps: float) -> float:
    """
    Calculate duty cycle (fraction of time in science mode).
    
    Args:
        ifo: Detector name
        start_gps: Start GPS time
        end_gps: End GPS time
    
    Returns:
        Duty cycle (0.0 to 1.0)
    """
    requested_duration = end_gps - start_gps
    if requested_duration <= 0:
        return 0.0
    
    science_duration = get_overlap_duration(ifo, start_gps, end_gps)
    return science_duration / requested_duration
