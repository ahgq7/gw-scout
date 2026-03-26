"""Runtime patches for third-party library compatibility.

This module applies monkey patches to fix compatibility issues with PyCBC
and other dependencies when running with NumPy 2.x and modern CUDA.
"""


def apply_runtime_patches() -> None:
    """Apply all runtime compatibility patches.
    
    This function patches:
    1. NumPy 2.x + PyCBC ThresholdCluster fix (copy=False → asarray)
    2. PyCUDA memset_d32 type casting fix (float → int for size/data)
    
    Must be called BEFORE any PyCBC imports to take effect.
    """
    import numpy as np
    
    # Patch 1: NumPy 2.x + PyCBC ThresholdCluster fix
    # Patch numpy.array globally BEFORE PyCBC is imported
    if int(np.__version__.split(".")[0]) >= 2:
        _original_array = np.array
        
        def _array_compat(obj, *args, **kwargs):
            # If copy=False is explicitly requested, just remove it
            # This fixes NumPy 2.x ValueError when PyCBC uses copy=False
            if 'copy' in kwargs and kwargs['copy'] is False:
                # Remove copy=False and let numpy use default behavior
                kwargs_copy = kwargs.copy()
                kwargs_copy.pop('copy')
                # Call original array with copy removed - this preserves subclasses
                return _original_array(obj, *args, **kwargs_copy)
            # Otherwise use the original implementation
            return _original_array(obj, *args, **kwargs)
        
        # Globally patch numpy.array
        np.array = _array_compat
    
    # Patch 2: PyCUDA memset_d32 size/data casting fix
    try:
        import pycuda.driver as drv
        _orig_memset = drv.memset_d32
        
        def _memset_d32(dest, data, size):
            """Wrapper that ensures data and size are integers."""
            return _orig_memset(dest, int(data), int(size))
        
        drv.memset_d32 = _memset_d32
        
    except Exception:
        # Silently ignore if PyCUDA is not installed or patch fails
        pass
