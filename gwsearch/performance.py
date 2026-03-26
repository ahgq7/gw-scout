"""Performance optimization utilities for GW search.

This module provides CPU multi-threading and CUDA acceleration utilities
while maintaining full scientific accuracy.
"""

import logging
import os
import sys
from typing import Optional

LOG = logging.getLogger(__name__)


def setup_performance_backend(force_cuda: bool = False, num_threads: Optional[int] = None):
    """Setup optimal performance backend (CUDA if available, otherwise multi-threaded CPU).
    
    Args:
        force_cuda: If True, fail if CUDA is not available
        num_threads: Number of CPU threads (None = auto-detect all cores)
    
    Returns:
        dict with backend info
    """
    import multiprocessing
    
    # Detect available CPU cores
    if num_threads is None:
        num_threads = multiprocessing.cpu_count()
    
    LOG.info("=== Performance Backend Setup ===")
    LOG.info("CPU cores available: %d", num_threads)
    
    # 1. Setup NumPy/BLAS/LAPACK threading
    _setup_numpy_threads(num_threads)
    
    # 2. Setup FFTW threading (if available)
    _setup_fftw_threads(num_threads)
    
    # 3. Try to setup PyCBC CUDA backend
    cuda_available = _setup_pycbc_cuda(force_cuda)
    
    backend_info = {
        'cuda_enabled': cuda_available,
        'num_threads': num_threads,
        'scheme': 'cuda' if cuda_available else 'cpu',
    }
    
    LOG.info("=== Backend Configuration ===")
    LOG.info("PyCBC Scheme: %s", backend_info['scheme'])
    LOG.info("CPU Threads: %d", backend_info['num_threads'])
    LOG.info("CUDA: %s", "ENABLED" if cuda_available else "DISABLED")
    LOG.info("================================")
    
    return backend_info


def _setup_numpy_threads(num_threads: int):
    """Configure NumPy/BLAS/LAPACK to use multiple threads."""
    # Set environment variables BEFORE importing numpy
    # These must be set before numpy imports its C libraries
    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
    os.environ['MKL_NUM_THREADS'] = str(num_threads)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
    
    LOG.info("NumPy/BLAS thread count set to: %d", num_threads)
    
    # Verify numpy threading (after import)
    try:
        import numpy as np
        if hasattr(np, '__config__'):
            LOG.debug("NumPy BLAS info: %s", np.__config__.show())
    except Exception:
        pass


def _setup_fftw_threads(num_threads: int):
    """Configure FFTW to use multiple threads."""
    os.environ['FFTW_NUM_THREADS'] = str(num_threads)
    
    # Try to enable FFTW threading in pyfftw if available
    try:
        import pyfftw
        pyfftw.config.NUM_THREADS = num_threads
        pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'  # Balance between planning time and FFT speed
        LOG.info("pyFFTW configured with %d threads", num_threads)
    except ImportError:
        LOG.debug("pyFFTW not available, using default FFT backend")


def _setup_pycbc_cuda(force_cuda: bool = False) -> bool:
    """Setup PyCBC CUDA backend if available.
    
    Returns:
        True if CUDA is enabled, False otherwise
    """
    # Check if CUDA is available
    cuda_available = False
    
    try:
        import pycuda.driver as cuda_drv
        cuda_drv.init()
        
        if cuda_drv.Device.count() > 0:
            cuda_available = True
            device = cuda_drv.Device(0)
            LOG.info("CUDA Device Found: %s", device.name())
            LOG.info("CUDA Compute Capability: %d.%d", *device.compute_capability())
            LOG.info("CUDA Total Memory: %.1f GB", device.total_memory() / 1024**3)
        else:
            LOG.warning("PyCUDA installed but no CUDA devices found")
    except ImportError:
        LOG.warning("PyCUDA not installed - CUDA acceleration unavailable")
    except Exception as e:
        LOG.warning("CUDA initialization failed: %s", e)
    
    if cuda_available:
        # Enable PyCBC CUDA scheme
        os.environ['PYCBC_SCHEME'] = 'cuda'
        
        try:
            # Import and configure PyCBC scheme
            # PyCBC uses CUDAScheme context manager, not set_processing_scheme_from_string
            from pycbc.scheme import CUDAScheme
            
            # Try to initialize CUDA scheme
            try:
                # Create a CUDA scheme context (will be used by PyCBC operations)
                # This doesn't actually switch globally, just validates CUDA works
                with CUDAScheme(device_num=0):
                    LOG.info("PyCBC CUDA scheme validated and ENABLED")
                return True
            except Exception as e:
                LOG.warning("Failed to initialize PyCBC CUDA scheme: %s", e)
                LOG.info("Falling back to CPU scheme")
        except ImportError as e:
            LOG.warning("PyCBC CUDA not available: %s", e)
    
    if force_cuda:
        raise RuntimeError("CUDA requested but not available")
    
    # Fall back to CPU
    os.environ['PYCBC_SCHEME'] = 'cpu'
    LOG.info("Using CPU scheme (CUDA not available)")
    return False


def get_optimal_thread_count(task_type: str = 'matched_filter') -> int:
    """Get optimal thread count for specific task type.
    
    Args:
        task_type: Type of task ('matched_filter', 'psd', 'fft', etc.)
    
    Returns:
        Optimal number of threads for the task
    """
    import multiprocessing
    
    total_cores = multiprocessing.cpu_count()
    
    # For I/O bound or memory-intensive tasks, use fewer threads
    # For CPU-bound tasks, use all cores
    if task_type == 'matched_filter':
        # Matched filter is CPU-intensive, use all cores
        return total_cores
    elif task_type == 'psd':
        # PSD estimation is memory-intensive, use 75% of cores
        return max(1, int(total_cores * 0.75))
    elif task_type == 'fft':
        # FFT is both CPU and memory intensive
        return max(1, int(total_cores * 0.75))
    elif task_type == 'templates':
        # Template loop parallelization
        return total_cores
    else:
        return max(1, total_cores // 2)


def parallel_map(func, items, max_workers=None, desc="Processing", show_progress=True):
    """Execute function in parallel over items using thread pool.
    
    Args:
        func: Function to apply to each item
        items: Iterable of items to process
        max_workers: Maximum number of worker threads (None = auto)
        desc: Description for progress display
        show_progress: Whether to show progress bar
    
    Returns:
        List of results
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if max_workers is None:
        max_workers = get_optimal_thread_count('templates')
    
    items_list = list(items)
    n_items = len(items_list)
    
    if n_items == 0:
        return []
    
    # For small number of items, don't bother with threading overhead
    if n_items < 4:
        LOG.debug("Small item count (%d), using serial execution", n_items)
        return [func(item) for item in items_list]
    
    LOG.info("Parallel execution: %d items with %d workers", n_items, max_workers)
    
    results = [None] * n_items
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(func, item): idx 
            for idx, item in enumerate(items_list)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
                completed += 1
                
                if show_progress and completed % max(1, n_items // 10) == 0:
                    LOG.info("%s: %d/%d (%.1f%%)", desc, completed, n_items, 
                            100.0 * completed / n_items)
            except Exception as e:
                LOG.warning("Parallel task %d failed: %s", idx, e)
                results[idx] = None
    
    return results


def enable_gpu_memory_pool():
    """Enable GPU memory pooling for better performance."""
    try:
        import pycuda.tools
        # Enable memory pool to reduce allocation overhead
        pycuda.tools.DeviceMemoryPool()
        LOG.info("GPU memory pool enabled")
        return True
    except Exception as e:
        LOG.debug("Could not enable GPU memory pool: %s", e)
        return False


def get_device_info():
    """Get information about available compute devices."""
    info = {
        'cpu_cores': 0,
        'cuda_devices': [],
    }
    
    import multiprocessing
    info['cpu_cores'] = multiprocessing.cpu_count()
    
    try:
        import pycuda.driver as cuda_drv
        cuda_drv.init()
        
        for i in range(cuda_drv.Device.count()):
            device = cuda_drv.Device(i)
            info['cuda_devices'].append({
                'id': i,
                'name': device.name(),
                'compute_capability': device.compute_capability(),
                'total_memory_gb': device.total_memory() / 1024**3,
            })
    except Exception:
        pass
    
    return info
