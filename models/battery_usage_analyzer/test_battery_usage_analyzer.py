"""
Test suite for the Battery Usage Analyzer model.

This module contains comprehensive tests for the BatteryUsageAnalyzer model,
including basic functionality, configuration parameters, edge cases, error handling,
and performance tests.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

try:
    from .battery_usage_analyzer import BatteryUsageAnalyzer
except ImportError:
    # When running as standalone module
    from battery_usage_analyzer import BatteryUsageAnalyzer


def generate_test_data(hours: int = 24, sample_rate_s: int = 60):
    """Generate test battery data for testing purposes."""
    timestamps = pd.date_range(
        start='2023-01-01 00:00:00', 
        periods=hours * 3600 // sample_rate_s,
        freq=f'{sample_rate_s}S',
        tz='UTC'
    )
    n = len(timestamps)
    
    # Simulate charge/discharge cycles
    t = np.linspace(0, hours * 2 * np.pi / 24, n)
    current = 50 * np.sin(t)  # ±50A cycles
    
    # Simulate SoC cycling between 20-80%
    soc = 50 + 25 * np.sin(t - np.pi/2)  # 25-75% range
    
    # Simulate voltage response (simplified)
    v_base = 3.7
    v_cell_min = pd.Series(v_base + 0.2 * (soc/100 - 0.5) + 0.1 * np.random.normal(0, 0.1, n), 
                           index=timestamps, name='v_cell_min')
    v_cell_max = v_cell_min + 0.05 + 0.02 * np.random.normal(0, 0.1, n)
    
    # Simulate temperature variation
    t_base = 25
    t_cell_min = pd.Series(t_base + 5 * np.sin(t/4) + 2 * np.random.normal(0, 1, n), 
                           index=timestamps, name='t_cell_min')
    t_cell_max = t_cell_min + 5 + np.random.normal(0, 1, n)
    
    return {
        'current': pd.Series(current, index=timestamps, name='current'),
        'soc': pd.Series(soc, index=timestamps, name='soc'),
        'v_cell_min': v_cell_min.clip(3.0, 4.2),
        'v_cell_max': v_cell_max.clip(3.0, 4.2),
        't_cell_min': t_cell_min.clip(-10, 60),
        't_cell_max': t_cell_max.clip(-10, 60)
    }


def test_basic_functionality():
    """Test basic segmentation with minimal configuration"""
    model = BatteryUsageAnalyzer({})
    
    data = generate_test_data(hours=6)
    
    result = model.process(data)
    
    # Verify all expected outputs are present
    expected_outputs = ["seg_l0_id", "mode", "boundary_l1", "seg_l1_id", "phase", "seg_l2_id", "segment_summary"]
    for output in expected_outputs:
        assert output in result, f"Missing output: {output}"
        if output == "segment_summary":
            assert isinstance(result[output], pd.DataFrame), f"Output {output} should be pandas DataFrame"
        else:
            assert isinstance(result[output], pd.Series), f"Output {output} should be pandas Series"
            assert len(result[output]) == len(data['current']), f"Output {output} length mismatch"
    
    # Check segment IDs are monotonic increasing
    assert result["seg_l0_id"].is_monotonic_increasing, "L0 segment IDs should be monotonic"
    assert result["seg_l1_id"].is_monotonic_increasing, "L1 segment IDs should be monotonic"
    assert result["seg_l2_id"].is_monotonic_increasing, "L2 segment IDs should be monotonic"
    
    # Check mode labels are valid
    valid_modes = {"charge", "discharge", "idle"}
    assert set(result["mode"].unique()).issubset(valid_modes), "Invalid mode labels"
    
    # Check phase labels are valid
    valid_phases = {"rest", "cv_charge", "cc_charge", "cc_discharge", "discharge", "idle"}
    assert set(result["phase"].unique()).issubset(valid_phases), "Invalid phase labels"
    
    print(f"  Detected {result['seg_l0_id'].iloc[-1]} L0 segments")
    print(f"  Detected {result['seg_l1_id'].iloc[-1]} L1 segments")
    print(f"  Detected {result['seg_l2_id'].iloc[-1]} L2 segments")
    print(f"  Modes found: {set(result['mode'].unique())}")
    print(f"  Phases found: {set(result['phase'].unique())}")
    
    # Verify DataFrame content
    df = result['segment_summary']
    print(f"  DataFrame shape: {df.shape}")
    assert 'segment_id' in df.columns, "DataFrame should have segment_id column"
    assert 'mode' in df.columns, "DataFrame should have mode column"
    assert 'duration_s' in df.columns, "DataFrame should have duration_s column"
    assert 'mode_transitions_filtered' in df.columns, "DataFrame should have mode_transitions_filtered column"
    
    # Check that consolidation worked - no consecutive segments with same mode
    if len(df) > 1:
        consecutive_same_mode = (df['mode'].shift(1) == df['mode']).any()
        assert not consecutive_same_mode, "DataFrame should not have consecutive segments with same mode"
    
    print(f"  Consolidated to {len(df)} segments with no consecutive same-mode segments")
    
    return True


def test_configuration():
    """Test with custom configuration parameters"""
    model = BatteryUsageAnalyzer({
        "window_s": 10.0,
        "i_rest_th_A": 1.0,
        "i_charge_on_A": 5.0,
        "i_discharge_on_A": 5.0,
        "min_state_duration_s": 120.0,
        "min_seg_duration_s": 180.0,
        "min_phase_duration_s": 90.0,
        "change_quantile": 0.95,
        "enable_rainflow": True,
        "rf_min_range_soc_pct": 10.0
    })
    
    data = generate_test_data(hours=12)
    result = model.process(data)
    
    # With stricter thresholds, should have fewer segments
    print(f"  With custom config - L0 segments: {result['seg_l0_id'].iloc[-1]}")
    print(f"  With custom config - L1 segments: {result['seg_l1_id'].iloc[-1]}")
    print(f"  With custom config - L2 segments: {result['seg_l2_id'].iloc[-1]}")
    
    return True


def test_edge_cases():
    """Test various edge cases"""
    model = BatteryUsageAnalyzer({})
    
    # Test with single data point
    timestamps = pd.date_range('2023-01-01', periods=1, tz='UTC')
    data = {
        'current': pd.Series([0.0], index=timestamps),
        'soc': pd.Series([50.0], index=timestamps),
        'v_cell_min': pd.Series([3.7], index=timestamps),
        'v_cell_max': pd.Series([3.7], index=timestamps),
        't_cell_min': pd.Series([25.0], index=timestamps),
        't_cell_max': pd.Series([25.0], index=timestamps)
    }
    result = model.process(data)
    assert all(result[key].iloc[0] == 1 for key in ["seg_l0_id", "seg_l1_id", "seg_l2_id"])
    
    # Test with constant values
    timestamps = pd.date_range('2023-01-01', periods=100, freq='1min', tz='UTC')
    data = {
        'current': pd.Series(np.zeros(100), index=timestamps),
        'soc': pd.Series(np.full(100, 50.0), index=timestamps),
        'v_cell_min': pd.Series(np.full(100, 3.7), index=timestamps),
        'v_cell_max': pd.Series(np.full(100, 3.7), index=timestamps),
        't_cell_min': pd.Series(np.full(100, 25.0), index=timestamps),
        't_cell_max': pd.Series(np.full(100, 25.0), index=timestamps)
    }
    result = model.process(data)
    # Should have minimal segmentation with constant data
    assert result["seg_l0_id"].iloc[-1] <= 2, "Constant data should have minimal segments"
    
    print("  All edge cases handled correctly")
    return True


def test_error_handling():
    """Test error handling for invalid inputs"""
    model = BatteryUsageAnalyzer({})
    
    # Test missing required data
    try:
        data = {"current": pd.Series([1, 2, 3])}  # Missing other required inputs
        model.process(data)
        return False  # Should have raised error
    except ValueError as e:
        assert "Missing required input" in str(e)
    
    # Test empty data
    try:
        timestamps = pd.DatetimeIndex([])
        data = {
            'current': pd.Series([], index=timestamps),
            'soc': pd.Series([], index=timestamps),
            'v_cell_min': pd.Series([], index=timestamps),
            'v_cell_max': pd.Series([], index=timestamps),
            't_cell_min': pd.Series([], index=timestamps),
            't_cell_max': pd.Series([], index=timestamps)
        }
        model.process(data)
        return False  # Should have raised error
    except ValueError as e:
        assert "empty" in str(e).lower()
    
    print("  Error handling works correctly")
    return True


def test_performance():
    """Test performance with large dataset"""
    # Generate 1 week of data at 10-second resolution
    print("  Generating 1 week of 10-second data...")
    data = generate_test_data(hours=168, sample_rate_s=10)  # ~60,480 points
    
    model = BatteryUsageAnalyzer({
        "enable_rainflow": True
    })
    
    start_time = time.time()
    result = model.process(data)
    elapsed = time.time() - start_time
    
    n_points = len(data['current'])
    print(f"  Processed {n_points:,} points in {elapsed:.3f} seconds")
    print(f"  Rate: {n_points/elapsed:,.0f} points/second")
    print(f"  Final segments - L0: {result['seg_l0_id'].iloc[-1]}, L1: {result['seg_l1_id'].iloc[-1]}, L2: {result['seg_l2_id'].iloc[-1]}")
    
    assert elapsed < 10.0, f"Processing took too long: {elapsed:.3f}s"
    return True


def run_test(test_name: str, test_func):
    """Run a test and report results"""
    try:
        print(f"\n{test_name}...")
        result = test_func()
        if result:
            print(f"✓ PASSED: {test_name}")
            return True
        else:
            print(f"✗ FAILED: {test_name}")
            return False
    except Exception as e:
        print(f"✗ ERROR in {test_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and return summary results"""
    print("=" * 80)
    print("BatteryUsageAnalyzer Test Suite")
    print("=" * 80)
    
    # Test definitions
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Configuration Parameters", test_configuration),
        ("Edge Cases", test_edge_cases),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        if run_test(test_name, test_func):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 80)
    print(f"Test Summary: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 80)
    
    return passed, failed


if __name__ == "__main__":
    """Standalone test runner"""
    passed, failed = run_all_tests()
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if failed == 0 else 1)
