"""
Enhanced Equivalent Cycles Model

Improved usage tracking with optional efficiency corrections and SOH compensation.
Maintains backward compatibility - works with minimal inputs if optional parameters unavailable.
Data preprocessing is handled by the platform before reaching the model.
"""

from typing import Dict, Union
import pandas as pd
import numpy as np

from altergo_sdk.boiler_plate import Model, register_model


@register_model("eq_cycles", metadata={
    "category": "Performance",
    "complexity": "Simple", 
    "computational_cost": "Low"
})
class EqCyclesModel(Model):
    """
    Enhanced Equivalent Cycles Estimation Model
    
    Features:
    - Improved usage tracking with coulombic efficiency
    - Optional SOH-based capacity compensation
    - Backward compatible (works with minimal inputs)
    - All metadata defined in model.json manifest
    """

    
    def process(self, data: Dict[str, Union[pd.Series, float]]) -> Dict[str, pd.Series]:
        """
        Calculate enhanced equivalent cycles from current data.
        
        Args:
            data: Dictionary containing:
                - current: Battery current time series (required)
                - capacity: Battery nominal capacity (required)
                - soh: State of Health in % (optional, for capacity compensation)
                
        Returns:
            Dictionary containing equivalent_cycles time series (with logical names)
        """
        
        try:
            # Get required data
            current_data = data["current"]
            battery_capacity = data["capacity"]
            
            # Get optional data (backward compatibility)
            init_eq_cycle = data.get("initial_equivalent_cycles", 0)
            soh_data = data.get("soh", None)
            
            # Handle case where SDK returns empty Series for missing sensors
            if soh_data is not None and hasattr(soh_data, 'empty') and soh_data.empty:
                soh_data = None
            
            # Essential validations only (data comes pre-cleaned from platform)
            if current_data.empty:
                raise ValueError("Current data cannot be empty")
            if battery_capacity <= 0:
                raise ValueError(f"Battery capacity must be positive, got {battery_capacity}")
            
            # Get configuration parameters
            charge_eff = self.config.get("charge_efficiency", 0.98)
            discharge_eff = self.config.get("discharge_efficiency", 0.99)
            enable_efficiency = self.config.get("enable_efficiency_correction", True)
            
            # Calculate equivalent cycles with enhancements
            eq_cycles_result = self._estimate_eq_cycles_enhanced(
                current_data=current_data,
                nominal_capacity=battery_capacity,
                soh_data=soh_data,
                charge_efficiency=charge_eff,
                discharge_efficiency=discharge_eff,
                enable_efficiency=enable_efficiency,
                eq_cycle_init=init_eq_cycle
            )
            
            # Return with logical output name (framework handles blueprint mapping for upload)
            return {
                "equivalent_cycles": eq_cycles_result
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate equivalent cycles: {str(e)}") from e
    
    def _estimate_eq_cycles_enhanced(self, current_data: pd.Series, nominal_capacity: float, 
                                   soh_data: pd.Series = None, charge_efficiency: float = 0.98,
                                   discharge_efficiency: float = 0.99, enable_efficiency: bool = True,
                                   eq_cycle_init: float = 0) -> pd.Series:
        """
        Enhanced equivalent cycles calculation with efficiency and SOH compensation.
        
        Args:
            current_data: Time series of current measurements (A)
            nominal_capacity: Battery nominal capacity (Ah)
            soh_data: Optional SOH time series (%) for capacity compensation
            charge_efficiency: Coulombic efficiency for charging (0.95-0.99)
            discharge_efficiency: Coulombic efficiency for discharging (0.95-0.99)
            enable_efficiency: Enable efficiency corrections
            eq_cycle_init: Initial equivalent cycle count
            
        Returns:
            Series with cumulative equivalent cycles
        """
        # Calculate time differences in hours (optimized)
        time_diff_hours = current_data.index.to_series().diff().dt.total_seconds().values / 3600.0
        time_diff_hours[0] = 0.0
        
        # Get current values (positive = discharge, negative = charge)
        current_values = current_data.values
        
        # Calculate effective capacity with optional SOH compensation
        if soh_data is not None and not soh_data.empty and len(soh_data) > 0:
            # Align SOH data with current data and apply capacity compensation
            soh_aligned = soh_data.reindex(current_data.index, method='ffill')
            soh_aligned = soh_aligned.fillna(100.0)  # Default to 100% if no SOH data
            effective_capacity = nominal_capacity * (soh_aligned.values / 100.0)
        else:
            # Use nominal capacity if no SOH data available or SOH data is empty
            effective_capacity = np.full_like(current_values, nominal_capacity)
        
        # Apply efficiency corrections if enabled
        if enable_efficiency:
            efficiency_factors = np.where(
                current_values >= 0,  # Positive current = discharging
                discharge_efficiency,
                charge_efficiency     # Negative current = charging
            )
        else:
            efficiency_factors = np.ones_like(current_values)
        
        # Calculate charge transferred with efficiency correction
        current_abs = np.abs(current_values)
        charge_transferred = current_abs * time_diff_hours * efficiency_factors
        
        # Calculate equivalent cycles for each step
        # One full cycle = charge + discharge = 2 * effective_capacity
        eq_cycles_step = charge_transferred / (2.0 * effective_capacity)
        
        # Calculate cumulative equivalent cycles
        eq_cycle_cumulative = np.cumsum(eq_cycles_step) + eq_cycle_init
        
        # Create result series
        result = pd.Series(
            eq_cycle_cumulative,
            index=current_data.index,
            name="CumulativeEqCycles"
        )
        
        return result


if __name__ == "__main__":
    """Standalone self-test routine for EqCyclesModel"""
    import json
    from datetime import datetime, timedelta
    
    print("=" * 80)
    print("EqCyclesModel Self-Test Routine")
    print("=" * 80)
    
    # Test helper function
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
    
    # Generate sample data
    def generate_test_data(hours: int = 24, sample_rate_s: int = 60):
        """Generate test current data"""
        timestamps = pd.date_range(
            start='2023-01-01 00:00:00', 
            periods=hours * 3600 // sample_rate_s,
            freq=f'{sample_rate_s}S',
            tz='UTC'
        )
        # Simulate charge/discharge cycles
        t = np.linspace(0, hours * 2 * np.pi / 24, len(timestamps))
        current = 50 * np.sin(t)  # ±50A cycles
        return pd.Series(current, index=timestamps, name='current')
    
    # Test 1: Basic functionality with minimal inputs
    def test_basic_functionality():
        """Test basic eq cycles calculation with minimal inputs"""
        model = EqCyclesModel({
            "initial_eq_cycle": 0,
            "enable_efficiency_correction": False
        })
        
        current = generate_test_data(hours=24)
        data = {
            "current": current,
            "capacity": 100.0  # 100 Ah
        }
        
        result = model.process(data)
        eq_cycles = result["equivalent_cycles"]
        
        # Verify results
        assert isinstance(eq_cycles, pd.Series), "Output should be pandas Series"
        assert len(eq_cycles) == len(current), "Output length should match input"
        assert eq_cycles.iloc[0] >= 0, "First value should be non-negative"
        assert eq_cycles.is_monotonic_increasing, "Eq cycles should be monotonic increasing"
        
        # With 50A amplitude over 24h and 100Ah capacity, expect ~24 cycles
        final_cycles = eq_cycles.iloc[-1]
        expected_cycles = 24.0  # Approximate
        assert abs(final_cycles - expected_cycles) < 5, f"Expected ~{expected_cycles} cycles, got {final_cycles:.2f}"
        
        print(f"  Final equivalent cycles: {final_cycles:.2f}")
        return True
    
    # Test 2: With efficiency corrections
    def test_with_efficiency():
        """Test with efficiency corrections enabled"""
        model = EqCyclesModel({
            "initial_eq_cycle": 10.5,
            "enable_efficiency_correction": True,
            "charge_efficiency": 0.95,
            "discharge_efficiency": 0.98
        })
        
        current = generate_test_data(hours=12)
        data = {
            "current": current,
            "capacity": 50.0
        }
        
        result = model.process(data)
        eq_cycles = result["equivalent_cycles"]
        
        # Should start from initial value
        assert eq_cycles.iloc[0] >= 10.5, "Should start from initial value"
        
        # With efficiency < 1, should accumulate slower than ideal
        final_cycles = eq_cycles.iloc[-1]
        print(f"  With efficiency: {final_cycles:.2f} cycles (starting from 10.5)")
        return True
    
    # Test 3: With SOH compensation
    def test_with_soh():
        """Test with SOH-based capacity compensation"""
        model = EqCyclesModel({
            "initial_eq_cycle": 0,
            "enable_efficiency_correction": True
        })
        
        current = generate_test_data(hours=24)
        # Create degrading SOH data
        soh = pd.Series(
            np.linspace(100, 80, len(current)),  # Degrade from 100% to 80%
            index=current.index,
            name='soh'
        )
        
        data = {
            "current": current,
            "capacity": 100.0,
            "soh": soh
        }
        
        result = model.process(data)
        eq_cycles = result["equivalent_cycles"]
        
        # With degrading SOH, cycles should accumulate faster
        final_cycles = eq_cycles.iloc[-1]
        print(f"  With SOH degradation (100% → 80%): {final_cycles:.2f} cycles")
        assert final_cycles > 24, "Should have more cycles due to reduced effective capacity"
        return True
    
    # Test 4: Edge cases
    def test_edge_cases():
        """Test various edge cases"""
        model = EqCyclesModel({})
        
        # Test with single data point
        current = pd.Series([10.0], index=pd.date_range('2023-01-01', periods=1, tz='UTC'))
        data = {"current": current, "capacity": 100.0}
        result = model.process(data)
        assert result["equivalent_cycles"].iloc[0] == 0, "Single point should have 0 cycles"
        
        # Test with zero current
        current = pd.Series(np.zeros(100), index=pd.date_range('2023-01-01', periods=100, freq='1H', tz='UTC'))
        data = {"current": current, "capacity": 100.0}
        result = model.process(data)
        assert all(result["equivalent_cycles"] == 0), "Zero current should produce zero cycles"
        
        # Test with constant current
        current = pd.Series(np.ones(100) * 20, index=pd.date_range('2023-01-01', periods=100, freq='1H', tz='UTC'))
        data = {"current": current, "capacity": 100.0}
        result = model.process(data)
        expected = 20 * 99 / (2 * 100)  # 20A * 99 hours / (2 * 100Ah)
        assert abs(result["equivalent_cycles"].iloc[-1] - expected) < 0.1, "Constant current calculation incorrect"
        
        print("  All edge cases handled correctly")
        return True
    
    # Test 5: Error handling
    def test_error_handling():
        """Test error handling for invalid inputs"""
        model = EqCyclesModel({})
        
        # Test empty current data
        try:
            current = pd.Series([], index=pd.DatetimeIndex([]), name='current')
            data = {"current": current, "capacity": 100.0}
            model.process(data)
            return False  # Should have raised error
        except ValueError as e:
            assert "empty" in str(e).lower()
        
        # Test invalid capacity
        try:
            current = generate_test_data(hours=1)
            data = {"current": current, "capacity": -100.0}
            model.process(data)
            return False  # Should have raised error
        except ValueError as e:
            assert "positive" in str(e).lower()
        
        # Test missing required data
        try:
            data = {"capacity": 100.0}  # Missing current
            model.process(data)
            return False  # Should have raised error
        except Exception:
            pass
        
        print("  Error handling works correctly")
        return True
    
    # Test 6: Incremental mode initialization
    def test_incremental_mode():
        """Test incremental mode with initial value seeding"""
        # Simulate incremental mode by setting equivalent_cycles_initial_value
        model = EqCyclesModel({
            "equivalent_cycles_initial_value": 50.0,  # Seed value from previous run
            "initial_eq_cycle": 0  # This should be overridden
        })
        
        current = generate_test_data(hours=1)
        data = {
            "current": current,
            "capacity": 100.0
        }
        
        result = model.process(data)
        eq_cycles = result["equivalent_cycles"]
        
        # Should start from the seeded value
        assert eq_cycles.iloc[0] >= 50.0, "Should use incremental seed value"
        print(f"  Incremental mode: Started from {eq_cycles.iloc[0]:.2f}")
        return True
    
    # Test 7: Performance test
    def test_performance():
        """Test performance with large dataset"""
        import time
        
        # Generate 1 week of data at 1-second resolution
        print("  Generating 1 week of 1-second data...")
        current = generate_test_data(hours=168, sample_rate_s=1)  # 604,800 points
        
        # Add some SOH data too
        soh = pd.Series(
            np.linspace(100, 95, len(current)),
            index=current.index,
            name='soh'
        )
        
        model = EqCyclesModel({
            "enable_efficiency_correction": True
        })
        
        data = {
            "current": current,
            "capacity": 100.0,
            "soh": soh
        }
        
        start_time = time.time()
        result = model.process(data)
        elapsed = time.time() - start_time
        
        print(f"  Processed {len(current):,} points in {elapsed:.3f} seconds")
        print(f"  Rate: {len(current)/elapsed:,.0f} points/second")
        
        assert elapsed < 5.0, f"Processing took too long: {elapsed:.3f}s"
        return True
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Efficiency Corrections", test_with_efficiency),
        ("SOH Compensation", test_with_soh),
        ("Edge Cases", test_edge_cases),
        ("Error Handling", test_error_handling),
        ("Incremental Mode", test_incremental_mode),
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
    
    # Exit with appropriate code
    import sys
    sys.exit(0 if failed == 0 else 1)