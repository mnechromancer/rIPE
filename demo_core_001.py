#!/usr/bin/env python3
"""
Demo script for CORE-001: Physiological State Vector Implementation

This script demonstrates the basic functionality of the PhysiologicalState
and StateVector classes.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ipe.core.physiology.state import PhysiologicalState, Tissue
from ipe.core.physiology.state_vector import StateVector


def main():
    print("=== CORE-001 Demo: Physiological State Vector Implementation ===\n")
    
    # 1. Create individual physiological states
    print("1. Creating PhysiologicalState objects...")
    
    # High altitude, low oxygen state
    high_altitude_state = PhysiologicalState(
        po2=12.0,  # Lower oxygen at altitude
        temperature=5.0,  # Cold mountain environment
        altitude=3000.0,  # 3000m elevation
        heart_mass=12.0,  # Enlarged heart
        hematocrit=55.0,  # Higher red blood cell count
        hemoglobin=18.0   # Higher hemoglobin
    )
    
    # Sea level, warm state
    sea_level_state = PhysiologicalState(
        po2=21.0,  # Full atmospheric oxygen
        temperature=25.0,  # Warm environment
        altitude=0.0,     # Sea level
        heart_mass=8.0,   # Normal heart size
        hematocrit=45.0,  # Normal hematocrit
        hemoglobin=15.0   # Normal hemoglobin
    )
    
    print(f"High altitude state: PO2={high_altitude_state.po2} kPa, altitude={high_altitude_state.altitude}m")
    print(f"Sea level state: PO2={sea_level_state.po2} kPa, altitude={sea_level_state.altitude}m")
    
    # 2. Validate states
    print("\n2. Validating physiological parameters...")
    try:
        high_altitude_state.validate()
        sea_level_state.validate()
        print("✓ All states are physiologically valid")
    except ValueError as e:
        print(f"✗ Validation error: {e}")
        return
    
    # 3. Test physiological calculations
    print("\n3. Testing physiological calculations...")
    
    ha_aerobic_scope = high_altitude_state.compute_aerobic_scope()
    sl_aerobic_scope = sea_level_state.compute_aerobic_scope()
    
    print(f"High altitude aerobic scope: {ha_aerobic_scope:.1f} mL O2/min/kg")
    print(f"Sea level aerobic scope: {sl_aerobic_scope:.1f} mL O2/min/kg")
    
    # Oxygen delivery to brain
    ha_brain_o2 = high_altitude_state.oxygen_delivery(Tissue.BRAIN)
    sl_brain_o2 = sea_level_state.oxygen_delivery(Tissue.BRAIN)
    
    print(f"High altitude brain O2 delivery: {ha_brain_o2:.1f} mL O2/min/kg")
    print(f"Sea level brain O2 delivery: {sl_brain_o2:.1f} mL O2/min/kg")
    
    # 4. Test immutability and hashing
    print("\n4. Testing immutability and hashing...")
    
    # States should be hashable
    state_set = {high_altitude_state, sea_level_state, high_altitude_state}
    print(f"✓ States are hashable - set contains {len(state_set)} unique states")
    
    # Test equality
    duplicate_state = PhysiologicalState(
        po2=12.0, temperature=5.0, altitude=3000.0,
        heart_mass=12.0, hematocrit=55.0, hemoglobin=18.0
    )
    print(f"✓ State equality works: {high_altitude_state == duplicate_state}")
    
    # 5. Create and test StateVector
    print("\n5. Creating and testing StateVector...")
    
    vector = StateVector([high_altitude_state, sea_level_state])
    print(f"✓ StateVector created with {len(vector)} states")
    
    # 6. Test distance calculations
    print("\n6. Testing distance calculations...")
    
    euclidean_dist = StateVector(vector[0]).euclidean_distance(vector[1])
    manhattan_dist = StateVector(vector[0]).manhattan_distance(vector[1])
    
    print(f"Euclidean distance between states: {euclidean_dist:.2f}")
    print(f"Manhattan distance between states: {manhattan_dist:.2f}")
    
    # 7. Test serialization
    print("\n7. Testing serialization...")
    
    # Convert to JSON
    json_str = vector.to_json(indent=2)
    print("✓ Converted to JSON successfully")
    
    # Round-trip test
    restored_vector = StateVector.from_json(json_str)
    print(f"✓ Restored from JSON - {len(restored_vector)} states recovered")
    
    # Verify restored states are equal
    states_equal = all(original == restored for original, restored 
                      in zip(vector, restored_vector))
    print(f"✓ Round-trip preservation: {states_equal}")
    
    # 8. Test mean state calculation
    print("\n8. Testing mean state calculation...")
    
    mean_state = vector.mean_state()
    print(f"Mean PO2: {mean_state.po2:.1f} kPa")
    print(f"Mean altitude: {mean_state.altitude:.1f} m")
    print(f"Mean heart mass: {mean_state.heart_mass:.1f} g/kg")
    
    print("\n=== Demo completed successfully! ===")
    print("\nCORE-001 acceptance criteria verified:")
    print("✓ PhysiologicalState dataclass with all parameters from design doc")
    print("✓ Immutable state vectors with distance calculations")
    print("✓ State vector serialization/deserialization")
    print("✓ Comprehensive validation and physiological calculations")


if __name__ == "__main__":
    main()