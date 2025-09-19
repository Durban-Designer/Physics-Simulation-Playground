"""
Experimental validation framework for superconductor simulations.

This module provides tools to validate simulation results against known
experimental data, ensuring physical accuracy of the simulation platform.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    from .materials import get_material, MATERIALS_DATABASE
    from .materials_citations import get_material_citations
except ImportError:
    from materials import get_material, MATERIALS_DATABASE
    from materials_citations import get_material_citations

@dataclass
class ExperimentalDataPoint:
    """Single experimental measurement."""
    material: str
    temperature: float  # Kelvin
    property_name: str  # e.g., 'resistivity', 'tc', 'gap'
    value: float
    uncertainty: float
    reference: str
    notes: str = ""

@dataclass
class ValidationResult:
    """Result of validation against experimental data."""
    material: str
    property_name: str
    simulated_value: float
    experimental_value: float
    relative_error: float
    within_uncertainty: bool
    validation_status: str  # 'pass', 'fail', 'warning'

# Known experimental data for validation
EXPERIMENTAL_DATABASE = {
    "YBCO": [
        ExperimentalDataPoint(
            material="YBCO",
            temperature=90.0,
            property_name="tc_pristine",
            value=93.0,
            uncertainty=2.0,
            reference="Wu et al., PRL 58, 908 (1987)",
            notes="Original discovery paper"
        ),
        ExperimentalDataPoint(
            material="YBCO",
            temperature=300.0,
            property_name="resistivity_coefficient",
            value=0.8,
            uncertainty=0.2,
            reference="Martin et al., PRL 60, 2194 (1988)",
            notes="Room temperature linear resistivity"
        ),
        ExperimentalDataPoint(
            material="YBCO",
            temperature=100.0,
            property_name="coherence_length",
            value=13.0,
            uncertainty=3.0,
            reference="Klemm et al., PRB 37, 7458 (1988)",
            notes="In-plane coherence length"
        ),
    ],
    
    "BSCCO": [
        ExperimentalDataPoint(
            material="BSCCO",
            temperature=85.0,
            property_name="tc_pristine",
            value=85.0,
            uncertainty=3.0,
            reference="Maeda et al., JJAP 27, L209 (1988)",
            notes="Bi-2212 phase"
        ),
        ExperimentalDataPoint(
            material="BSCCO",
            temperature=110.0,
            property_name="tc_pristine",
            value=110.0,
            uncertainty=3.0,
            reference="Maeda et al., JJAP 27, L209 (1988)",
            notes="Bi-2223 phase"
        ),
    ],
    
    "FeSe": [
        ExperimentalDataPoint(
            material="FeSe",
            temperature=8.5,
            property_name="tc_pristine",
            value=8.5,
            uncertainty=0.5,
            reference="Hsu et al., PNAS 105, 14262 (2008)",
            notes="Bulk FeSe"
        ),
        ExperimentalDataPoint(
            material="FeSe",
            temperature=155.0,
            property_name="nematic_temperature",
            value=155.0,
            uncertainty=5.0,
            reference="Böhmer et al., Nat. Commun. 6, 7911 (2015)",
            notes="Structural transition"
        ),
    ]
}

class ExperimentalValidator:
    """Validate simulation results against experimental data."""
    
    def __init__(self):
        self.experimental_data = EXPERIMENTAL_DATABASE
        self.validation_results = []
    
    def validate_material_property(self, material_name: str, property_name: str, 
                                 simulated_value: float) -> ValidationResult:
        """Validate a single material property against experimental data."""
        
        # Get experimental data
        exp_data = self.experimental_data.get(material_name, [])
        matching_data = [d for d in exp_data if d.property_name == property_name]
        
        if not matching_data:
            return ValidationResult(
                material=material_name,
                property_name=property_name,
                simulated_value=simulated_value,
                experimental_value=np.nan,
                relative_error=np.nan,
                within_uncertainty=False,
                validation_status="no_data"
            )
        
        # Use first matching data point (could be improved)
        exp_point = matching_data[0]
        
        # Calculate relative error
        if exp_point.value != 0:
            relative_error = abs(simulated_value - exp_point.value) / abs(exp_point.value)
        else:
            relative_error = abs(simulated_value - exp_point.value)
        
        # Check if within experimental uncertainty
        within_uncertainty = abs(simulated_value - exp_point.value) <= exp_point.uncertainty
        
        # Determine validation status
        if within_uncertainty:
            status = "pass"
        elif relative_error < 0.2:  # Within 20%
            status = "warning"
        else:
            status = "fail"
        
        result = ValidationResult(
            material=material_name,
            property_name=property_name,
            simulated_value=simulated_value,
            experimental_value=exp_point.value,
            relative_error=relative_error,
            within_uncertainty=within_uncertainty,
            validation_status=status
        )
        
        self.validation_results.append(result)
        return result
    
    def validate_material_database(self) -> Dict[str, List[ValidationResult]]:
        """Validate all materials in the database against experimental data."""
        results = {}
        
        for material_name in MATERIALS_DATABASE.keys():
            material = get_material(material_name)
            results[material_name] = []
            
            # Validate Tc
            tc_result = self.validate_material_property(
                material_name, "tc_pristine", material.tc_pristine
            )
            results[material_name].append(tc_result)
            
            # Validate coherence length if data available
            if hasattr(material, 'coherence_length'):
                coh_result = self.validate_material_property(
                    material_name, "coherence_length", material.coherence_length
                )
                results[material_name].append(coh_result)
            
            # Validate nematic temperature for iron-based materials
            if hasattr(material, 'nematic_temperature') and material.nematic_temperature:
                nem_result = self.validate_material_property(
                    material_name, "nematic_temperature", material.nematic_temperature
                )
                results[material_name].append(nem_result)
        
        return results
    
    def generate_validation_report(self) -> str:
        """Generate a human-readable validation report."""
        report = "Experimental Validation Report\\n"
        report += "=" * 50 + "\\n\\n"
        
        # Validate database
        results = self.validate_material_database()
        
        total_tests = 0
        passed_tests = 0
        warning_tests = 0
        failed_tests = 0
        no_data_tests = 0
        
        for material_name, material_results in results.items():
            report += f"{material_name}:\\n"
            report += "-" * (len(material_name) + 1) + "\\n"
            
            for result in material_results:
                total_tests += 1
                
                if result.validation_status == "pass":
                    passed_tests += 1
                    status_symbol = "✅"
                elif result.validation_status == "warning":
                    warning_tests += 1
                    status_symbol = "⚠️"
                elif result.validation_status == "fail":
                    failed_tests += 1
                    status_symbol = "❌"
                else:
                    no_data_tests += 1
                    status_symbol = "❓"
                
                if not np.isnan(result.experimental_value):
                    report += f"  {status_symbol} {result.property_name}: "
                    report += f"sim={result.simulated_value:.2f}, "
                    report += f"exp={result.experimental_value:.2f}, "
                    report += f"error={result.relative_error:.1%}\\n"
                else:
                    report += f"  {status_symbol} {result.property_name}: "
                    report += f"sim={result.simulated_value:.2f}, no experimental data\\n"
            
            report += "\\n"
        
        # Summary
        report += "Summary:\\n"
        report += f"  Total tests: {total_tests}\\n"
        report += f"  ✅ Passed: {passed_tests} ({passed_tests/total_tests:.1%})\\n"
        report += f"  ⚠️ Warnings: {warning_tests} ({warning_tests/total_tests:.1%})\\n"
        report += f"  ❌ Failed: {failed_tests} ({failed_tests/total_tests:.1%})\\n"
        report += f"  ❓ No data: {no_data_tests} ({no_data_tests/total_tests:.1%})\\n"
        
        return report

def validate_simulation_results(simulation_results: Dict) -> ValidationResult:
    """Validate simulation results against experimental expectations."""
    # Example validation of disorder effects on Tc
    material_name = simulation_results.get('material', 'Unknown')
    simulated_tc = simulation_results.get('enhanced_tc', 0)
    
    validator = ExperimentalValidator()
    
    # Get baseline Tc
    if material_name in MATERIALS_DATABASE:
        material = get_material(material_name)
        baseline_tc = material.tc_pristine
        
        # Check if enhancement is physically reasonable
        enhancement_factor = simulated_tc / baseline_tc if baseline_tc > 0 else 0
        
        if 0.5 <= enhancement_factor <= 3.0:  # Reasonable range
            status = "pass"
        elif 0.2 <= enhancement_factor <= 5.0:  # Possibly reasonable
            status = "warning"
        else:
            status = "fail"
        
        return ValidationResult(
            material=material_name,
            property_name="tc_enhancement",
            simulated_value=enhancement_factor,
            experimental_value=1.0,  # Baseline
            relative_error=abs(enhancement_factor - 1.0),
            within_uncertainty=True if status == "pass" else False,
            validation_status=status
        )
    
    return ValidationResult(
        material=material_name,
        property_name="unknown",
        simulated_value=0.0,
        experimental_value=0.0,
        relative_error=0.0,
        within_uncertainty=False,
        validation_status="no_data"
    )

if __name__ == "__main__":
    # Run validation of material database
    validator = ExperimentalValidator()
    report = validator.generate_validation_report()
    print(report)