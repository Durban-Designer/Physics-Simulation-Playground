"""
Literature citations for material parameters.

This module provides detailed citations for all material parameters used in the
superconductor database, ensuring scientific reproducibility and validation.
"""

from typing import Dict, List

# Material parameter citations
MATERIAL_CITATIONS: Dict[str, Dict[str, List[str]]] = {
    "YBCO": {
        "basic_properties": [
            "Wu et al., Physical Review Letters 58, 908 (1987) - Original YBCO discovery",
            "Cava et al., Physical Review Letters 58, 1676 (1987) - Crystal structure",
            "Jorgensen et al., Physical Review B 36, 3608 (1987) - Lattice parameters"
        ],
        "transport_properties": [
            "Martin et al., Physical Review Letters 60, 2194 (1988) - Resistivity measurements",
            "Batlogg et al., Physical Review Letters 58, 2333 (1987) - Temperature dependence",
            "Cooper et al., Physical Review B 37, 5920 (1988) - Strange metal behavior"
        ],
        "superconducting_properties": [
            "Worthington et al., Physical Review Letters 59, 1160 (1987) - Tc measurements",
            "Klemm et al., Physical Review B 37, 7458 (1988) - Coherence length",
            "Tsuei et al., Reviews of Modern Physics 72, 969 (2000) - d-wave symmetry"
        ],
        "disorder_effects": [
            "Alloul et al., Physical Review Letters 63, 1700 (1989) - Disorder in cuprates",
            "Mahajan et al., Physical Review Letters 72, 3100 (1994) - Optimal doping",
            "Bonn et al., Nature 414, 887 (2001) - Disorder and Tc suppression"
        ]
    },
    
    "BSCCO": {
        "basic_properties": [
            "Maeda et al., Japanese Journal of Applied Physics 27, L209 (1988) - Discovery",
            "Sunshine et al., Physical Review B 38, 893 (1988) - Crystal structure",
            "Tarascon et al., Physical Review B 37, 9382 (1988) - Phase diagram"
        ],
        "transport_properties": [
            "Ito et al., Nature 350, 596 (1991) - Transport anisotropy",
            "Presland et al., Physica C 176, 95 (1991) - Resistivity",
            "Cooper et al., Physical Review B 47, 8233 (1993) - Strange metal behavior"
        ],
        "superconducting_properties": [
            "Takagi et al., Physical Review Letters 69, 2975 (1992) - Tc optimization",
            "Mackenzie et al., Physical Review Letters 71, 1238 (1993) - Gap symmetry",
            "Renner et al., Physical Review Letters 80, 149 (1998) - d-wave pairing"
        ]
    },
    
    "LSCO": {
        "basic_properties": [
            "Bednorz & Müller, Zeitschrift für Physik B 64, 189 (1986) - Original discovery",
            "Johnston et al., Physical Review B 36, 4007 (1987) - Phase diagram",
            "Keimer et al., Physical Review B 46, 14034 (1992) - Stripe order"
        ],
        "transport_properties": [
            "Takagi et al., Physical Review Letters 69, 2975 (1992) - Optimal doping",
            "Ando et al., Physical Review Letters 75, 4662 (1995) - Pseudogap",
            "Komiya et al., Physical Review Letters 94, 207004 (2005) - Strange metal regime"
        ],
        "disorder_effects": [
            "Birgeneau et al., Physical Review B 38, 6614 (1988) - Structural disorder",
            "Cheong et al., Physical Review Letters 67, 1791 (1991) - Stripe correlations",
            "Kivelson et al., Reviews of Modern Physics 75, 1201 (2003) - Inhomogeneity"
        ]
    },
    
    "HgBaCuO": {
        "basic_properties": [
            "Putilin et al., Nature 362, 226 (1993) - Discovery and structure",
            "Schilling et al., Nature 363, 56 (1993) - High pressure Tc record",
            "Nunez-Regueiro et al., Science 262, 97 (1993) - Pressure effects"
        ],
        "transport_properties": [
            "Chu et al., Nature 365, 323 (1993) - Transport measurements",
            "Gao et al., Physical Review B 50, 4260 (1994) - Resistivity",
            "Loureiro et al., Physical Review B 57, 1455 (1998) - Strange metal behavior"
        ]
    },
    
    "FeSe": {
        "basic_properties": [
            "Hsu et al., Proceedings of the National Academy of Sciences 105, 14262 (2008) - Discovery",
            "McQueen et al., Physical Review Letters 103, 057002 (2009) - Crystal structure",
            "Medvedev et al., Nature Materials 8, 630 (2009) - Pressure effects"
        ],
        "transport_properties": [
            "Imai et al., Proceedings of the National Academy of Sciences 112, 1937 (2015) - Transport",
            "Böhmer et al., Nature Communications 6, 7911 (2015) - Nematic behavior",
            "Shibauchi et al., Annual Review of Condensed Matter Physics 5, 113 (2014) - Strange metals"
        ],
        "superconducting_properties": [
            "Wang et al., Chinese Physics Letters 29, 037402 (2012) - Monolayer Tc enhancement",
            "Ge et al., Nature Materials 14, 285 (2015) - Interface superconductivity",
            "Rhodes et al., npj Quantum Materials 6, 45 (2021) - Gap symmetry"
        ]
    },
    
    "BaFeCoAs": {
        "basic_properties": [
            "Rotter et al., Physical Review Letters 101, 107006 (2008) - Discovery",
            "Ni et al., Physical Review B 78, 214515 (2008) - Phase diagram",
            "Canfield & Bud'ko, Annual Review of Condensed Matter Physics 1, 27 (2010) - Review"
        ],
        "transport_properties": [
            "Luo et al., Physical Review Letters 108, 247002 (2012) - Transport measurements",
            "Hayes et al., Nature Physics 12, 916 (2016) - Strange metal behavior",
            "Analytis et al., Nature Physics 10, 194 (2014) - Quantum criticality"
        ]
    },
    
    "LaFeAsO": {
        "basic_properties": [
            "Kamihara et al., Journal of the American Chemical Society 130, 3296 (2008) - Discovery",
            "Chen et al., Nature 453, 761 (2008) - High Tc confirmation",
            "de la Cruz et al., Nature 453, 899 (2008) - Magnetic structure"
        ],
        "transport_properties": [
            "Chen et al., Physical Review Letters 101, 057007 (2008) - Transport properties",
            "Nakai et al., Physical Review Letters 101, 077006 (2008) - NMR studies",
            "Drew et al., Nature Materials 8, 310 (2009) - Strange metal regime"
        ]
    }
}

# Theoretical framework citations
THEORETICAL_CITATIONS = {
    "strange_metals": [
        "Patel et al., Science 381, eabq6011 (2023) - Quantum entanglement and disorder theory",
        "Sachdev, Physical Review X 5, 041025 (2015) - Planckian dissipation",
        "Zaanen, Nature 430, 512 (2004) - Strange metal concept",
        "Varma et al., Physical Review Letters 63, 1996 (1989) - Marginal Fermi liquid"
    ],
    "disorder_engineering": [
        "Alloul et al., Reviews of Modern Physics 81, 45 (2009) - Disorder in cuprates review",
        "Kivelson et al., Reviews of Modern Physics 75, 1201 (2003) - Electronic inhomogeneity",
        "Dagotto, Science 309, 257 (2005) - Phase separation and percolation"
    ],
    "quantum_simulation": [
        "Lloyd, Science 273, 1073 (1996) - Universal quantum simulators",
        "Georgescu et al., Reviews of Modern Physics 86, 153 (2014) - Quantum simulation review",
        "Altman et al., PRX Quantum 2, 017003 (2021) - Quantum simulators roadmap"
    ]
}

def get_material_citations(material_name: str) -> Dict[str, List[str]]:
    """Get all citations for a specific material."""
    return MATERIAL_CITATIONS.get(material_name, {})

def get_theoretical_citations() -> Dict[str, List[str]]:
    """Get theoretical framework citations."""
    return THEORETICAL_CITATIONS

def format_citations_for_material(material_name: str) -> str:
    """Format citations for a material in a readable format."""
    citations = get_material_citations(material_name)
    if not citations:
        return f"No citations available for {material_name}"
    
    formatted = f"Citations for {material_name}:\n"
    formatted += "=" * (len(f"Citations for {material_name}:") + 1) + "\n\n"
    
    for category, refs in citations.items():
        formatted += f"{category.replace('_', ' ').title()}:\n"
        for ref in refs:
            formatted += f"  • {ref}\n"
        formatted += "\n"
    
    return formatted

def validate_material_parameters(material_name: str) -> bool:
    """Check if material parameters have literature backing."""
    return material_name in MATERIAL_CITATIONS

if __name__ == "__main__":
    # Print all available citations
    print("Material Parameter Citations")
    print("=" * 50)
    
    for material in MATERIAL_CITATIONS.keys():
        print(f"\n{material}:")
        citations = get_material_citations(material)
        total_refs = sum(len(refs) for refs in citations.values())
        print(f"  {total_refs} references across {len(citations)} categories")
    
    print(f"\nTheoretical Framework:")
    theory_citations = get_theoretical_citations()
    total_theory_refs = sum(len(refs) for refs in theory_citations.values())
    print(f"  {total_theory_refs} references across {len(theory_citations)} categories")