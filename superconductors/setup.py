"""
Setup configuration for the Tri-Hybrid Superconductor Discovery Platform.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("readme.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="strange-metal-superconductor",
    version="0.1.0",
    author="Physics Simulation Team",
    author_email="team@physics-simulation.com",
    description="Tri-Hybrid Superconductor Discovery Platform leveraging strange metal physics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/physics-simulation/strange-metal-superconductor",
    project_urls={
        "Bug Tracker": "https://github.com/physics-simulation/strange-metal-superconductor/issues",
        "Documentation": "https://strange-metal-superconductor.readthedocs.io/",
        "Source Code": "https://github.com/physics-simulation/strange-metal-superconductor",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.3.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.19.0",
        ],
        "gpu": [
            "cupy>=10.0.0",
            "jax[cuda]>=0.4.0",
        ],
        "quantum-chemistry": [
            "pyscf>=2.0.0",
            "openfermion>=1.5.0",
        ],
        "visualization": [
            "plotly>=5.8.0",
            "ipywidgets>=7.7.0",
            "jupyterlab>=3.4.0",
        ],
        "cloud": [
            "google-cloud-storage>=2.8.0",
            "google-cloud-firestore>=2.11.0",
            "boto3>=1.24.0",  # AWS support
        ],
    },
    entry_points={
        "console_scripts": [
            "strange-metal-discover=experiments.discover_material:main",
            "strange-metal-verify=experiments.verify_strange_metal:main",
            "strange-metal-optimize=experiments.optimize_disorder:main",
        ],
    },
    package_data={
        "": ["*.md", "*.txt", "*.yaml", "*.json"],
        "data": ["materials/*", "experiments/*", "discoveries/*"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "superconductivity",
        "strange metals",
        "quantum simulation",
        "materials discovery",
        "quantum computing",
        "condensed matter physics",
        "high-tc superconductors",
        "disorder engineering",
        "pasqal",
        "rydberg atoms",
    ],
)