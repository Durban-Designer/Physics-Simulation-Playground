# Personal Superconductor Research Platform

A simplified, personal research platform for discovering novel superconductors through quantum simulation and engineered disorder patterns.

## 🎯 Philosophy

**Personal research first, no over-engineering.** This platform is designed for your individual research workflow, not enterprise deployment. Simple, effective, and focused on the science.

## 🚀 Quick Start

### 1. Prerequisites
- Docker Desktop with GPU support (for RTX 5090)
- Python 3.10+ with virtual environment
- Google Cloud account (for Pasqal Cloud access)

### 2. Setup

```bash
# Clone and navigate
cd superconductors/

# Copy environment configuration
cp .env.example .env
# Edit .env with your Pasqal API key and GCP project

# Start the stack
docker-compose -f docker-compose.simple.yml up -d

# Install Python dependencies (in virtual environment)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verify everything works
python research/run_experiment.py
```

### 3. Access Your Lab
- **Web UI**: http://localhost:3000 - View experiments and results
- **Database Admin**: http://localhost:5050 (admin@research.local / admin123)
- **Python Scripts**: `python research/run_experiment.py`

## 🧪 Running Experiments

### Single Experiment
```python
from research.run_experiment import SuperconductorResearcher

researcher = SuperconductorResearcher()

# Run a single experiment
result = researcher.run_single_experiment(
    material_name='YBCO',
    temperature=150.0,
    disorder_strength=0.08,
    notes="Testing disorder enhancement"
)

print(f"Predicted Tc: {result['tc_predicted']:.1f}K")
researcher.close()
```

### Parameter Sweep
```python
# Systematic exploration
results = researcher.run_parameter_sweep(
    material_name='YBCO',
    temperature_range=(100, 300),
    disorder_range=(0.0, 0.2),
    n_temps=5,
    n_disorders=5
)

# Find the best result
best = max(results, key=lambda x: x['tc_predicted'])
print(f"Best Tc: {best['tc_predicted']:.1f}K at T={best['temperature']}K, disorder={best['disorder']}")
```

### Discovery Search
```python
# Look for breakthrough materials
discoveries = researcher.discover_promising_candidates(
    materials=['YBCO', 'BSCCO', 'LSCO'],
    target_tc=140.0  # Above current record
)

for discovery in discoveries:
    print(f"🌟 {discovery['material']}: Tc = {discovery['tc_predicted']:.1f}K")
```

## 🖥️ Architecture

```
Local Machine
├── PostgreSQL (experiments database)
├── Next.js Web UI (view results)
├── Python Research Scripts (run experiments)
│   ├── Local GPU (RTX 5090 for small simulations)
│   └── Pasqal Cloud (for large quantum systems)
└── Google Cloud Storage (backup database)
```

## 📊 Backend Selection

The system automatically chooses the optimal computational backend:

- **≤12 qubits**: Local CPU (QuTiP) - fast, free
- **13-25 qubits**: Local GPU (cuQuantum) - very fast, free  
- **>25 qubits**: Pasqal Cloud (EMU-MPS) - scalable, ~$0.01/shot

## 💾 Data Management

### Database Schema
All experiments stored in PostgreSQL with:
- Material parameters (temperature, disorder, qubits)
- Simulation results (Tc prediction, confidence, order parameter)
- Backend used and cost tracking
- Notes and metadata

### Backup to Cloud
```bash
# Backup database to Google Cloud Storage
./scripts/backup.sh

# Restore from backup
./scripts/restore.sh --list                    # Show available backups
./scripts/restore.sh --file backup_file.sql.gz # Restore local backup
./scripts/restore.sh --cloud backup_file.sql.gz # Restore from GCS
```

## 🔬 Scientific Focus

### Materials Included
- **YBCO** (YBa₂Cu₃O₇) - High-Tc cuprate
- **BSCCO** (Bi₂Sr₂CaCu₂O₈) - Bismuth cuprate  
- **LSCO** (La₂₋ₓSrₓCuO₄) - Doped cuprate
- **FeSe** - Iron-based superconductor

### Physics Implementation
- **Disorder Engineering**: Spatial correlation, patchwork patterns
- **Strange Metal Physics**: T-linear resistivity from quantum entanglement
- **Tc Enhancement**: Disorder optimization for maximum critical temperature

### Discovery Target
**Goal**: Find materials with Tc > 140K (above current ambient pressure record of 135K)

## 📈 Cost Control

### Typical Costs
- **Local simulations**: Free (your GPU/CPU)
- **Pasqal Cloud**: ~$0.01 per shot
- **Monthly estimate**: $20-50 for active research

### Budget Tracking
- All costs tracked in database
- Real-time spending analysis in web UI
- Automatic backend selection to minimize costs

## 🛠️ Customization

### Adding New Materials
Edit `core/materials.py` to add your material definitions:
```python
from core.materials import Material, MaterialType

my_material = Material(
    name="MyCompound",
    material_type=MaterialType.CUPRATE,
    lattice_constant=3.8,
    tc_pristine=95.0,
    # ... other parameters
)
```

### Custom Experiments
Modify `research/run_experiment.py` to implement your specific research workflows.

## 🔧 Troubleshooting

### GPU Not Working
```bash
# Check GPU access
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi

# Verify cuQuantum installation
python -c "import cupy; print('GPU available:', cupy.cuda.is_available())"
```

### Pasqal Cloud Issues
```bash
# Verify API key in .env
echo $PASQAL_API_KEY

# Test cloud access
python -c "
from pasqal_cloud import SDK
sdk = SDK()
print('Pasqal Cloud connected')
"
```

### Database Issues
```bash
# Check PostgreSQL container
docker logs research-postgres

# Connect to database manually
docker exec -it research-postgres psql -U postgres experiments
```

## 📚 Files Overview

```
superconductors/
├── docker-compose.simple.yml     # Simple PostgreSQL + Next.js
├── research/
│   └── run_experiment.py         # Main research script
├── web/                          # Next.js web interface
│   ├── app/page.tsx             # Dashboard
│   ├── app/experiments/         # Experiment browser
│   └── app/api/                 # TypeScript API
├── scripts/
│   ├── backup.sh                # Database backup script
│   └── restore.sh               # Database restore script
├── db/init.sql                  # PostgreSQL schema
└── .env                         # Your configuration
```

## 🎯 Next Steps

1. **Configure** your `.env` with Pasqal API key
2. **Run** your first experiment: `python research/run_experiment.py`
3. **Explore** results in web UI: http://localhost:3000
4. **Iterate** on your research ideas
5. **Backup** regularly: `./scripts/backup.sh`

Remember: This is YOUR research platform. Modify it to fit your specific needs and research questions. The goal is to focus on the science, not the infrastructure.

Happy superconductor hunting! 🧪✨