# Development Guide

Contributing to NSE-Alpha. Guidelines for code quality, testing, and extending the system.

## Project Structure Reference

```
stock-signal-engine/
├── config/              # Configuration management
├── data/                # Data acquisition and validation
├── features/            # Feature engineering pipeline
├── models/              # Model training and inference
├── backtest/            # Backtesting and performance metrics
├── dashboard/           # Streamlit web interface
├── notebooks/           # Research and exploration
├── scripts/             # Scheduled scripts (cron jobs)
├── outputs/             # Generated signals and reports
├── tests/               # Unit and integration tests (to add)
└── .github/workflows/   # CI/CD pipelines
```

## Development Workflow

### 1. Setup Development Environment

```bash
# Clone and install
git clone https://github.com/Harshit-sys169/stock-signal-engine.git
cd stock-signal-engine

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools (optional)
pip install pytest black pylint mypy
```

### 2. Making Changes

**Branch Naming:**
- `feature/description` — New features
- `bugfix/description` — Bug fixes
- `docs/description` — Documentation updates

**Code Style:**
- Follow PEP 8
- Use type hints for function arguments and returns
- Max line length: 100 characters
- Use meaningful variable names (no single letters except in loops)

**Example Function:**

```python
def build_feature_matrix(
    ohlcv_data: pd.DataFrame,
    macro_data: pd.DataFrame,
    normalize: bool = True
) -> pd.DataFrame:
    """
    Build feature matrix from OHLCV and macro data.
    
    Args:
        ohlcv_data: DataFrame with columns [open, high, low, close, volume]
        macro_data: DataFrame with macro indicators
        normalize: Whether to normalize features to [0, 1]
    
    Returns:
        Feature matrix ready for model inference
    
    Raises:
        ValueError: If required columns are missing
    """
    if not all(col in ohlcv_data.columns for col in ['open', 'high', 'low', 'close', 'volume']):
        raise ValueError("Missing required OHLCV columns")
    
    # Implementation here
    return features
```

### 3. Testing

#### Run Existing Tests

```bash
pytest tests/ -v
```

#### Add New Tests

Create test files in `tests/` directory matching the module structure:

```
tests/
├── test_data.py
├── test_features.py
├── test_models.py
└── test_backtest.py
```

**Example Test:**

```python
import pytest
from features.technical import calculate_rsi

def test_rsi_calculation():
    """RSI should be between 0 and 100"""
    prices = [100, 101, 102, 101, 100, 99, 98, 99, 100]
    rsi = calculate_rsi(prices, period=14)
    
    assert 0 <= rsi <= 100
```

### 4. Documentation

**Docstring Format (NumPy style):**

```python
def function_name(param1, param2):
    """
    Short description on one line.
    
    Longer explanation if needed, describing the algorithm,
    edge cases, or important design decisions.
    
    Parameters
    ----------
    param1 : type
        Description of param1
    param2 : type
        Description of param2
    
    Returns
    -------
    type
        Description of return value
    
    Raises
    ------
    ValueError
        When parameter validation fails
    
    Examples
    --------
    >>> result = function_name(10, 20)
    >>> print(result)
    30
    """
```

### 5. Common Development Tasks

#### Add a New Feature Indicator

1. Implement in `features/technical.py`:

```python
def calculate_new_indicator(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate your new indicator."""
    return data['close'].rolling(period).mean()
```

2. Update `features/pipeline.py` to include it:

```python
features['new_indicator'] = calculate_new_indicator(ohlcv_data, period=14)
```

3. Retrain the model:

```bash
make train
```

#### Debug Model Performance

1. Check feature importance:

```python
from models.train import train_model
import matplotlib.pyplot as plt

model = train_model()
importance = model.feature_importances_
plt.barh(range(len(importance)), importance)
plt.show()
```

2. Check feature statistics:

```python
from features.pipeline import build_feature_panel
features = build_feature_panel(datasets)
print(features.describe())  # Check for NaN, outliers, scales
```

3. Run backtest with verbose output:

```python
from backtest.engine import backtest_strategy
results = backtest_strategy(verbose=True)
```

#### Modify Configuration

All system parameters are in `config/settings.py`. Change one place, entire pipeline updates:

```python
# In config/settings.py
FEATURE_PARAMS = {
    'rsi_period': 14,           # Change to 12
    'macd_fast': 12,            # Change to 10
    'bollinger_period': 20,     # Change to 25
}

MODEL_PARAMS = {
    'learning_rate': 0.05,      # Change to 0.1
    'max_depth': 7,             # Change to 10
    'num_leaves': 31,
}
```

Then retrain:

```bash
make train
```

---

## Troubleshooting Development Issues

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| Import errors | Module not in `__init__.py` | Check all `__init__.py` files exist |
| Data validation fails | Missing columns | Check `data/validate.py` expectations |
| Model won't train | Out of memory | Reduce feature count or data window |
| Dashboard won't load | Streamlit cache issue | Delete `.streamlit/` cache and restart |

---

## Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes, commit regularly
git add .
git commit -m "Add new feature with tests"

# Push and create pull request
git push origin feature/new-feature

# After review, merge to main
git checkout main
git merge feature/new-feature
git push origin main
```

---

## CI/CD Pipeline

GitHub Actions runs on every push:

- **Lint & Format:** Black, Pylint
- **Tests:** pytest with coverage
- **Type Checking:** mypy

See `.github/workflows/ci.yml` for details.

---

## Performance Profiling

Identify bottlenecks:

```python
import cProfile
import pstats

cProfile.run('from features.pipeline import build_feature_panel; build_feature_panel(datasets)', 'stats.prof')

stats = pstats.Stats('stats.prof')
stats.sort_stats('cumulative').print_stats(20)  # Top 20 functions
```

---

## Documentation Updates

- Update [ARCHITECTURE.md](ARCHITECTURE.md) if changing system design
- Update [QUICKSTART.md](QUICKSTART.md) if changing setup process
- Add docstrings to all public functions
- Update this guide if adding new development workflows

---

## Getting Help

- **Questions on strategy?** Check [ARCHITECTURE.md](ARCHITECTURE.md)
- **How do I setup?** See [QUICKSTART.md](QUICKSTART.md)
- **Is there a bug?** Open an issue with reproduction steps
- **Want to contribute?** Submit a PR with tests and documentation

---

## Code Review Checklist

Before submitting a PR:

- [ ] Code follows PEP 8 style guide
- [ ] All functions have docstrings
- [ ] Type hints are included
- [ ] Tests are added for new code
- [ ] All existing tests pass (`pytest`)
- [ ] No hardcoded values (use config)
- [ ] Performance impact is acceptable
- [ ] Documentation updated if needed

---

**Last updated:** January 2025
