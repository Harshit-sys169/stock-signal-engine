# Development

Guidelines for contributing to the project.

## Setup

```bash
git clone https://github.com/Harshit-sys169/stock-signal-engine.git
cd stock-signal-engine

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install pytest black pylint mypy  # optional dev tools
```

## Code Style

- Follow PEP 8
- Use type hints for function arguments and returns
- Max line length: 100 characters
- Use meaningful variable names

Example:

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

## Testing

Run tests with:

```bash
pytest tests/ -v
```

Create test files in `tests/` matching module structure:

```
tests/
├── test_data.py
├── test_features.py
├── test_models.py
└── test_backtest.py
```

## Commits

- Keep commits focused on one thing
- Use clear commit messages
- Example: "Add RSI indicator to technical features"

## Branches

Name branches by type:

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Run lint checks
5. Push and open a PR
6. Address review comments
7. Once approved, maintainers will merge

## Common Tasks

### Add a New Feature Indicator

1. Implement in `features/technical.py`:

```python
def calculate_new_indicator(data: pd.DataFrame, period: int) -> pd.Series:
    """Calculate your new indicator."""
    return data['close'].rolling(period).mean()
```

2. Update `features/pipeline.py`:

```python
features['new_indicator'] = calculate_new_indicator(ohlcv_data, period=14)
```

3. Retrain:

```bash
make train
```

### Debug Model Performance

Check feature importance:

```python
from models.train import train_model
import matplotlib.pyplot as plt

model = train_model()
importance = model.feature_importances_
plt.barh(range(len(importance)), importance)
plt.show()
```

Check feature statistics:

```python
from features.pipeline import build_feature_panel
features = build_feature_panel(datasets)
print(features.describe())
```

### Change Configuration

Edit `config/settings.py`. The entire pipeline uses these values.

```python
FEATURE_PARAMS = {
    'rsi_period': 14,
    'macd_fast': 12,
}

MODEL_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 7,
}
```

Then retrain:

```bash
make train
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Check all `__init__.py` files exist |
| Data validation fails | Check `data/validate.py` expectations |
| Model won't train | Reduce feature count or data window |
| Streamlit cache issue | Delete `.streamlit/` and restart |

## Questions?

Check [Architecture](ARCHITECTURE.md) for system design, [Quickstart](QUICKSTART.md) for setup help, or open an issue.
