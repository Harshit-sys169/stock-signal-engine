# Contributing to NSE-Alpha

Thank you for your interest in contributing! This guide explains how to report issues, suggest improvements, and submit code contributions.

## Ways to Contribute

### 1. **Report Bugs**
Found a bug? Open an issue with:
- Description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Your environment (Python version, OS, etc.)

### 2. **Suggest Features**
Have an idea? Open a discussion with:
- Clear description of the feature
- Why it's useful for the project
- Suggested implementation approach (if you have one)

### 3. **Improve Documentation**
- Clarify existing guides
- Add examples
- Fix typos or errors

### 4. **Submit Code**
- New features
- Performance improvements
- Bug fixes

## Before You Start

1. **Check existing issues** — Maybe someone is already working on it
2. **Discuss major changes** — Open an issue first to get feedback
3. **Read the [Development Guide](DEVELOPMENT.md)** — Understand code style and testing

## Submitting a Pull Request

### Step 1: Fork & Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/your-username/stock-signal-engine.git
cd stock-signal-engine
```

### Step 2: Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### Step 3: Make Changes

- Keep commits focused and logical
- Write clear commit messages
- Follow the code style (see [Development Guide](DEVELOPMENT.md))
- Add tests for new code
- Update documentation

### Step 4: Test Your Changes

```bash
# Run tests
pytest tests/ -v

# Run linting
black . && pylint . && mypy .
```

### Step 5: Push & Create PR

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear title (e.g., "Add new volatility indicator")
- Description of changes
- Reference to related issues (e.g., "Fixes #42")
- Screenshot or metrics if applicable

### Step 6: Code Review

- Maintainers will review your code
- Respond to feedback
- Make requested changes
- Once approved, we'll merge!

## Code Guidelines

See [DEVELOPMENT.md](DEVELOPMENT.md) for detailed guidelines on:
- Code style (PEP 8)
- Type hints
- Docstrings
- Testing
- Configuration management

## Questions?

- **Setup help:** See [QUICKSTART.md](QUICKSTART.md)
- **Architecture questions:** See [ARCHITECTURE.md](ARCHITECTURE.md)
- **Development workflow:** See [DEVELOPMENT.md](DEVELOPMENT.md)

---

**Thank you for making NSE-Alpha better!** 🎉
