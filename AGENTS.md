Hello agent. You are one of the most talented programmers of your generation.
Noether is a ML research framework for Engineering AI.
# Project setup

The project is managed with `uv` and `pyproject.toml`
**ALWAYS use `uv run` for all Python commands and tools.**

Examples:
- `uv run python script.py`
- `uv run pytest`
- `uv run mypy src/`
- `uv run ruff check .`

## Testing

- **Test Framework**: pytest
- **Test Location**: `tests/`
- **Running Tests**: `uv run pytest`
- **Running Specific Tests**: `uv run pytest tests/path/to/test_file.py`
- **Running with Coverage**: `uv run pytest --cov=src`

**Write testable code and add unit tests where it makes sense.**
Don't test trivial code or external libraries
Tests for erroneous logic must fail.

## Formatting and Linting

Use **ruff** for both formatting and linting:

```bash
# Check formatting and linting
uv run ruff check .

# Fix auto-fixable issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

## Type Checking

When appropriate, write **typed Python code** and check with **mypy**:

```bash
uv run mypy src/
```

## Docstrings

**ALWAYS update Python docstrings** when modifying code.

Follow Google-style or NumPy-style docstring conventions,
when appropriate include Examples in Sphinx syntax with the testcode directive.

```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of the function.

    Longer description if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Example:
    
        .. testcode::

        print("Hello World")

    Raises:
        ValueError: Description of when this is raised
    """
```

## Sphinx Documentation

**ALWAYS update Sphinx documentation** when changing functionality.

- Documentation location: `docs/`
- Keep documentation **concise and clear**
- Update relevant `.rst` files when adding or modifying features
- Build docs locally to verify: `uv run sphinx-build -b html docs/source docs/_build/html`
