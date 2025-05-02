# PCCL

Python Communication Library (PCCL)

## Code Style and Linting

This project uses the following tools to enforce code style and quality:

### C++ Code Style

- **clang-tidy**: Static analyzer for C++ code
- **clang-format**: Code formatter for C++ following Google style guide

To check C++ code style:
```bash
clang-format -i path/to/your/file.cpp
clang-tidy path/to/your/file.cpp
```

### Python Code Style

- **pylint**: Comprehensive Python code analyzer that checks for bugs and style issues

To check Python code style:
```bash
# Check for issues
pylint pccl

# Or use pre-commit to run all checks
pre-commit run --all-files
```

### Pre-commit Hooks

This project uses pre-commit hooks to enforce code style before each commit.
To install the pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

After installation, the hooks will run automatically on each commit.
