# Contributing to Credit Risk MLOps Platform

Thank you for your interest in contributing to our MLflow-centric MLOps platform! 🎉

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**
3. **Create a feature branch**
4. **Make your changes**
5. **Submit a pull request**

## 🏗️ Development Setup

### Prerequisites
- Python 3.8+
- Docker & Docker Compose
- Git

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/shanojpillai/credit-risk-mlops-platform.git
cd credit-risk-mlops-platform

# Create virtual environment
python -m venv banking-mlops-env
banking-mlops-env\Scripts\activate  # Windows
# source banking-mlops-env/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
pip install mlflow xgboost lightgbm

# Setup MLflow
python scripts/setup_mlflow.py

# Start MLflow server
mlflow server --host 127.0.0.1 --port 5000
```

## 📋 Contribution Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Include type hints where appropriate

### MLflow Best Practices
- Always log experiments to MLflow
- Use meaningful experiment and run names
- Log all relevant parameters, metrics, and artifacts
- Register models in the MLflow Model Registry

### Commit Messages
Use conventional commit format:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `refactor:` for code refactoring
- `test:` for adding tests

### Pull Request Process
1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update the README if necessary
5. Request review from maintainers

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test category
python -m pytest tests/mlflow/
python -m pytest tests/models/
```

## 📊 MLflow Integration

When contributing ML models or experiments:

1. **Use MLflow Tracking**: Log all experiments
2. **Model Registry**: Register production-ready models
3. **Artifacts**: Store datasets, plots, and model files
4. **Reproducibility**: Ensure experiments can be reproduced

## 🐛 Reporting Issues

When reporting issues:
- Use the issue template
- Provide clear reproduction steps
- Include environment details
- Add relevant logs or screenshots

## 🎯 Areas for Contribution

- **New ML Models**: Implement additional credit risk models
- **MLflow Plugins**: Custom MLflow integrations
- **Documentation**: Improve guides and tutorials
- **Testing**: Add more comprehensive tests
- **Infrastructure**: Kubernetes deployments, monitoring
- **CI/CD**: GitHub Actions workflows

## 📞 Getting Help

- Open an issue for questions
- Check existing documentation
- Review MLflow best practices

## 🏆 Recognition

Contributors will be recognized in:
- README contributors section
- Release notes
- Project documentation

Thank you for contributing! 🙏
