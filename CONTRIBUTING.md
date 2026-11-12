# Contributing to QuantLab-Research

We welcome contributions from quantitative researchers, econometricians, and algorithmic developers.

## Development Workflow
1. Fork the repository and create your feature branch (`git checkout -b feat/factor-pca-residualization`).
2. Adhere to strict typing and PEP 8 guidelines.
3. Validate syntax and linting:
   ```bash
   flake8 src cli.py main.py --count --select=E9,F63,F7,F82,F824
   ```
4. Verify execution suite:
   ```bash
   python cli.py demo
   ```
5. Submit a descriptive Pull Request with research methodology.
