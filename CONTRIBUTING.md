# Contributing to Semantixel

First off, thank you for considering contributing to Semantixel! It's people like you that make Semantixel a powerful and accessible semantic image retrieval tool.

The following is a set of guidelines for contributing to Semantixel and its packages. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for Semantixel. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

- Ensure the bug was not already reported by searching the GitHub issues.
- If you're unable to find an open issue addressing the problem, open a new one.
- Include detailed steps to reproduce the issue.
- Provide details about your environment (OS, Python version, CUDA availability, GPU model).
- Include terminal logs or backend stack traces when applicable.

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement issue, please:

- Use a clear and descriptive title.
- Provide a step-by-step description of the suggested enhancement.
- Provide specific examples to demonstrate the steps.
- Describe the current behavior and explain which behavior you expected to see instead and why.
- Explain why this enhancement would be useful to most Semantixel users.

### Pull Requests

The process described here has several goals:
- Maintain Semantixel's code quality and architectural integrity.
- Fix problems that are important to users.
- Enable a sustainable system for maintainers to review contributions.

Please follow these steps to have your contribution considered by the maintainers:

1. **Fork** the repository and clone it to your local machine.
2. **Branch out** from the `main` branch to a descriptive branch name (e.g., `feature/add-new-provider` or `bugfix/fix-ocr-crash`).
3. **Commit** your changes following conventional commit messages (e.g., `feat: add Milvus support` or `fix: correct threshold calculation in search service`).
4. **Push** to your fork and submit a **Pull Request** to the `main` branch.
5. In your PR description, thoroughly explain *what* you changed and *why*. If it addresses an open issue, please link it.

## Development Setup

To contribute to Semantixel, you will need to set up a local development environment.

1. **Prerequisites:** 
   - Python 3.11
   - CUDA-capable GPU (highly recommended for model inference)

2. **Environment Setup:**
   ```bash
   conda create -n semantixel python=3.11 -y
   conda activate semantixel
   pip install -r requirements.txt
   ```

3. **Running Locally:**
   - Configure your local settings: `python settings.py`
   - Run the local backend server and UI: `python main.py --serve`

## Understanding the Architecture

Before making sweeping changes, please review the documentation inside the `docs/` folder, specifically:
- `docs/02_System_Architecture.md`
- `docs/03_Model_Design.md`

Semantixel follows a highly modular structure under the `semantixel/` directory:
- **`semantixel/providers/`**: Core machine learning wrappers (e.g., Hugging Face CLIP, DocTR OCR). If you are adding a new model or inference engine, it belongs here.
- **`semantixel/services/`**: Orchestration logic for indexing (`index_service.py`), searching (`search_service.py`), and managing BM25 tokenization.
- **`semantixel/api/`**: Flask REST endpoints. Keep routes clean and delegate heavy logic to the services.
- **`semantixel/utils/`**: Helper methods for scanning directories and processing media.

## Styleguides

### Python Guidelines
- All Python code must be compliant with **PEP 8**.
- Use clear, descriptive variable and function names.
- Provide **Docstrings** for all new classes, services, and complex functions.
- If introducing new backend features, ensure your code leverages batched GPU tensors where possible to preserve high throughput.

### UI Guidelines
- The frontend resides in `UI/Semantixel WebUI/`.
- Keep the design clean, responsive, and intuitive.
- Maintain the modular CSS structure and avoid writing inline styles unless necessary.

## Getting Help

If you have questions about the architecture or how to implement a specific feature, feel free to open a "Discussion" on the GitHub repository.

Thank you for contributing to Semantixel!
