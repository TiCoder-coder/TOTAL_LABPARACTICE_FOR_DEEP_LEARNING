# Contributing Guidelines

Thank you for your interest in contributing to this project!

## 1. Coding Style (PEP 8)
- All Python code should strictly follow [PEP 8](https://peps.python.org/pep-0008/) standards.
- Use explicit type hints for all function signatures.
- Max line length is 88 characters (e.g., using `black`).
- Imports should be logically grouped (standard library, third-party packages, local modules).

## 2. Commit Convention
Please follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:
- `feat:` A new feature.
- `fix:` A bug fix.
- `docs:` Documentation only changes.
- `style:` Changes that do not affect the meaning of the code (white-space, formatting, etc).
- `refactor:` A code change that neither fixes a bug nor adds a feature.
- `test:` Adding missing tests or correcting existing tests.
- `chore:` Changes to the build process or auxiliary tools and libraries.

*Example:* `feat: add Vision Transformer to pretrained models list`

## 3. Pull Request Guideline
1. Fork the repository and create your feature branch from `main` (`git checkout -b feature/amazing-feature`).
2. Ensure you have added/updated tests in the `tests/` directory.
3. Run the automated test suite locally to verify that all tests pass:
   ```bash
   pytest -q tests/
   ```
4. Run the fast pipeline check to ensure no integration breaks:
   ```bash
   python -m processing_own_phase.main --quick
   ```
5. Ensure no trailing whitespaces or formatting issues:
   ```bash
   git diff --check
   ```
6. Submit a Pull Request detailing the changes, motivation, and any open issues it addresses.
