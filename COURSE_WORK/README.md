# COURSE WORK

## Environment setup

From the repository root:

```bash
venv/bin/python -m pip install --editable COURSE_WORK --no-deps --no-build-isolation
```

Select the `python3` notebook kernel from the repository virtual environment, restart the kernel, then run `notebook_course_work/CourseWork.ipynb` from top to bottom.

The editable installation exposes the canonical `src/course_work` package without adding path bootstrap logic to the notebook.

## Validation

```bash
venv/bin/python -m pytest COURSE_WORK/tests
```
