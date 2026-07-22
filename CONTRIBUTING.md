# Contributing

Bug reports and focused pull requests are welcome. Please include the task
(SfP or DfP), environment, command, minimal data layout, full traceback, and
checkpoint type in an issue.

Before opening a pull request, run:

```bash
python scripts/check_release.py .
python -m compileall -q SfP DfP scripts tests
python -m unittest discover -s tests -v
```

Exclude datasets, model weights, generated results, and machine-specific paths
from commits. Contributions containing third-party code must identify its
source and license.
