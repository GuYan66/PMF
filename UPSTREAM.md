# Upstream and licensing

- Original project: https://github.com/thuiar/MMSA
- Base commit: a94e65d07fa1ae0d44e552390074b29b0898edfd
- Original package version: 2.2.1
- Team package version: 2.2.1+pmf.1
- License: MIT; the original LICENSE and copyright notice are retained.

This distribution retains the original MMSA model implementations and adds PMF, its polarity/magnitude trainer, aligned padding masks, optional test2 evaluation, portable entry points, experiment outputs and regression tests.
Earlier polarity/magnitude modifications to 14 existing baseline models have been reverted. PMF is the only model registered to use the new trainer.

Original README text is preserved in docs/UPSTREAM_README.md; links in that archived document follow the upstream repository layout. The team README describes the modified PMF workflow.

The original automatic PyPI publishing workflow has been replaced by test-only GitHub Actions. This source snapshot does not contain .git, datasets, recovered labels/media, model checkpoints, or environment credentials.
