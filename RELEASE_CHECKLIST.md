# Release checklist

Items marked **BLOCKING** must be resolved before making the GitHub repository
public.

- [ ] **BLOCKING:** confirm redistribution rights for all PolarFree-derived DfP
      files and record the permission or upstream license in
      `THIRD_PARTY_NOTICES.md`.
- [ ] **BLOCKING:** choose the project license. If MIT is approved, review and
      rename `LICENSE.template` to `LICENSE`.
- [ ] Replace `REPLACE_WITH_ORG` in `CITATION.cff` with the final GitHub owner.
- [ ] Publish SfP and DfP checkpoints and replace the placeholder commands in
      both task READMEs with stable URLs and SHA-256 checksums.
- [ ] Confirm that every reported paper result maps to a published config and
      checkpoint.
- [ ] Run `python scripts/check_release.py .` from the repository root.
- [ ] Run `python -m compileall -q SfP DfP scripts tests`.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Inspect `git status` and verify that no data, weights, private paths,
      temporary images, or experiment logs are staged.
- [ ] Create the release archive from tracked files only (`git archive`) and
      exclude working-directory artifacts.
