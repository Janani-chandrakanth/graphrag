# Fix Plan: Test Case Generation Errors

## Bug 1: PermissionError in `prompts/prompt_library.py`
- [x] Fix `_save_all()` to clean up `.tmp` file on failure
- [x] Wrap `_save_all()` in `_load()` in try/except for graceful degradation

## Bug 2: `str.format()` crash in `combined_test_case_generator.py`
- [x] Escape curly braces in `source_text`, `catalog_text`, and `steps` before `.format()`

## Bug 3: `str.format()` crash in `negative_scenario_generator.py`
- [x] Escape curly braces in `catalog_text`, `steps`, `actor`, `feature`, `expected_result` before `.format()`

## Testing
- [ ] Delete stale `.tmp` file and restart the app
- [ ] Verify test case generation works without errors

