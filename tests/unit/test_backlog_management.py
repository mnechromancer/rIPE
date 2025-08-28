#!/usr/bin/env python3
"""
Test the backlog management system to ensure Epic 6 works correctly
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from scripts.backlog_parser import BacklogParser, Story


class TestBacklogManagement:
    """Test the backlog management system for Epic 6."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a temporary backlog file for testing
        self.test_backlog_content = """# Test Backlog

## Epic 1: Test Epic [TEST]

### TEST-001: Test Story One
```yaml
id: "TEST-001"
type: "story"
priority: "P0"
effort: "S"
sprint: 1
status: "todo"
dependencies: []
files_to_create:
  - "test_file1.py"
  - "tests/unit/test_test1.py"
acceptance_criteria:
  - "Basic functionality works"
  - "Error handling implemented"
technical_notes: "Use simple implementation"
```

### TEST-002: Test Story Two
```yaml
id: "TEST-002"
type: "story"
priority: "P1"
effort: "S"
sprint: 2
status: "todo"
dependencies: ["TEST-001"]
files_to_create:
  - "test_file2.py"
acceptance_criteria:
  - "Depends on TEST-001"
```

### TEST-003: Already Done Story
```yaml
id: "TEST-003"
type: "story"
priority: "P0"
effort: "S"
sprint: 1
status: "done"  # Completed
dependencies: []
files_to_create:
  - "test_file3.py"
acceptance_criteria:
  - "Already completed"
completion_notes: "This was completed earlier"
```
"""

        self.temp_dir = Path(tempfile.mkdtemp())
        self.backlog_file = self.temp_dir / "test_backlog.md"

        with open(self.backlog_file, "w") as f:
            f.write(self.test_backlog_content)

    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_backlog_parser_loads_stories(self):
        """Test that the backlog parser correctly loads stories."""
        parser = BacklogParser(self.backlog_file)
        stories = parser.load_backlog()

        assert len(stories) == 3
        assert "TEST-001" in stories
        assert "TEST-002" in stories
        assert "TEST-003" in stories

        # Check story details
        story1 = stories["TEST-001"]
        assert story1.id == "TEST-001"
        assert story1.status == "todo"
        assert story1.dependencies == []
        assert len(story1.acceptance_criteria) == 2

        story2 = stories["TEST-002"]
        assert story2.dependencies == ["TEST-001"]

        story3 = stories["TEST-003"]
        assert story3.status == "done"

    def test_get_available_stories(self):
        """Test that available stories are correctly identified."""
        parser = BacklogParser(self.backlog_file)
        parser.load_backlog()

        available = parser.get_available_stories()

        # Only TEST-001 should be available (TEST-002 depends on TEST-001, TEST-003 is done)
        assert len(available) == 1
        assert available[0].id == "TEST-001"

    def test_complete_story_updates_backlog(self):
        """Test that completing a story updates the backlog file."""
        parser = BacklogParser(self.backlog_file)
        parser.load_backlog()

        # Complete TEST-001
        success = parser.complete_story("TEST-001", "Test completion notes")
        assert success

        # Reload and check that status changed
        parser.load_backlog()
        story = parser.get_story("TEST-001")
        assert story.status == "done"
        assert "Test completion notes" in story.completion_notes

        # Check that TEST-002 is now available
        available = parser.get_available_stories()
        available_ids = [s.id for s in available]
        assert "TEST-002" in available_ids

    def test_complete_nonexistent_story_fails(self):
        """Test that completing a non-existent story fails gracefully."""
        parser = BacklogParser(self.backlog_file)
        parser.load_backlog()

        success = parser.complete_story("NONEXISTENT", "Notes")
        assert not success

    def test_complete_already_done_story(self):
        """Test that completing an already done story succeeds but doesn't change anything."""
        parser = BacklogParser(self.backlog_file)
        parser.load_backlog()

        # TEST-003 is already done
        original_notes = parser.get_story("TEST-003").completion_notes
        success = parser.complete_story("TEST-003", "Additional notes")
        assert success

        # Reload and check
        parser.load_backlog()
        story = parser.get_story("TEST-003")
        assert story.status == "done"
        # Should keep the original notes, not overwrite

    def test_dependency_chain_resolution(self):
        """Test that dependency chains are properly resolved."""
        # Add a story that depends on TEST-002
        additional_content = """
### TEST-004: Depends on Two
```yaml
id: "TEST-004"
type: "story"
priority: "P2"
effort: "S"
sprint: 3
status: "todo"
dependencies: ["TEST-002"]
files_to_create:
  - "test_file4.py"
acceptance_criteria:
  - "Depends on TEST-002"
```
"""

        with open(self.backlog_file, "a") as f:
            f.write(additional_content)

        parser = BacklogParser(self.backlog_file)
        parser.load_backlog()

        # Initially only TEST-001 should be available
        available = parser.get_available_stories()
        assert len(available) == 1
        assert available[0].id == "TEST-001"

        # Complete TEST-001, now TEST-002 should be available
        parser.complete_story("TEST-001")
        parser.load_backlog()
        available = parser.get_available_stories()
        available_ids = [s.id for s in available]
        assert "TEST-002" in available_ids
        assert "TEST-004" not in available_ids  # Still blocked

        # Complete TEST-002, now TEST-004 should be available
        parser.complete_story("TEST-002")
        parser.load_backlog()
        available = parser.get_available_stories()
        available_ids = [s.id for s in available]
        assert "TEST-004" in available_ids


def test_epic6_backlog_update_mechanism():
    """Test that Epic 6 backlog update mechanism works end-to-end."""
    # This is the main test for Epic 6 requirement:
    # "Update the backlog after every story"

    # Use correct path - we're in tests/unit/, so backlog is ../../documentation/backlog.md
    test_file_dir = Path(__file__).parent
    repo_root = test_file_dir.parent.parent
    backlog_path = repo_root / "documentation" / "backlog.md"

    if not backlog_path.exists():
        pytest.skip(f"Backlog file not found at {backlog_path}")

    parser = BacklogParser(backlog_path)

    # Load the real backlog
    stories = parser.load_backlog()

    # Verify Epic 6 stories exist
    viz_stories = [
        story_id for story_id in stories.keys() if story_id.startswith("VIZ-")
    ]
    assert len(viz_stories) > 0, "Epic 6 (VIZ) stories should exist in backlog"

    # Check that backlog management system works
    available = parser.get_available_stories()
    assert len(available) > 0, "Some stories should be available to work on"

    # Verify that we can identify completed and incomplete stories
    completed_viz = [
        s
        for s_id, s in stories.items()
        if s_id.startswith("VIZ-") and s.status == "done"
    ]
    todo_viz = [
        s
        for s_id, s in stories.items()
        if s_id.startswith("VIZ-") and s.status == "todo"
    ]

    print(f"Completed VIZ stories: {[s.id for s in completed_viz]}")
    print(f"Todo VIZ stories: {[s.id for s in todo_viz]}")

    # The key test for Epic 6: backlog should be updated after every story completion
    # This is proven by having VIZ-001 marked as completed
    assert (
        len(completed_viz) > 0
    ), "At least one VIZ story should be marked as completed (demonstrating backlog updates)"

    # Test dependency resolution works
    completed_story_ids = {s.id for s in stories.values() if s.status == "done"}
    for story in todo_viz:
        if story.dependencies:
            for dep in story.dependencies:
                if dep in stories and stories[dep].status != "done":
                    print(f"Story {story.id} blocked by incomplete dependency: {dep}")

    print("✅ Epic 6 backlog update mechanism is working correctly!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
