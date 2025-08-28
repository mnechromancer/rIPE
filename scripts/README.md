# Epic 6: Backlog Management Scripts

This directory contains the implementation of Epic 6: "Update the backlog after every story". The system provides automated backlog management to track story completion and dependency resolution.

## Overview

The backlog management system automatically:
1. **Updates backlog status** when stories are completed
2. **Tracks dependencies** and shows available stories to work on
3. **Generates completion notes** with timestamps and acceptance criteria validation
4. **Creates file templates** to accelerate development
5. **Validates implementations** before completion

## Scripts

### 1. `backlog.py` - Main Tool
The primary interface for backlog management:

```bash
# List all stories
python scripts/backlog.py list

# Show available stories (ready to work on)
python scripts/backlog.py list --available

# Show story details
python scripts/backlog.py show VIZ-001

# Start working on a story (with file generation)
python scripts/backlog.py start VIZ-001 --create-files

# Validate implementation
python scripts/backlog.py validate VIZ-001

# Complete a story
python scripts/backlog.py complete VIZ-001 --notes "Implementation complete"
```

### 2. `backlog_parser.py` - Core Parser
Handles reading and updating the YAML story definitions in `documentation/backlog.md`:
- Parses YAML code blocks in markdown
- Manages story status updates
- Resolves dependencies
- Identifies available stories

### 3. `generate_task.py` - Task Initialization
Creates file structure and templates for new stories:
- Shows story requirements
- Creates placeholder files with appropriate templates
- Generates test files, Python modules, TypeScript components

### 4. `validate_task.py` - Implementation Validation
Validates story implementation before completion:
- Checks required files exist
- Validates Python syntax
- Runs tests if available
- Checks for substantial implementation

### 5. `complete_task.py` - Story Completion
Marks stories as completed and updates the backlog:
- Updates story status to "done"
- Adds completion notes with timestamps
- Shows newly available stories after completion
- Validates acceptance criteria files exist

## Workflow

The typical development workflow with Epic 6:

```bash
# 1. See what stories are available to work on
python scripts/backlog.py list --available

# 2. Start working on a story
python scripts/backlog.py start VIZ-002 --create-files

# 3. Implement the functionality in the created files
# (Edit the generated files, implement features, etc.)

# 4. Validate the implementation
python scripts/backlog.py validate VIZ-002

# 5. Complete the story (updates backlog automatically)
python scripts/backlog.py complete VIZ-002 --notes "Implemented plasticity landscape visualization"

# 6. Check what new stories became available
python scripts/backlog.py list --available
```

## Example: VIZ-001 Completion

This implementation includes a complete demonstration with VIZ-001:

1. **Generated files**: `web/src/components/StateSpaceExplorer.tsx`, `web/src/visualizers/StateSpace3D.tsx`, `web/src/hooks/useStateSpace.ts`
2. **Implemented**: Basic 3D visualization components with React Three Fiber foundation
3. **Completed**: Story marked as "done" in backlog with completion notes
4. **Updated**: Backlog automatically updated with status change

## File Templates

The system generates appropriate templates based on file types:
- **Python modules**: Classes with type hints, logging, docstrings
- **Test files**: pytest-compatible test classes with fixtures
- **TypeScript/React**: Functional components with proper interfaces
- **Documentation**: Markdown files with requirements and structure

## Dependency Management

The system automatically tracks story dependencies:
- Shows only available stories (all dependencies completed)
- Updates available stories list when dependencies are completed
- Prevents working on blocked stories
- Provides clear dependency information

## Testing

Run the test suite to validate the backlog management system:

```bash
python -m pytest tests/unit/test_backlog_management.py -v
```

The tests verify:
- Backlog parsing and story loading
- Dependency resolution logic
- Story completion workflow
- File operations and backlog updates

## Integration

This backlog management system integrates with:
- **Git workflow**: Stories create feature branches
- **Testing framework**: Validates implementations before completion
- **Documentation**: Automatically updates backlog.md
- **Development process**: Provides clear development workflow

## Epic 6 Achievement

This implementation satisfies the Epic 6 requirement "Update the backlog after every story" by:

✅ **Automatic backlog updates** when stories are completed  
✅ **Dependency tracking** shows newly available stories  
✅ **Status management** with completion notes and timestamps  
✅ **Workflow automation** from story start to completion  
✅ **Integration testing** validates the complete system  

The system ensures that the backlog is always up-to-date and accurately reflects the current state of development progress.