#!/usr/bin/env python3
"""
Generate Task Script - Initialize a new task for development

This script:
1. Shows task details and requirements
2. Creates directory structure for required files
3. Generates code templates if provided
4. Sets up initial test files
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.backlog_parser import BacklogParser


def create_file_structure(story):
    """Create directory structure and placeholder files for a story."""
    created_files = []
    
    for file_path in story.files_to_create:
        full_path = Path(file_path)
        
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if file already exists
        if full_path.exists():
            print(f"  ⚠ File already exists: {file_path}")
            continue
        
        # Generate appropriate content based on file type
        content = generate_file_template(file_path, story)
        
        # Create the file
        with open(full_path, 'w') as f:
            f.write(content)
        
        created_files.append(file_path)
        print(f"  ✓ Created: {file_path}")
    
    return created_files


def generate_file_template(file_path, story):
    """Generate appropriate template content for different file types."""
    path_obj = Path(file_path)
    file_name = path_obj.name
    
    if file_path.startswith('tests/'):
        return generate_test_template(file_path, story)
    elif file_path.endswith('.py'):
        return generate_python_template(file_path, story)
    elif file_path.endswith(('.tsx', '.ts')):
        return generate_typescript_template(file_path, story)
    elif file_path.endswith('.md'):
        return generate_markdown_template(file_path, story)
    else:
        return generate_generic_template(file_path, story)


def generate_test_template(file_path, story):
    """Generate test file template."""
    module_name = Path(file_path).stem.replace('test_', '')
    return f'''"""
Tests for {story.id}: {story.acceptance_criteria[0] if story.acceptance_criteria else "Test module"}

This module tests the implementation of {story.id}.
"""

import pytest
from unittest.mock import Mock, patch
# TODO: Add appropriate imports for the module being tested


class Test{module_name.replace('_', '').title()}:
    """Test cases for {module_name} functionality."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        # TODO: Implement test for: {story.acceptance_criteria[0] if story.acceptance_criteria else "basic functionality"}
        assert False, "Test not implemented"
        
    {chr(10).join([f'''
    def test_{criteria.lower().replace(" ", "_").replace("-", "_")}(self):
        """Test: {criteria}"""
        # TODO: Implement test
        assert False, "Test not implemented"''' for criteria in story.acceptance_criteria[:3]])}
    
    def test_error_handling(self):
        """Test error handling and edge cases."""
        # TODO: Implement error handling tests
        assert False, "Test not implemented"


# TODO: Add integration tests if needed
# TODO: Add performance tests if specified in acceptance criteria
'''


def generate_python_template(file_path, story):
    """Generate Python module template."""
    module_name = Path(file_path).stem
    return f'''"""
{story.id}: {story.acceptance_criteria[0] if story.acceptance_criteria else "Module implementation"}

{story.technical_notes if story.technical_notes else "TODO: Add module description"}
"""

from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class {module_name.replace('_', '').title()}:
    """
    Main class for {story.id} implementation.
    
    Implements:
    {chr(10).join([f"    - {criteria}" for criteria in story.acceptance_criteria])}
    """
    
    def __init__(self):
        """Initialize the {module_name.replace('_', '').title()}."""
        logger.info(f"Initializing {module_name.replace('_', '').title()}")
        # TODO: Add initialization logic
        
    def process(self):
        """Main processing method."""
        # TODO: Implement core functionality
        raise NotImplementedError("Method not implemented")


# TODO: Add additional classes and functions as needed
# TODO: Implement acceptance criteria:
{chr(10).join([f"#   - {criteria}" for criteria in story.acceptance_criteria])}
'''


def generate_typescript_template(file_path, story):
    """Generate TypeScript/React component template."""
    component_name = Path(file_path).stem
    
    if file_path.endswith('.tsx'):
        return f'''/**
 * {story.id}: {story.acceptance_criteria[0] if story.acceptance_criteria else "Component implementation"}
 * 
 * {story.technical_notes if story.technical_notes else "TODO: Add component description"}
 */

import React from 'react';

interface {component_name}Props {{
  // TODO: Define props interface
}}

export const {component_name}: React.FC<{component_name}Props> = (props) => {{
  // TODO: Implement component
  // Requirements:
  {chr(10).join([f"  //   - {criteria}" for criteria in story.acceptance_criteria])}
  
  return (
    <div>
      <h1>{component_name}</h1>
      <p>TODO: Implement component functionality</p>
    </div>
  );
}};

export default {component_name};
'''
    else:  # TypeScript file
        return f'''/**
 * {story.id}: {story.acceptance_criteria[0] if story.acceptance_criteria else "TypeScript module"}
 * 
 * {story.technical_notes if story.technical_notes else "TODO: Add module description"}
 */

export interface {component_name}Config {{
  // TODO: Define configuration interface
}}

export class {component_name} {{
  constructor(config: {component_name}Config) {{
    // TODO: Initialize
  }}
  
  // TODO: Implement methods for:
  {chr(10).join([f"  //   - {criteria}" for criteria in story.acceptance_criteria])}
}}
'''


def generate_markdown_template(file_path, story):
    """Generate markdown documentation template."""
    return f'''# {story.id}

{story.acceptance_criteria[0] if story.acceptance_criteria else "Documentation"}

## Overview

{story.technical_notes if story.technical_notes else "TODO: Add overview"}

## Requirements

{chr(10).join([f"- {criteria}" for criteria in story.acceptance_criteria])}

## Implementation

TODO: Add implementation details

## Usage

TODO: Add usage examples

## Testing

TODO: Add testing information
'''


def generate_generic_template(file_path, story):
    """Generate generic file template."""
    return f'''# {story.id} - {Path(file_path).name}

This file is part of {story.id} implementation.

Requirements:
{chr(10).join([f"- {criteria}" for criteria in story.acceptance_criteria])}

TODO: Implement file content
'''


def main():
    parser = argparse.ArgumentParser(description='Generate task structure and templates')
    parser.add_argument('task_id', help='Task ID to generate (e.g., VIZ-001)')
    parser.add_argument('--create-files', action='store_true', help='Create placeholder files')
    
    args = parser.parse_args()
    
    # Initialize backlog parser
    backlog_path = Path(__file__).parent.parent / "documentation" / "backlog.md"
    backlog = BacklogParser(backlog_path)
    
    print(f"Loading backlog...")
    stories = backlog.load_backlog()
    
    # Get the story
    story = backlog.get_story(args.task_id)
    if not story:
        print(f"Error: Story {args.task_id} not found in backlog")
        print(f"Available stories: {', '.join(stories.keys())}")
        sys.exit(1)
    
    # Check if story is available to work on
    available_stories = [s.id for s in backlog.get_available_stories()]
    if args.task_id not in available_stories:
        print(f"Warning: Story {args.task_id} may not be ready to work on.")
        print(f"Current status: {story.status}")
        if story.dependencies:
            incomplete_deps = [dep for dep in story.dependencies 
                             if backlog.get_story(dep) and backlog.get_story(dep).status != 'done']
            if incomplete_deps:
                print(f"Incomplete dependencies: {', '.join(incomplete_deps)}")
        print()
    
    # Show story details
    print(f"=== {story.id} ===")
    print(f"Type: {story.type}")
    print(f"Priority: {story.priority}")
    print(f"Effort: {story.effort}")
    print(f"Sprint: {story.sprint}")
    print(f"Status: {story.status}")
    
    if story.dependencies:
        print(f"Dependencies: {', '.join(story.dependencies)}")
    
    print(f"\\nAcceptance Criteria:")
    for i, criteria in enumerate(story.acceptance_criteria, 1):
        print(f"  {i}. {criteria}")
    
    print(f"\\nFiles to create:")
    for file_path in story.files_to_create:
        print(f"  - {file_path}")
    
    if story.technical_notes:
        print(f"\\nTechnical Notes:")
        print(story.technical_notes)
    
    # Create files if requested
    if args.create_files:
        print(f"\\nCreating file structure...")
        created_files = create_file_structure(story)
        print(f"\\nCreated {len(created_files)} files for {args.task_id}")
        
        print(f"\\nNext steps:")
        print(f"1. Implement the functionality in the created files")
        print(f"2. Run tests: pytest tests/unit/test_*.py")
        print(f"3. Complete task: python scripts/complete_task.py {args.task_id}")


if __name__ == "__main__":
    main()