#!/usr/bin/env python3
"""
Backlog parser for Epic 6 story management.
Provides functionality to parse, update, and manage story statuses in the backlog.
"""

import re
import yaml
from typing import Dict, List, Optional
from pathlib import Path


class Story:
    """Represents a user story from the backlog."""
    
    def __init__(self, data: Dict):
        self.id = data.get('id', '')
        self.type = data.get('type', 'story')
        self.priority = data.get('priority', 'P2')
        self.effort = data.get('effort', 'M')
        self.sprint = data.get('sprint', 1)
        self.status = data.get('status', 'todo')
        self.dependencies = data.get('dependencies', [])
        self.files_to_create = data.get('files_to_create', [])
        self.acceptance_criteria = data.get('acceptance_criteria', [])
        self.completion_notes = data.get('completion_notes', '')
        self.technical_notes = data.get('technical_notes', '')
        self.code_template = data.get('code_template', '')


class BacklogParser:
    """Parser for the backlog markdown file with story management capabilities."""
    
    def __init__(self, backlog_path: Path):
        self.backlog_path = Path(backlog_path)
        self.stories: Dict[str, Story] = {}
        self.backlog_content = ""
        
    def load_backlog(self) -> Dict[str, Story]:
        """Load and parse stories from the backlog file."""
        if not self.backlog_path.exists():
            raise FileNotFoundError(f"Backlog file not found: {self.backlog_path}")
            
        with open(self.backlog_path, 'r', encoding='utf-8') as f:
            self.backlog_content = f.read()
            
        self.stories = self._parse_stories()
        return self.stories
    
    def _parse_stories(self) -> Dict[str, Story]:
        """Parse stories from the markdown content."""
        stories = {}
        
        # Find all YAML blocks in the markdown
        yaml_pattern = r'```yaml\n(.*?)\n```'
        yaml_matches = re.findall(yaml_pattern, self.backlog_content, re.DOTALL)
        
        for yaml_content in yaml_matches:
            try:
                data = yaml.safe_load(yaml_content)
                if isinstance(data, dict) and 'id' in data:
                    story = Story(data)
                    stories[story.id] = story
            except yaml.YAMLError:
                # Skip malformed YAML blocks
                continue
                
        return stories
    
    def get_story(self, story_id: str) -> Optional[Story]:
        """Get a specific story by ID."""
        return self.stories.get(story_id)
    
    def get_available_stories(self) -> List[Story]:
        """Get stories that are available to work on (all dependencies completed)."""
        available = []
        
        for story in self.stories.values():
            if story.status != 'todo':
                continue
                
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in story.dependencies:
                dep_story = self.stories.get(dep_id)
                if dep_story and dep_story.status != 'done':
                    dependencies_met = False
                    break
                    
            if dependencies_met:
                available.append(story)
                
        return available
    
    def complete_story(self, story_id: str, completion_notes: str = "") -> bool:
        """Mark a story as completed and update the backlog file."""
        story = self.stories.get(story_id)
        if not story:
            return False
            
        # Update the story status
        story.status = 'done'
        if completion_notes:
            story.completion_notes = completion_notes
            
        # Update the backlog file
        return self._update_backlog_file(story_id, story)
    
    def _update_backlog_file(self, story_id: str, story: Story) -> bool:
        """Update the backlog file with the new story status."""
        try:
            # Find the YAML block for this story
            pattern = rf'(### {re.escape(story_id)}:.*?\n```yaml\n)(.*?)(\nstatus: "[^"]*")(.*?\n```)'
            
            def replace_status(match):
                header = match.group(1)
                yaml_before = match.group(2)
                status_line = match.group(3)
                yaml_after = match.group(4)
                
                # Replace the status
                new_status_line = f'\nstatus: "done"  # Completed'
                
                # Add completion notes if they don't exist and we have notes
                if story.completion_notes and 'completion_notes:' not in yaml_after:
                    # Insert completion_notes before the closing ```
                    yaml_after_lines = yaml_after.split('\n')
                    # Find the line before ```
                    for i, line in enumerate(yaml_after_lines):
                        if line.strip() == '```':
                            yaml_after_lines.insert(i, f'completion_notes: "{story.completion_notes}"')
                            break
                    yaml_after = '\n'.join(yaml_after_lines)
                
                return header + yaml_before + new_status_line + yaml_after
            
            new_content = re.sub(pattern, replace_status, self.backlog_content, flags=re.DOTALL)
            
            # Write back to file
            with open(self.backlog_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            self.backlog_content = new_content
            return True
            
        except Exception as e:
            print(f"Error updating backlog file: {e}")
            return False