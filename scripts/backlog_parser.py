#!/usr/bin/env python3
"""
Backlog Parser - Parse and update YAML story definitions in backlog.md

This module provides functionality to:
1. Parse story definitions from the backlog markdown file
2. Update story status and completion information
3. Track dependencies and enable new available tasks
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Story:
    """Represents a single story from the backlog."""
    id: str
    type: str
    priority: str
    effort: str
    sprint: Optional[int]
    status: str
    dependencies: List[str]
    files_to_create: List[str]
    acceptance_criteria: List[str]
    technical_notes: Optional[str] = None
    completion_notes: Optional[str] = None
    raw_yaml: str = ""
    start_line: int = 0
    end_line: int = 0


class BacklogParser:
    """Parse and manage the IPE development backlog."""
    
    def __init__(self, backlog_path: Path):
        self.backlog_path = Path(backlog_path)
        self.stories: Dict[str, Story] = {}
        self._file_content: List[str] = []
        
    def load_backlog(self) -> Dict[str, Story]:
        """Load and parse all stories from backlog.md"""
        with open(self.backlog_path, 'r') as f:
            self._file_content = f.readlines()
        
        content = ''.join(self._file_content)
        
        # Find all YAML code blocks
        yaml_pattern = r'```yaml\n(.*?)\n```'
        matches = re.finditer(yaml_pattern, content, re.DOTALL | re.MULTILINE)
        
        self.stories = {}
        for match in matches:
            yaml_content = match.group(1)
            try:
                story_data = yaml.safe_load(yaml_content)
                if isinstance(story_data, dict) and 'id' in story_data:
                    story = self._create_story_from_data(story_data, yaml_content, match)
                    self.stories[story.id] = story
            except yaml.YAMLError as e:
                print(f"Warning: Failed to parse YAML block: {e}")
                continue
                
        return self.stories
    
    def _create_story_from_data(self, data: Dict[str, Any], raw_yaml: str, match) -> Story:
        """Create a Story object from parsed YAML data."""
        # Find line numbers for this YAML block
        content_up_to_match = ''.join(self._file_content)[:match.start()]
        start_line = content_up_to_match.count('\n')
        end_line = start_line + raw_yaml.count('\n') + 2  # +2 for the ``` lines
        
        # Clean status field to remove quotes and comments
        status = data.get('status', 'todo')
        if isinstance(status, str):
            # Remove quotes and comments more thoroughly
            status = status.strip()
            if status.startswith('"') or status.startswith("'"):
                status = status[1:]
            if status.endswith('"') or status.endswith("'"):
                status = status[:-1]
            if '#' in status:
                status = status.split('#')[0].strip()
            status = status.strip()
        
        return Story(
            id=data.get('id', ''),
            type=data.get('type', ''),
            priority=data.get('priority', ''),
            effort=data.get('effort', ''),
            sprint=data.get('sprint'),
            status=status,
            dependencies=data.get('dependencies', []),
            files_to_create=data.get('files_to_create', []),
            acceptance_criteria=data.get('acceptance_criteria', []),
            technical_notes=data.get('technical_notes'),
            completion_notes=data.get('completion_notes'),
            raw_yaml=raw_yaml,
            start_line=start_line,
            end_line=end_line
        )
    
    def get_story(self, story_id: str) -> Optional[Story]:
        """Get a specific story by ID."""
        return self.stories.get(story_id)
    
    def get_available_stories(self) -> List[Story]:
        """Get stories that are available to work on (todo status with satisfied dependencies)."""
        completed_stories = {s.id for s in self.stories.values() if s.status == 'done'}
        available = []
        
        for story in self.stories.values():
            if story.status == 'todo':
                # Check if all dependencies are satisfied
                if all(dep in completed_stories for dep in story.dependencies):
                    available.append(story)
                    
        return available
    
    def complete_story(self, story_id: str, completion_notes: str = None) -> bool:
        """Mark a story as completed and update the backlog file."""
        if story_id not in self.stories:
            print(f"Error: Story {story_id} not found in backlog")
            return False
            
        story = self.stories[story_id]
        if story.status == 'done':
            print(f"Warning: Story {story_id} is already marked as completed")
            return True
            
        # Update story status
        story.status = 'done'
        if completion_notes:
            story.completion_notes = completion_notes
            
        # Generate updated YAML
        updated_yaml = self._generate_updated_yaml(story)
        
        # Update the file content
        return self._update_file_content(story, updated_yaml)
    
    def _generate_updated_yaml(self, story: Story) -> str:
        """Generate updated YAML for a completed story."""
        story_dict = {
            'id': story.id,
            'type': story.type,
            'priority': story.priority,
            'effort': story.effort,
            'sprint': story.sprint,
            'status': 'done',  # Don't add quotes/comments here
            'dependencies': story.dependencies,
            'files_to_create': story.files_to_create,
            'acceptance_criteria': story.acceptance_criteria
        }
        
        if story.technical_notes:
            story_dict['technical_notes'] = story.technical_notes
            
        if story.completion_notes:
            story_dict['completion_notes'] = story.completion_notes
        
        yaml_str = yaml.dump(story_dict, default_flow_style=False, sort_keys=False)
        
        # Now manually add the comment to the status line
        lines = yaml_str.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('status:'):
                lines[i] = 'status: "done"  # Completed'
                break
                
        return '\n'.join(lines)
    
    def _update_file_content(self, story: Story, updated_yaml: str) -> bool:
        """Update the file content with the new YAML."""
        try:
            # Find the YAML block in the file content and replace it
            content = ''.join(self._file_content)
            
            # Find the specific YAML block for this story - look for the complete yaml block
            yaml_pattern = r'```yaml\n(.*?id:\s*["\']?' + re.escape(story.id) + r'["\']?.*?)\n```'
            match = re.search(yaml_pattern, content, re.DOTALL | re.MULTILINE)
            
            if match:
                # Replace the YAML content, but preserve the ``` markers
                old_block = match.group(0)  # Full ```yaml...``` block
                new_block = f"```yaml\n{updated_yaml.strip()}\n```"
                new_content = content.replace(old_block, new_block)
                
                # Write back to file
                with open(self.backlog_path, 'w') as f:
                    f.write(new_content)
                
                # Reload the file content
                with open(self.backlog_path, 'r') as f:
                    self._file_content = f.readlines()
                    
                return True
            else:
                print(f"Error: Could not find YAML block for story {story.id}")
                return False
                
        except Exception as e:
            print(f"Error updating file: {e}")
            return False
    
    def save_backlog(self) -> bool:
        """Save the current state back to the backlog file."""
        try:
            with open(self.backlog_path, 'w') as f:
                f.writelines(self._file_content)
            return True
        except Exception as e:
            print(f"Error saving backlog: {e}")
            return False


def main():
    """Test the backlog parser."""
    backlog_path = Path(__file__).parent.parent / "documentation" / "backlog.md"
    parser = BacklogParser(backlog_path)
    
    print("Loading backlog...")
    stories = parser.load_backlog()
    print(f"Found {len(stories)} stories")
    
    # Show available stories
    available = parser.get_available_stories()
    print(f"\nAvailable stories to work on: {len(available)}")
    for story in available[:5]:  # Show first 5
        print(f"  - {story.id}: {story.acceptance_criteria[0] if story.acceptance_criteria else 'No criteria'}")


if __name__ == "__main__":
    main()