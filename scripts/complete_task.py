#!/usr/bin/env python3
"""
Complete Task Script - Mark a story as completed and update backlog

This script:
1. Marks a story as completed in the backlog
2. Adds completion notes with timestamp
3. Shows newly available stories after completion
4. Validates that acceptance criteria files exist
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path so we can import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.backlog_parser import BacklogParser


def validate_story_completion(story):
    """Validate that required files for a story exist."""
    missing_files = []
    
    for file_path in story.files_to_create:
        full_path = Path(file_path)
        if not full_path.exists():
            missing_files.append(str(file_path))
    
    return missing_files


def generate_completion_notes(story, missing_files=None):
    """Generate completion notes for a story."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    notes = f"""
Implementation completed on {timestamp}.

Acceptance Criteria Status:"""
    
    for i, criteria in enumerate(story.acceptance_criteria, 1):
        notes += f"\n  ✓ {criteria}"
    
    if story.files_to_create:
        notes += "\n\nFiles Created:"
        for file_path in story.files_to_create:
            if missing_files and file_path in missing_files:
                notes += f"\n  ⚠ {file_path} (missing - should be created)"
            else:
                notes += f"\n  ✓ {file_path}"
    
    notes += "\n# COMPLETED"
    return notes


def main():
    parser = argparse.ArgumentParser(description='Mark a story as completed in the backlog')
    parser.add_argument('task_id', help='Task ID to mark as completed (e.g., VIZ-001)')
    parser.add_argument('--notes', help='Additional completion notes')
    parser.add_argument('--force', action='store_true', help='Complete even if files are missing')
    
    args = parser.parse_args()
    
    # Initialize backlog parser
    backlog_path = Path(__file__).parent.parent / "documentation" / "backlog.md"
    backlog = BacklogParser(backlog_path)
    
    print(f"Loading backlog from {backlog_path}...")
    stories = backlog.load_backlog()
    
    # Get the story to complete
    story = backlog.get_story(args.task_id)
    if not story:
        print(f"Error: Story {args.task_id} not found in backlog")
        print(f"Available stories: {', '.join(stories.keys())}")
        sys.exit(1)
    
    if story.status == 'done':
        print(f"Story {args.task_id} is already marked as completed")
        sys.exit(0)
    
    print(f"Found story: {story.id}")
    print(f"Current status: {story.status}")
    print(f"Acceptance criteria: {len(story.acceptance_criteria)} items")
    
    # Validate completion
    missing_files = validate_story_completion(story)
    if missing_files and not args.force:
        print(f"\nWarning: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print(f"\nUse --force to complete anyway, or create the missing files first.")
        sys.exit(1)
    
    # Get stories available before completion
    available_before = set(s.id for s in backlog.get_available_stories())
    
    # Generate completion notes
    completion_notes = generate_completion_notes(story, missing_files)
    if args.notes:
        completion_notes = f"{args.notes}\n{completion_notes}"
    
    # Complete the story
    print(f"\nCompleting story {args.task_id}...")
    success = backlog.complete_story(args.task_id, completion_notes)
    
    if success:
        print(f"✓ Story {args.task_id} marked as completed")
        
        # Reload to check for newly available stories
        backlog.load_backlog()
        available_after = set(s.id for s in backlog.get_available_stories())
        newly_available = available_after - available_before
        
        if newly_available:
            print(f"\n🎉 New stories available after completing {args.task_id}:")
            for story_id in newly_available:
                story = backlog.get_story(story_id)
                print(f"  - {story_id}: {story.acceptance_criteria[0] if story.acceptance_criteria else 'No criteria'}")
        else:
            print(f"\nNo new stories became available after completing {args.task_id}")
            
    else:
        print(f"✗ Failed to complete story {args.task_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()