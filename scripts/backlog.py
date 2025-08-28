#!/usr/bin/env python3
"""
Backlog Management Tool - Main entry point for Epic 6 backlog management

Usage:
    python scripts/backlog.py list              # List available stories
    python scripts/backlog.py show STORY_ID     # Show story details  
    python scripts/backlog.py start STORY_ID    # Start working on a story
    python scripts/backlog.py validate STORY_ID # Validate story implementation
    python scripts/backlog.py complete STORY_ID # Mark story as completed
"""

import sys
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.backlog_parser import BacklogParser


def list_stories(parser, status_filter=None):
    """List stories in the backlog."""
    stories = parser.load_backlog()
    
    if status_filter:
        filtered_stories = {k: v for k, v in stories.items() if v.status == status_filter}
    else:
        filtered_stories = stories
    
    if not filtered_stories:
        print(f"No stories found" + (f" with status '{status_filter}'" if status_filter else ""))
        return
    
    print(f"Stories" + (f" with status '{status_filter}'" if status_filter else "") + f": {len(filtered_stories)}")
    print()
    
    for story_id, story in sorted(filtered_stories.items()):
        status_icon = "✅" if story.status == "done" else "📋" if story.status == "todo" else "🔄"
        print(f"{status_icon} {story_id} [{story.priority}] - {story.acceptance_criteria[0] if story.acceptance_criteria else 'No criteria'}")
        if story.dependencies:
            print(f"    Dependencies: {', '.join(story.dependencies)}")


def show_available_stories(parser):
    """Show stories that are available to work on."""
    available = parser.get_available_stories()
    
    if not available:
        print("No stories are currently available to work on.")
        print("Check dependencies or complete prerequisite stories first.")
        return
    
    print(f"Available stories to work on: {len(available)}")
    print()
    
    for story in available:
        epic = story.id.split('-')[0]
        print(f"🎯 {story.id} [{story.priority}] ({epic})")
        print(f"   Effort: {story.effort} | Sprint: {story.sprint}")
        print(f"   {story.acceptance_criteria[0] if story.acceptance_criteria else 'No criteria'}")
        if story.dependencies:
            print(f"   Dependencies: {', '.join(story.dependencies)}")
        print()


def main():
    parser_args = argparse.ArgumentParser(description='Backlog management tool for Epic 6')
    subparsers = parser_args.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List stories')
    list_parser.add_argument('--status', choices=['todo', 'done', 'in_progress'], 
                           help='Filter by status')
    list_parser.add_argument('--available', action='store_true',
                           help='Show only available stories (ready to work on)')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show story details')
    show_parser.add_argument('story_id', help='Story ID (e.g., VIZ-001)')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start working on a story')
    start_parser.add_argument('story_id', help='Story ID (e.g., VIZ-001)')
    start_parser.add_argument('--create-files', action='store_true',
                            help='Create placeholder files')
    
    # Validate command  
    validate_parser = subparsers.add_parser('validate', help='Validate story implementation')
    validate_parser.add_argument('story_id', help='Story ID (e.g., VIZ-001)')
    
    # Complete command
    complete_parser = subparsers.add_parser('complete', help='Mark story as completed')
    complete_parser.add_argument('story_id', help='Story ID (e.g., VIZ-001)')
    complete_parser.add_argument('--notes', help='Completion notes')
    
    args = parser_args.parse_args()
    
    if not args.command:
        parser_args.print_help()
        sys.exit(1)
    
    # Initialize backlog parser
    backlog_path = Path(__file__).parent.parent / "documentation" / "backlog.md"
    backlog_parser = BacklogParser(backlog_path)
    
    if args.command == 'list':
        if args.available:
            show_available_stories(backlog_parser)
        else:
            list_stories(backlog_parser, args.status)
            
    elif args.command == 'show':
        # Delegate to generate_task.py
        import subprocess
        result = subprocess.run([
            sys.executable, 'scripts/generate_task.py', args.story_id
        ], cwd=Path.cwd())
        sys.exit(result.returncode)
        
    elif args.command == 'start':
        # Delegate to generate_task.py with file creation
        import subprocess
        cmd = [sys.executable, 'scripts/generate_task.py', args.story_id]
        if args.create_files:
            cmd.append('--create-files')
        result = subprocess.run(cmd, cwd=Path.cwd())
        sys.exit(result.returncode)
        
    elif args.command == 'validate':
        # Delegate to validate_task.py
        import subprocess
        result = subprocess.run([
            sys.executable, 'scripts/validate_task.py', args.story_id
        ], cwd=Path.cwd())
        sys.exit(result.returncode)
        
    elif args.command == 'complete':
        # Delegate to complete_task.py
        import subprocess
        cmd = [sys.executable, 'scripts/complete_task.py', args.story_id]
        if args.notes:
            cmd.extend(['--notes', args.notes])
        result = subprocess.run(cmd, cwd=Path.cwd())
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()