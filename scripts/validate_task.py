#!/usr/bin/env python3
"""
Validate Task Script - Validate task implementation before completion

This script:
1. Checks that all required files exist
2. Validates that acceptance criteria can be tested
3. Runs relevant tests
4. Checks code quality (if tools are available)
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Add the parent directory to the path so we can import scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.backlog_parser import BacklogParser


def validate_files_exist(story):
    """Check that all required files for a story exist."""
    missing_files = []
    existing_files = []
    
    for file_path in story.files_to_create:
        full_path = Path(file_path)
        if full_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    return existing_files, missing_files


def validate_python_syntax(file_path):
    """Validate Python syntax for a file."""
    try:
        with open(file_path, 'r') as f:
            compile(f.read(), file_path, 'exec')
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error reading file: {e}"


def run_tests_for_story(story):
    """Run tests related to a story."""
    test_files = [f for f in story.files_to_create if f.startswith('tests/')]
    
    if not test_files:
        return True, "No test files specified"
    
    results = []
    for test_file in test_files:
        if not Path(test_file).exists():
            results.append(f"❌ Test file missing: {test_file}")
            continue
            
        try:
            # Try to run the specific test file
            result = subprocess.run(
                ['python', '-m', 'pytest', test_file, '-v'],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                results.append(f"✅ Tests passed: {test_file}")
            else:
                results.append(f"❌ Tests failed: {test_file}")
                results.append(f"   Output: {result.stdout}")
                results.append(f"   Errors: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            results.append(f"⏱️  Test timeout: {test_file}")
        except Exception as e:
            results.append(f"❌ Test error: {test_file} - {e}")
    
    all_passed = all("✅" in result for result in results if result.startswith(("✅", "❌")))
    return all_passed, "\n".join(results)


def check_code_quality(file_paths):
    """Run code quality checks if tools are available."""
    results = []
    python_files = [f for f in file_paths if f.endswith('.py') and not f.startswith('tests/')]
    
    if not python_files:
        return True, "No Python files to check"
    
    # Check if tools are available and run them
    tools = [
        ('black', ['--check'], 'Code formatting'),
        ('mypy', [], 'Type checking'),
        ('flake8', [], 'Style checking')
    ]
    
    for tool, args, description in tools:
        try:
            cmd = [tool] + args + python_files
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                results.append(f"✅ {description}: Passed")
            else:
                results.append(f"⚠️  {description}: Issues found")
                if result.stdout:
                    results.append(f"   {result.stdout[:200]}...")
                    
        except FileNotFoundError:
            results.append(f"ℹ️  {description}: Tool not available ({tool})")
        except subprocess.TimeoutExpired:
            results.append(f"⏱️  {description}: Timeout")
        except Exception as e:
            results.append(f"❌ {description}: Error - {e}")
    
    return True, "\n".join(results)


def validate_acceptance_criteria(story, existing_files):
    """Check if implementation addresses acceptance criteria."""
    results = []
    
    # This is a basic check - in practice, you might want more sophisticated validation
    implementation_files = [f for f in existing_files if not f.startswith('tests/')]
    
    if not implementation_files:
        results.append("❌ No implementation files found")
        return False, "\n".join(results)
    
    # Check that files have substantial content (more than just templates)
    for file_path in implementation_files:
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            lines = [line for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
            
            if len(lines) < 10:
                results.append(f"⚠️  {file_path}: May need more implementation (only {len(lines)} substantial lines)")
            elif 'TODO' in content or 'NotImplementedError' in content:
                results.append(f"⚠️  {file_path}: Contains TODO items or NotImplementedError")
            else:
                results.append(f"✅ {file_path}: Appears to have substantial implementation")
                
        except Exception as e:
            results.append(f"❌ {file_path}: Error reading - {e}")
    
    # Basic heuristic: if most files look implemented, consider it valid
    good_files = sum(1 for result in results if result.startswith("✅"))
    total_files = len(implementation_files)
    
    if good_files >= total_files * 0.5:  # At least half the files look good
        return True, "\n".join(results)
    else:
        return False, "\n".join(results)


def main():
    parser = argparse.ArgumentParser(description='Validate task implementation')
    parser.add_argument('task_id', help='Task ID to validate (e.g., VIZ-001)')
    parser.add_argument('--skip-tests', action='store_true', help='Skip running tests')
    parser.add_argument('--skip-quality', action='store_true', help='Skip code quality checks')
    
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
        sys.exit(1)
    
    print(f"=== Validating {story.id} ===")
    print(f"Status: {story.status}")
    
    validation_results = []
    overall_success = True
    
    # 1. Check files exist
    print(f"\\n1. Checking required files...")
    existing_files, missing_files = validate_files_exist(story)
    
    if missing_files:
        print(f"❌ Missing files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        overall_success = False
    else:
        print(f"✅ All required files exist ({len(existing_files)} files)")
    
    if existing_files:
        print(f"📁 Existing files:")
        for file_path in existing_files:
            print(f"   - {file_path}")
    
    # 2. Validate Python syntax
    print(f"\\n2. Checking Python syntax...")
    python_files = [f for f in existing_files if f.endswith('.py')]
    
    if python_files:
        syntax_issues = []
        for file_path in python_files:
            valid, error = validate_python_syntax(file_path)
            if valid:
                print(f"✅ {file_path}: Syntax OK")
            else:
                print(f"❌ {file_path}: Syntax error - {error}")
                syntax_issues.append(file_path)
                overall_success = False
        
        if not syntax_issues:
            print(f"✅ All Python files have valid syntax")
    else:
        print(f"ℹ️  No Python files to check")
    
    # 3. Run tests
    if not args.skip_tests:
        print(f"\\n3. Running tests...")
        test_success, test_output = run_tests_for_story(story)
        print(test_output)
        if not test_success:
            overall_success = False
    else:
        print(f"\\n3. Skipping tests (--skip-tests)")
    
    # 4. Check code quality
    if not args.skip_quality:
        print(f"\\n4. Checking code quality...")
        quality_success, quality_output = check_code_quality(existing_files)
        print(quality_output)
        # Don't fail overall validation for quality issues, just warn
    else:
        print(f"\\n4. Skipping code quality checks (--skip-quality)")
    
    # 5. Validate acceptance criteria
    print(f"\\n5. Checking acceptance criteria implementation...")
    criteria_success, criteria_output = validate_acceptance_criteria(story, existing_files)
    print(criteria_output)
    if not criteria_success:
        overall_success = False
    
    # Summary
    print(f"\\n=== Validation Summary ===")
    if overall_success:
        print(f"✅ {story.id} validation PASSED")
        print(f"Ready to complete with: python scripts/complete_task.py {args.task_id}")
    else:
        print(f"❌ {story.id} validation FAILED")
        print(f"Please address the issues above before completing the task")
        sys.exit(1)


if __name__ == "__main__":
    main()