#!/usr/bin/env python3
"""
run_tests.py - Test runner for RLM with agent-based testing

Provides easy commands to run different test suites.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py budget             # Budget advisor tests only
    python run_tests.py rlm                # RLM REPL tests only
    python run_tests.py quick              # Quick smoke tests
    python run_tests.py --verbose          # Detailed output
"""

import sys
import subprocess
from pathlib import Path


BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║            RLM AGENT-BASED TESTING FRAMEWORK                  ║
║                                                               ║
║     Testing Agents with Agents: User Simulator + Judge       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""


def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_command(cmd, description):
    """Run a shell command and show results."""
    print_header(description)
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode == 0:
        print("\n✅ Tests passed!")
    else:
        print("\n❌ Some tests failed.")
    
    return result.returncode


def run_all_tests(verbose=False):
    """Run the complete test suite."""
    cmd = ["pytest"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-v")
    
    cmd.append(".")
    
    return run_command(cmd, "Running Full Test Suite")


def run_budget_tests(verbose=False):
    """Run budget advisor tests only."""
    cmd = ["pytest", "-m", "budget"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    cmd.append("test_budget_advisor.py")
    
    return run_command(cmd, "Running Budget Advisor Tests")


def run_rlm_tests(verbose=False):
    """Run RLM REPL tests only."""
    cmd = ["pytest", "-m", "rlm"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    cmd.append("test_rlm_repl.py")
    
    return run_command(cmd, "Running RLM REPL Tests")


def run_quick_tests(verbose=False):
    """Run quick smoke tests."""
    cmd = ["pytest", "-k", "test_affordable_purchase or test_fs_list_directory"]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    return run_command(cmd, "Running Quick Smoke Tests")


def run_specific_test(test_name, verbose=False):
    """Run a specific test by name."""
    cmd = ["pytest", "-k", test_name]
    
    if verbose:
        cmd.extend(["-v", "-s"])
    
    return run_command(cmd, f"Running Test: {test_name}")


def setup_test_environment():
    """Set up the test environment (create sample files, etc.)."""
    print_header("Setting Up Test Environment")
    
    # Create test results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    print("✓ Created test_results directory")
    
    # Create sample budget file if it doesn't exist
    sample_budget = Path("sample_budget.xlsx")
    if not sample_budget.exists():
        try:
            from create_budget_template import create_budget_template
            create_budget_template(str(sample_budget))
            print("✓ Created sample_budget.xlsx")
        except Exception as e:
            print(f"⚠️  Could not create sample budget: {e}")
    else:
        print("✓ sample_budget.xlsx already exists")
    
    print("\n✅ Test environment ready!")


def show_help():
    """Show usage information."""
    print(BANNER)
    print("""
Usage:
    python run_tests.py [command] [options]

Commands:
    all              Run all tests (default)
    budget           Run budget advisor tests only
    rlm              Run RLM REPL tests only
    quick            Run quick smoke tests
    setup            Set up test environment
    specific <name>  Run a specific test by name
    
Options:
    -v, --verbose    Show detailed output
    -h, --help       Show this help message

Examples:
    python run_tests.py                      # Run all tests
    python run_tests.py budget --verbose     # Verbose budget tests
    python run_tests.py quick                # Quick smoke tests
    python run_tests.py specific test_affordable_purchase
    
Test Markers (use with pytest directly):
    pytest -m budget      # Budget advisor tests
    pytest -m rlm         # RLM REPL tests
    pytest -m agent       # Agent-based behavior tests
    pytest -m filesystem  # File system tool tests
    pytest -m slow        # Long-running tests
    """)


def main():
    args = sys.argv[1:]
    
    if not args or args[0] in ["-h", "--help", "help"]:
        show_help()
        return 0
    
    verbose = "-v" in args or "--verbose" in args
    command = args[0]
    
    print(BANNER)
    
    if command == "setup":
        setup_test_environment()
        return 0
    
    elif command == "all":
        setup_test_environment()
        return run_all_tests(verbose)
    
    elif command == "budget":
        setup_test_environment()
        return run_budget_tests(verbose)
    
    elif command == "rlm":
        setup_test_environment()
        return run_rlm_tests(verbose)
    
    elif command == "quick":
        setup_test_environment()
        return run_quick_tests(verbose)
    
    elif command == "specific":
        if len(args) < 2:
            print("Error: Please specify a test name")
            return 1
        test_name = args[1]
        setup_test_environment()
        return run_specific_test(test_name, verbose)
    
    else:
        print(f"Unknown command: {command}")
        print("Use 'python run_tests.py --help' for usage information")
        return 1


if __name__ == "__main__":
    sys.exit(main())