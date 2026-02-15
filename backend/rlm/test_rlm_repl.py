"""
test_rlm_repl.py - Tests for RLM REPL core functionality

Tests the REPL environment, file system tools, and agent orchestration.
"""

import pytest
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_agents import (
    AgentAdapter,
    TestScenario,
    ScenarioRunner,
    TestResultAggregator
)


# ============================================================================
# RLM REPL Adapter
# ============================================================================

class RLMReplAdapter(AgentAdapter):
    """
    Adapter for testing RLM REPL functionality.
    """
    
    def __init__(self, test_files_dir: str = None):
        from rlm.rlm_repl import RLM_REPL
        
        self.test_files_dir = test_files_dir or "/tmp/rlm_test"
        Path(self.test_files_dir).mkdir(parents=True, exist_ok=True)
        
        self.rlm = RLM_REPL(
            allowed_roots=[self.test_files_dir, "/tmp/rlm_scratch"],
            max_iterations=10,
        )
    
    def call(self, input: str) -> str:
        """Execute RLM query and return result."""
        
        # Prepare context for RLM
        context = f"Test directory: {self.test_files_dir}"
        
        # Run RLM completion
        result = self.rlm.completion(context=context, query=input)
        
        return result or "No result from RLM"
    
    def reset(self):
        """Reset RLM state."""
        self.rlm.reset()


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_files_dir(tmp_path):
    """Create temporary directory with test files."""
    test_dir = tmp_path / "test_files"
    test_dir.mkdir()
    
    # Create sample CSV file
    csv_file = test_dir / "sales.csv"
    csv_file.write_text("""Month,Revenue,Expenses
January,10000,7000
February,12000,7500
March,15000,8000""")
    
    # Create sample text file
    txt_file = test_dir / "notes.txt"
    txt_file.write_text("These are test notes.\nSecond line.\nThird line.")
    
    # Create sample JSON file
    json_file = test_dir / "data.json"
    json_file.write_text(json.dumps({
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
    }))
    
    return str(test_dir)


@pytest.fixture
def rlm_adapter(test_files_dir):
    """Create RLM adapter with test files."""
    return RLMReplAdapter(test_files_dir)


@pytest.fixture
def llm_client():
    """Get LLM client for test agents."""
    from rlm.utils.llm import get_llm_client
    return get_llm_client(None, os.getenv("RLM_ROOT_MODEL", ""))


@pytest.fixture
def runner(rlm_adapter, llm_client):
    """Create scenario runner for RLM tests."""
    return ScenarioRunner(
        agent_under_test=rlm_adapter,
        llm_client=llm_client,
        verbose=True
    )


# ============================================================================
# File System Tests
# ============================================================================

def test_fs_list_directory(runner, test_files_dir):
    """Test that RLM can list files in a directory."""
    
    scenario = TestScenario(
        name="List Directory Contents",
        description="List all files in the test directory",
        initial_message=f"List all files in {test_files_dir}",
        criteria=[
            "Agent lists at least 3 files",
            "Agent shows file names clearly",
            "Agent mentions CSV, TXT, and JSON files",
            "No errors in the output"
        ],
        max_turns=1
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_fs_parse_csv(runner, test_files_dir):
    """Test that RLM can parse and analyze CSV files."""
    
    scenario = TestScenario(
        name="Parse CSV File",
        description="Read and analyze sales.csv",
        initial_message=f"Read the sales.csv file in {test_files_dir} and tell me the total revenue",
        criteria=[
            "Agent successfully reads the CSV file",
            "Agent calculates total revenue (should be 37000)",
            "Agent provides the correct sum",
            "No parsing errors"
        ],
        max_turns=1,
        context={"expected_total": 37000}
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_fs_parse_json(runner, test_files_dir):
    """Test that RLM can parse and query JSON files."""
    
    scenario = TestScenario(
        name="Parse JSON File",
        description="Read and query data.json",
        initial_message=f"Read data.json in {test_files_dir} and tell me how many users there are",
        criteria=[
            "Agent successfully reads the JSON file",
            "Agent correctly identifies 2 users",
            "Agent can name the users (Alice and Bob)",
            "No parsing errors"
        ],
        max_turns=1
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


# ============================================================================
# REPL Code Execution Tests
# ============================================================================

def test_repl_basic_math(runner):
    """Test that RLM can execute simple calculations."""
    
    scenario = TestScenario(
        name="Basic Math Calculation",
        description="Perform a simple calculation",
        initial_message="Calculate 123 * 456 and tell me the result",
        criteria=[
            "Agent executes the calculation",
            "Agent provides the correct answer (56088)",
            "No execution errors"
        ],
        max_turns=1,
        context={"expected_answer": 56088}
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_repl_data_analysis(runner, test_files_dir):
    """Test that RLM can perform data analysis on loaded files."""
    
    scenario = TestScenario(
        name="Data Analysis",
        description="Analyze sales data to find the month with highest profit",
        initial_message=(
            f"Load sales.csv from {test_files_dir}, calculate profit "
            "(revenue - expenses) for each month, and tell me which month had the highest profit"
        ),
        criteria=[
            "Agent loads and parses the CSV",
            "Agent calculates profit for each month",
            "Agent correctly identifies March as highest profit month",
            "Agent shows the calculations clearly"
        ],
        max_turns=1,
        context={"expected_month": "March"}
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


# ============================================================================
# Error Handling Tests
# ============================================================================

def test_error_handling_nonexistent_file(runner):
    """Test that RLM handles errors gracefully when file doesn't exist."""
    
    scenario = TestScenario(
        name="Error Handling - Nonexistent File",
        description="Try to read a file that doesn't exist",
        initial_message="Read the file /nonexistent/path/fake_file.txt",
        criteria=[
            "Agent recognizes the file doesn't exist",
            "Agent provides a clear error message",
            "Agent doesn't crash or hang",
            "Agent offers to help with something else"
        ],
        max_turns=1
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_error_handling_invalid_calculation(runner):
    """Test that RLM handles syntax errors in calculations."""
    
    scenario = TestScenario(
        name="Error Handling - Invalid Syntax",
        description="Try to execute code with syntax errors",
        initial_message="Calculate the result of this expression: 5 + + 3",
        criteria=[
            "Agent recognizes the syntax error",
            "Agent explains what's wrong",
            "Agent doesn't crash",
            "Agent may offer to fix it or ask for clarification"
        ],
        max_turns=1
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


# ============================================================================
# Multi-Turn Interaction Tests
# ============================================================================

def test_multi_turn_file_analysis(runner, test_files_dir):
    """Test RLM's ability to handle multi-turn file analysis."""
    
    scenario = TestScenario(
        name="Multi-Turn File Analysis",
        description="Sequential analysis of sales data",
        initial_message=f"Load the sales.csv file from {test_files_dir}",
        criteria=[
            "Agent successfully completes the multi-turn interaction",
            "Agent remembers context from previous turns",
            "Agent provides consistent data across turns",
            "All calculations are accurate"
        ],
        max_turns=3,
        user_persona="expert"
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


# ============================================================================
# Full Test Suite
# ============================================================================

def test_rlm_full_suite(runner, test_files_dir):
    """Run comprehensive RLM test suite."""
    
    aggregator = TestResultAggregator()
    
    scenarios = [
        TestScenario(
            name="FS-1: List Files",
            description="List directory contents",
            initial_message=f"List files in {test_files_dir}",
            criteria=["Lists at least 3 files", "No errors"],
            max_turns=1
        ),
        TestScenario(
            name="FS-2: Read CSV",
            description="Read and parse CSV",
            initial_message=f"Read sales.csv from {test_files_dir}",
            criteria=["Successfully reads CSV", "Shows data", "No errors"],
            max_turns=1
        ),
        TestScenario(
            name="REPL-1: Math",
            description="Basic calculation",
            initial_message="Calculate 100 * 25",
            criteria=["Correct answer (2500)", "No errors"],
            max_turns=1
        ),
        TestScenario(
            name="REPL-2: Data Analysis",
            description="Analyze CSV data",
            initial_message=f"Calculate total revenue from sales.csv in {test_files_dir}",
            criteria=["Correct sum (37000)", "Shows calculation", "No errors"],
            max_turns=1
        ),
        TestScenario(
            name="Error-1: Missing File",
            description="Handle missing file gracefully",
            initial_message="Read /fake/path/missing.txt",
            criteria=["Recognizes error", "Clear message", "No crash"],
            max_turns=1
        ),
    ]
    
    for scenario in scenarios:
        result = runner.run(scenario)
        aggregator.add_result(result)
    
    aggregator.print_summary()
    aggregator.save_report("test_results/rlm_repl_report.json")
    
    summary = aggregator.summary()
    assert summary["pass_rate"] >= 0.6, f"Pass rate too low: {summary['pass_rate']:.1%}"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])