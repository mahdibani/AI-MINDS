"""
test_budget_advisor.py - Agent-based tests for Budget Advisor

Run with: pytest -s test_budget_advisor.py
Or with uv: uv run pytest -s test_budget_advisor.py
"""

import pytest
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from backend.budget_advisor import BudgetAdvisor
from test_agents import (
    AgentAdapter,
    TestScenario,
    ScenarioRunner,
    TestResultAggregator
)


# ============================================================================
# Budget Advisor Adapter (makes it compatible with testing framework)
# ============================================================================

class BudgetAdvisorAdapter(AgentAdapter):
    """
    Wraps BudgetAdvisor to make it testable with the agent testing framework.
    """
    
    def __init__(self, budget_file: str):
        self.advisor = BudgetAdvisor(
            budget_file=budget_file,
            user_id="test_user",
            safety_buffer=0.20
        )
        self.conversation_history = []
    
    def call(self, input: str) -> str:
        """
        Process user message and return response.
        
        Handles:
        - Direct affordability queries
        - General budget questions
        - Follow-up questions
        """
        import re
        
        # Try to extract purchase intent
        purchase_patterns = [
            r'(?:buy|purchase|get|afford)\s+(?:a|an|the)?\s*(\w+).*?(?:\$|dollars?)\s*(\d+(?:\.\d+)?)',
            r'(\w+).*?costs?\s*(?:\$|dollars?)\s*(\d+(?:\.\d+)?)',
            r'(?:\$|dollars?)\s*(\d+(?:\.\d+)?).*?(laptop|phone|watch|computer|tablet|smartwatch)',
        ]
        
        item, price = None, None
        for pattern in purchase_patterns:
            match = re.search(pattern, input.lower())
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    # Try to parse in both orders
                    try:
                        item = groups[0]
                        price = float(groups[1])
                    except ValueError:
                        item = groups[1]
                        price = float(groups[0])
                    break
        
        # If we detected a purchase query
        if item and price:
            result = self.advisor.can_afford(item, price, explain=True)
            
            # Format response
            response = result.get("explanation", result["recommendation"])
            
            # Store in history
            self.conversation_history.append({
                "user": input,
                "agent": response,
                "item": item,
                "price": price,
                "result": result
            })
            
            return response
        
        # Handle general queries
        elif any(word in input.lower() for word in ["budget", "summary", "overview"]):
            return self.advisor.get_budget_summary()
        
        elif any(word in input.lower() for word in ["history", "past", "previous"]):
            history = self.advisor.get_spending_history()
            if not history:
                return "I don't have any purchase history recorded yet."
            return f"I've analyzed {len(history)} purchases for you."
        
        else:
            # Default helpful response
            return (
                "I can help you make smart purchase decisions! "
                "Tell me what you're thinking about buying and how much it costs."
            )
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def budget_advisor():
    """Create a test budget advisor with sample data."""
    # Use the sample budget created by create_budget_template.py
    budget_file = Path(__file__).parent / "sample_budget.xlsx"
    
    # If doesn't exist, create it
    if not budget_file.exists():
        from create_budget_template import create_budget_template
        create_budget_template(str(budget_file))
    
    return BudgetAdvisorAdapter(str(budget_file))


@pytest.fixture
def llm_client():
    """Get LLM client for test agents."""
    from rlm.utils.llm import get_llm_client
    return get_llm_client(None, os.getenv("RLM_ROOT_MODEL", ""))


@pytest.fixture
def runner(budget_advisor, llm_client):
    """Create scenario runner."""
    return ScenarioRunner(
        agent_under_test=budget_advisor,
        llm_client=llm_client,
        verbose=True
    )


# ============================================================================
# Test Scenarios
# ============================================================================

def test_affordable_purchase(runner):
    """Test that agent correctly identifies an affordable purchase."""
    
    scenario = TestScenario(
        name="Affordable Purchase - Smartwatch",
        description="User wants to buy a smartwatch for $250",
        initial_message="I'm thinking about buying a smartwatch. It costs 250 dollars. Can I afford it?",
        criteria=[
            "Agent clearly states whether the purchase is affordable or not",
            "Agent provides specific budget numbers (income, expenses, available funds)",
            "Agent gives a confidence level or recommendation",
            "Response is under 500 words and easy to understand",
            "Agent shows consideration for the user's financial safety"
        ],
        max_turns=1,
        user_persona="default"
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_unaffordable_purchase(runner):
    """Test that agent correctly identifies an unaffordable purchase."""
    
    scenario = TestScenario(
        name="Unaffordable Purchase - Expensive Vacation",
        description="User wants to buy something they cannot afford",
        initial_message="I want to go on a $5000 vacation. Should I book it?",
        criteria=[
            "Agent clearly states the purchase is NOT affordable",
            "Agent explains WHY it's not affordable (with numbers)",
            "Agent provides alternatives or suggestions",
            "Tone is empathetic, not judgmental",
            "Agent mentions how much more money is needed (if applicable)"
        ],
        max_turns=1,
        user_persona="impulsive"
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_borderline_purchase(runner):
    """Test agent's handling of a borderline-affordable purchase."""
    
    scenario = TestScenario(
        name="Borderline Purchase - Laptop",
        description="User wants something that's technically affordable but uses most of their budget",
        initial_message="I need a new laptop for work. Found one for $800. What do you think?",
        criteria=[
            "Agent provides a nuanced answer (not just yes/no)",
            "Agent mentions this is close to the budget limit",
            "Agent asks if this is a priority or essential purchase",
            "Agent provides a confidence score or caveat",
            "Agent suggests considerations before making the purchase"
        ],
        max_turns=2,
        user_persona="default"
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_ambiguous_query(runner):
    """Test how agent handles unclear or incomplete requests."""
    
    scenario = TestScenario(
        name="Ambiguous Query - Missing Price",
        description="User mentions wanting to buy something but doesn't specify price",
        initial_message="I'm thinking about getting a new phone. What do you think?",
        criteria=[
            "Agent asks for the price or budget range",
            "Agent doesn't make assumptions about affordability",
            "Response is helpful and guides the user to provide more info",
            "Tone is friendly and non-judgmental"
        ],
        max_turns=2,
        user_persona="confused"
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_multiple_purchases(runner):
    """Test agent's handling of multiple purchase queries in sequence."""
    
    scenario = TestScenario(
        name="Multiple Purchases - Sequential Queries",
        description="User asks about multiple different purchases",
        initial_message="Can I buy a $150 coffee maker?",
        criteria=[
            "Agent correctly handles each purchase query independently",
            "Agent provides clear yes/no answer with reasoning",
            "Budget numbers are consistent across responses",
            "Agent remembers context from previous interaction (if applicable)"
        ],
        max_turns=3,
        user_persona="default"
    )
    
    result = runner.run(scenario)
    # Note: This might fail if the agent doesn't track that available
    # funds decrease after each purchase - that's a valid test finding!
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_budget_summary_request(runner):
    """Test agent's ability to provide budget overview."""
    
    scenario = TestScenario(
        name="Budget Summary Request",
        description="User wants to see their overall budget situation",
        initial_message="Can you show me my budget summary?",
        criteria=[
            "Agent provides monthly income",
            "Agent provides total expenses",
            "Agent calculates discretionary budget (income - expenses)",
            "Agent mentions current savings",
            "Summary is concise and well-formatted"
        ],
        max_turns=1,
        user_persona="default"
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_skeptical_user(runner):
    """Test agent's handling of a skeptical, risk-averse user."""
    
    scenario = TestScenario(
        name="Skeptical User - Risk Averse",
        description="User is very cautious about spending money",
        initial_message="I want to buy a $100 gadget but I'm worried about my finances. Is it safe?",
        criteria=[
            "Agent acknowledges the user's concerns",
            "Agent provides reassurance backed by numbers",
            "Agent mentions safety buffer or emergency fund status",
            "Tone is empathetic and supportive",
            "Agent doesn't dismiss concerns or push for the purchase"
        ],
        max_turns=2,
        user_persona="skeptical"
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


def test_typo_handling(runner):
    """Test agent's robustness to typos and informal language."""
    
    scenario = TestScenario(
        name="Typo Handling - Informal Language",
        description="User types with typos and informal language",
        initial_message="hey can i buy a smrtwatch? its like 250 bucks",
        criteria=[
            "Agent correctly understands the intent despite typos",
            "Agent extracts the item (smartwatch) and price ($250)",
            "Agent provides a proper analysis",
            "Agent doesn't comment on or correct the typos"
        ],
        max_turns=1,
        user_persona="default"
    )
    
    result = runner.run(scenario)
    assert result["passed"], f"Test failed: {result['evaluation']['feedback']}"


# ============================================================================
# Test Suite Runner
# ============================================================================

def test_full_suite(runner):
    """Run all tests and generate comprehensive report."""
    
    aggregator = TestResultAggregator()
    
    scenarios = [
        TestScenario(
            name="Test 1: Small Affordable Purchase",
            description="$50 item, well within budget",
            initial_message="Can I buy a $50 book?",
            criteria=[
                "Agent says yes/affordable",
                "Agent provides budget context",
                "Response is concise"
            ],
            max_turns=1
        ),
        TestScenario(
            name="Test 2: Large Unaffordable Purchase",
            description="$10,000 item, far exceeds budget",
            initial_message="Should I buy a $10,000 luxury watch?",
            criteria=[
                "Agent clearly says no/unaffordable",
                "Agent explains the deficit",
                "Tone is empathetic, not judgmental"
            ],
            max_turns=1
        ),
        TestScenario(
            name="Test 3: Borderline Case",
            description="Purchase at the edge of affordability",
            initial_message="I'm looking at an $800 laptop. Thoughts?",
            criteria=[
                "Agent provides nuanced response",
                "Agent mentions this is close to limit",
                "Agent asks about priority/necessity"
            ],
            max_turns=2
        ),
    ]
    
    for scenario in scenarios:
        result = runner.run(scenario)
        aggregator.add_result(result)
    
    aggregator.print_summary()
    aggregator.save_report("test_results/budget_advisor_report.json")
    
    # Assert overall pass rate is acceptable
    summary = aggregator.summary()
    assert summary["pass_rate"] >= 0.7, f"Pass rate too low: {summary['pass_rate']:.1%}"


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    # Run with: python test_budget_advisor.py
    pytest.main([__file__, "-v", "-s"])