"""
test_agents.py - Agent-based testing framework for RLM

Uses the LangWatch Scenario pattern:
1. Agent Under Test (AUT) - The agent we're testing
2. User Simulator Agent - Simulates real user behavior
3. Judge Agent - Evaluates the interaction

This allows us to test agents with natural language criteria instead of
exact string matching.
"""

from __future__ import annotations
import json
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ============================================================================
# Agent Adapter Protocol (for Scenario compatibility)
# ============================================================================

class AgentAdapter(ABC):
    """
    Base class for agents to be tested.
    
    Any agent implementing this can be plugged into the testing framework.
    """
    
    @abstractmethod
    def call(self, input: str | Dict[str, Any]) -> str:
        """
        Execute the agent with given input.
        
        Args:
            input: User message or structured data
            
        Returns:
            Agent's response as string
        """
        pass
    
    def reset(self):
        """Optional: Reset agent state between tests."""
        pass


# ============================================================================
# User Simulator Agent
# ============================================================================

class UserSimulatorAgent:
    """
    Simulates realistic user behavior for testing.
    
    Can generate:
    - Initial queries
    - Follow-up questions
    - Edge cases (typos, ambiguous requests, etc.)
    """
    
    def __init__(self, llm_client, persona: str = "default"):
        self.llm = llm_client
        self.persona = persona
        self.conversation_history = []
    
    def simulate_user_message(
        self,
        scenario: str,
        agent_response: Optional[str] = None,
        turn: int = 1
    ) -> str:
        """
        Generate a realistic user message.
        
        Args:
            scenario: What the user is trying to accomplish
            agent_response: Previous agent response (for follow-ups)
            turn: Conversation turn number
            
        Returns:
            Simulated user message
        """
        
        if turn == 1:
            # Initial message
            prompt = f"""You are simulating a real user.

SCENARIO: {scenario}

PERSONA: {self._get_persona_description()}

Generate a realistic first message this user would send.
Make it natural and conversational. Include any typos or informal language
the persona would use.

Respond with ONLY the user's message, nothing else."""
        
        else:
            # Follow-up message
            prompt = f"""You are simulating a real user in an ongoing conversation.

SCENARIO: {scenario}
PERSONA: {self._get_persona_description()}

AGENT'S LAST RESPONSE:
{agent_response}

CONVERSATION SO FAR:
{self._format_history()}

Generate the user's next realistic response. They might:
- Ask a follow-up question
- Request clarification
- Express concern or excitement
- Provide additional information
- Thank the agent or end the conversation

Respond with ONLY the user's message, nothing else."""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.completion(messages)
        
        # Store in history
        if agent_response:
            self.conversation_history.append({"role": "agent", "content": agent_response})
        self.conversation_history.append({"role": "user", "content": response})
        
        return response.strip()
    
    def _get_persona_description(self) -> str:
        """Return description of the simulated persona."""
        personas = {
            "default": "Average user, clear communicator, asks reasonable questions",
            "novice": "Not tech-savvy, asks basic questions, needs lots of explanation",
            "expert": "Knowledgeable, asks detailed questions, expects precision",
            "skeptical": "Questions recommendations, wants proof, risk-averse",
            "impulsive": "Makes quick decisions, doesn't read details, impatient",
            "confused": "Unclear requests, typos, needs guidance to clarify intent",
        }
        return personas.get(self.persona, personas["default"])
    
    def _format_history(self) -> str:
        """Format conversation history for context."""
        lines = []
        for msg in self.conversation_history[-4:]:  # Last 4 messages
            role = msg["role"].upper()
            lines.append(f"{role}: {msg['content'][:100]}")
        return "\n".join(lines)
    
    def reset(self):
        """Clear conversation history."""
        self.conversation_history = []


# ============================================================================
# Judge Agent
# ============================================================================

class JudgeAgent:
    """
    Evaluates agent performance using natural language criteria.
    
    Instead of exact string matching, uses LLM to judge if the agent's
    response meets specified requirements.
    """
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def evaluate(
        self,
        criteria: List[str],
        user_message: str,
        agent_response: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate agent response against criteria.
        
        Args:
            criteria: List of requirements in natural language
            user_message: What the user asked
            agent_response: What the agent responded
            context: Additional context for evaluation
            
        Returns:
            Dict with:
                - passed: bool
                - score: float (0-1)
                - feedback: str
                - criteria_results: Dict[str, bool]
        """
        
        # Format criteria
        criteria_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(criteria))
        
        # Build evaluation prompt
        prompt = f"""You are an expert judge evaluating an AI agent's response.

USER REQUEST:
{user_message}

AGENT RESPONSE:
{agent_response}

{self._format_context(context)}

EVALUATION CRITERIA:
{criteria_text}

For each criterion, determine if the agent's response meets it.

Respond in JSON format:
{{
    "overall_passed": true/false,
    "score": 0.0-1.0,
    "criteria_results": {{
        "criterion_1": {{"passed": true/false, "reason": "explanation"}},
        "criterion_2": {{"passed": true/false, "reason": "explanation"}},
        ...
    }},
    "feedback": "Overall assessment and suggestions for improvement"
}}

Be objective and fair. If information is missing but not explicitly required,
don't fail the criterion. Focus on what was delivered."""
        
        messages = [{"role": "user", "content": prompt}]
        response = self.llm.completion(messages)
        
        # Parse JSON response
        try:
            result = json.loads(self._extract_json(response))
            return {
                "passed": result.get("overall_passed", False),
                "score": result.get("score", 0.0),
                "feedback": result.get("feedback", "No feedback provided"),
                "criteria_results": result.get("criteria_results", {}),
            }
        except json.JSONDecodeError:
            return {
                "passed": False,
                "score": 0.0,
                "feedback": f"Failed to parse judge response: {response[:200]}",
                "criteria_results": {},
            }
    
    def _format_context(self, context: Optional[Dict]) -> str:
        """Format additional context for evaluation."""
        if not context:
            return ""
        
        lines = ["ADDITIONAL CONTEXT:"]
        for key, value in context.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def _extract_json(self, text: str) -> str:
        """Extract JSON from markdown code blocks or mixed text."""
        import re
        
        # Try to find JSON in code block
        match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Try to find raw JSON
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return match.group(0)
        
        return text


# ============================================================================
# Scenario Runner (orchestrates the test)
# ============================================================================

@dataclass
class TestScenario:
    """Configuration for a test scenario."""
    name: str
    description: str
    initial_message: str
    criteria: List[str]
    max_turns: int = 3
    context: Optional[Dict[str, Any]] = None
    user_persona: str = "default"


class ScenarioRunner:
    """
    Orchestrates agent testing using the three-agent pattern:
    1. Agent Under Test
    2. User Simulator
    3. Judge
    """
    
    def __init__(
        self,
        agent_under_test: AgentAdapter,
        llm_client,
        verbose: bool = True
    ):
        self.agent = agent_under_test
        self.user_simulator = UserSimulatorAgent(llm_client)
        self.judge = JudgeAgent(llm_client)
        self.verbose = verbose
    
    def run(self, scenario: TestScenario) -> Dict[str, Any]:
        """
        Run a test scenario.
        
        Returns:
            Dict with test results, transcript, and evaluation
        """
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"TEST: {scenario.name}")
            print(f"{'='*70}")
            print(f"Description: {scenario.description}\n")
        
        # Reset agents
        self.agent.reset()
        self.user_simulator.reset()
        self.user_simulator.persona = scenario.user_persona
        
        # Conversation transcript
        transcript = []
        
        # Turn 1: Initial interaction
        user_msg = scenario.initial_message
        transcript.append({"turn": 1, "role": "user", "content": user_msg})
        
        if self.verbose:
            print(f"USER: {user_msg}")
        
        agent_response = self.agent.call(user_msg)
        transcript.append({"turn": 1, "role": "agent", "content": agent_response})
        
        if self.verbose:
            print(f"AGENT: {agent_response[:200]}...\n")
        
        # Additional turns (if specified)
        for turn in range(2, scenario.max_turns + 1):
            user_msg = self.user_simulator.simulate_user_message(
                scenario=scenario.description,
                agent_response=agent_response,
                turn=turn
            )
            transcript.append({"turn": turn, "role": "user", "content": user_msg})
            
            if self.verbose:
                print(f"USER (turn {turn}): {user_msg}")
            
            agent_response = self.agent.call(user_msg)
            transcript.append({"turn": turn, "role": "agent", "content": agent_response})
            
            if self.verbose:
                print(f"AGENT (turn {turn}): {agent_response[:200]}...\n")
        
        # Evaluate using Judge
        if self.verbose:
            print(f"\n{'─'*70}")
            print("EVALUATION")
            print(f"{'─'*70}\n")
        
        evaluation = self.judge.evaluate(
            criteria=scenario.criteria,
            user_message=transcript[0]["content"],
            agent_response=transcript[1]["content"],
            context=scenario.context
        )
        
        if self.verbose:
            self._print_evaluation(evaluation)
        
        return {
            "scenario": scenario.name,
            "transcript": transcript,
            "evaluation": evaluation,
            "passed": evaluation["passed"],
        }
    
    def _print_evaluation(self, evaluation: Dict[str, Any]):
        """Pretty-print evaluation results."""
        status = "✅ PASSED" if evaluation["passed"] else "❌ FAILED"
        print(f"Result: {status}")
        print(f"Score: {evaluation['score']:.2f}/1.00\n")
        
        print("Criteria Results:")
        for criterion, result in evaluation["criteria_results"].items():
            status_icon = "✓" if result["passed"] else "✗"
            print(f"  {status_icon} {criterion}")
            print(f"    Reason: {result['reason']}")
        
        print(f"\nFeedback:\n{evaluation['feedback']}")
        print(f"\n{'='*70}\n")


# ============================================================================
# Test Result Aggregator
# ============================================================================

class TestResultAggregator:
    """Aggregate and report on multiple test runs."""
    
    def __init__(self):
        self.results: List[Dict[str, Any]] = []
    
    def add_result(self, result: Dict[str, Any]):
        """Add a test result."""
        self.results.append(result)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        
        avg_score = sum(r["evaluation"]["score"] for r in self.results) / total
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total,
            "average_score": avg_score,
        }
    
    def print_summary(self):
        """Print formatted summary."""
        summary = self.summary()
        
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Total tests:     {summary['total']}")
        print(f"Passed:          {summary['passed']} ✅")
        print(f"Failed:          {summary['failed']} ❌")
        print(f"Pass rate:       {summary['pass_rate']:.1%}")
        print(f"Average score:   {summary['average_score']:.2f}/1.00")
        print("="*70 + "\n")
        
        # Show failed tests
        failed_tests = [r for r in self.results if not r["passed"]]
        if failed_tests:
            print("FAILED TESTS:")
            for result in failed_tests:
                print(f"  ❌ {result['scenario']}")
                print(f"     {result['evaluation']['feedback'][:100]}...")
            print()
    
    def save_report(self, filepath: str = "test_report.json"):
        """Save detailed report to file."""
        import json
        from pathlib import Path
        
        report = {
            "summary": self.summary(),
            "results": self.results,
        }
        
        Path(filepath).write_text(json.dumps(report, indent=2))
        print(f"📄 Detailed report saved to: {filepath}")