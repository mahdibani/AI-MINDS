"""
budget_advisor.py – Autonomous Budget Analysis Agent

Automatically analyzes budget.xlsx and determines purchase feasibility.
Uses RLM with sub-agents for mathematical analysis and financial reasoning.

Usage:
    from budget_advisor import BudgetAdvisor
    
    advisor = BudgetAdvisor(budget_file="C:/Users/bani_/Downloads/budget.xlsx")
    result = advisor.can_afford("smartwatch", 250.00)
    print(result)
"""

from __future__ import annotations
import os, sys, re, json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

# Load environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure parser dependencies
def _ensure_deps():
    import subprocess
    for imp, pip in {"openpyxl":"openpyxl", "pandas":"pandas"}.items():
        try:
            __import__(imp)
        except ImportError:
            print(f"[advisor] Installing {pip}...")
            for cmd in (["uv","add",pip],[sys.executable,"-m","pip","install",pip,"-q"]):
                if subprocess.run(cmd, capture_output=True).returncode == 0:
                    break

_ensure_deps()

from rlm.fs_tools        import FSTools
from rlm.rlm_repl        import RLM_REPL
from rlm.utils.llm       import get_llm_client
from rlm.memory          import AgentMemory


class BudgetAdvisor:
    """
    Autonomous budget analysis agent.
    
    Features:
    - Automatically parses budget.xlsx
    - Calculates income, expenses, savings
    - Analyzes spending patterns
    - Determines purchase affordability
    - Provides financial recommendations
    """
    
    def __init__(
        self, 
        budget_file: str = "C:/Users/bani_/Downloads/budget.xlsx",
        user_id: str = "default_user",
        safety_buffer: float = 0.20  # 20% safety buffer by default
    ):
        self.budget_file = budget_file.replace("\\", "/")
        self.user_id = user_id
        self.safety_buffer = safety_buffer
        
        # Initialize components
        self.fs = FSTools(allowed_roots=[
            str(Path(budget_file).parent),
            "/tmp/rlm_scratch"
        ])
        self.memory = AgentMemory(user_id=user_id)
        
        # Budget data cache
        self._budget_data: Optional[Dict[str, Any]] = None
        self._last_analysis: Optional[str] = None
        
        print(f"[advisor] Initialized for: {self.budget_file}")
        print(f"[advisor] Safety buffer: {self.safety_buffer*100}%")
    
    
    def can_afford(
        self, 
        item: str, 
        price: float,
        category: str = "discretionary",
        explain: bool = True
    ) -> Dict[str, Any]:
        """
        Main entry point: Can the user afford this purchase?
        
        Args:
            item: Item name (e.g., "smartwatch")
            price: Price in dollars
            category: Spending category (discretionary, essential, etc.)
            explain: Include detailed explanation
            
        Returns:
            Dict with:
                - affordable: bool
                - confidence: float (0-1)
                - available_funds: float
                - recommendation: str
                - warnings: List[str]
                - explanation: str (if explain=True)
        """
        print(f"\n[advisor] Analyzing purchase: {item} (${price:.2f})")
        
        # Step 1: Load and analyze budget
        budget_analysis = self._analyze_budget()
        if not budget_analysis["success"]:
            return {
                "affordable": False,
                "confidence": 0.0,
                "recommendation": f"Cannot analyze budget: {budget_analysis['error']}",
                "warnings": ["Budget file unavailable or unreadable"],
                "explanation": ""
            }
        
        # Step 2: Run RLM analysis with sub-agents
        rlm_result = self._run_affordability_analysis(
            item=item,
            price=price,
            category=category,
            budget_data=budget_analysis["data"]
        )
        
        # Step 3: Parse and structure result
        result = self._parse_rlm_result(rlm_result, price)
        
        # Step 4: Add explanation if requested
        if explain:
            result["explanation"] = self._generate_explanation(
                result, item, price, budget_analysis["data"]
            )
        
        # Step 5: Store in memory
        self._store_decision(item, price, result)
        
        return result
    
    
    def _analyze_budget(self) -> Dict[str, Any]:
        """Load and parse budget.xlsx, extract key metrics."""
        try:
            # Check if file exists
            file_check = self.fs.exists(self.budget_file)
            if not file_check["exists"]:
                return {
                    "success": False,
                    "error": f"Budget file not found: {self.budget_file}"
                }
            
            # Parse Excel file
            print(f"[advisor] Parsing budget file...")
            parse_result = self.fs.parse(self.budget_file, max_chars=100_000)
            
            if parse_result["error"]:
                return {
                    "success": False,
                    "error": parse_result["error"]
                }
            
            # Cache for later use
            self._budget_data = {
                "raw_text": parse_result["text"],
                "file_size": parse_result["size"],
                "parsed_at": datetime.now().isoformat()
            }
            
            return {
                "success": True,
                "data": self._budget_data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    
    def _run_affordability_analysis(
        self,
        item: str,
        price: float,
        category: str,
        budget_data: Dict[str, Any]
    ) -> str:
        """Use RLM with sub-agents to perform financial analysis."""
        
        # Prepare RLM with budget context
        budget_path_var = self.budget_file
        
        rlm = RLM_REPL(
            allowed_roots=[str(Path(self.budget_file).parent), "/tmp/rlm_scratch"],
            max_iterations=5,
            extra_locals={
                "budget_path": budget_path_var,
                "item_name": item,
                "item_price": price,
                "safety_buffer": self.safety_buffer
            }
        )
        
        # Simplified analysis query that actually works
        analysis_query = f"""
Analyze if the user can afford to buy {item} for ${price:.2f}.

The Excel budget file path is stored in the variable 'budget_path'.

Execute this code to analyze the budget:

```python
import openpyxl
import json

# Load spreadsheet
wb = openpyxl.load_workbook(budget_path, data_only=True)
sheet = wb.active

# Extract data
budget = {{}}
for row in sheet.iter_rows(values_only=True):
    if row[0] and row[1] is not None:
        try:
            budget[str(row[0]).strip().lower()] = float(row[1])
        except:
            pass

# Calculate
income = budget.get('monthly income', 0)
expenses = sum(v for k,v in budget.items() if 'expense' in k or 'bill' in k or 'rent' in k)
savings = budget.get('current savings', 0) or budget.get('savings', 0)
discretionary = income - expenses
safe_money = discretionary * (1 - safety_buffer) + savings * (1 - safety_buffer)

# Determine affordability
affordable = item_price <= discretionary or item_price <= safe_money
confidence = 1.0 if item_price <= discretionary * 0.5 else (0.8 if item_price <= discretionary else (0.6 if item_price <= safe_money else 0.3))

# Result
result = {{
    "affordable": affordable,
    "confidence": confidence,
    "available_funds": safe_money,
    "monthly_income": income,
    "total_expenses": expenses,
    "discretionary_budget": discretionary,
    "current_savings": savings,
    "recommendation": f"You can afford {{item_name}}" if affordable else f"Cannot afford {{item_name}}",
    "warnings": [] if affordable else ["Exceeds budget"]
}}

print(json.dumps(result, indent=2))
```

FINAL(json.dumps(result, indent=2))
"""
        
        print(f"[advisor] Running RLM analysis...")
        result = rlm.completion(
            context=f"Budget analysis for {item}",
            query=analysis_query
        )
        
        return result or "{}"
    
    
    def _parse_rlm_result(self, rlm_output: str, price: float) -> Dict[str, Any]:
        """Extract structured data from RLM output."""
        try:
            # Try to find JSON in the output
            json_match = re.search(r'\{[\s\S]*\}', rlm_output)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "affordable": data.get("affordable", False),
                    "confidence": data.get("confidence", 0.0),
                    "available_funds": data.get("available_funds", 0.0),
                    "monthly_income": data.get("monthly_income", 0.0),
                    "total_expenses": data.get("total_expenses", 0.0),
                    "discretionary_budget": data.get("discretionary_budget", 0.0),
                    "current_savings": data.get("current_savings", 0.0),
                    "recommendation": data.get("recommendation", "Unable to determine"),
                    "warnings": data.get("warnings", []),
                }
        except json.JSONDecodeError:
            pass
        
        # Fallback: parse text output
        affordable = any(word in rlm_output.lower() for word in ["yes", "afford", "can buy"])
        
        return {
            "affordable": affordable,
            "confidence": 0.5,
            "available_funds": 0.0,
            "recommendation": rlm_output[:500] if rlm_output else "Analysis incomplete",
            "warnings": ["Could not parse detailed analysis"],
            "explanation": ""
        }
    
    
    def _generate_explanation(
        self,
        result: Dict[str, Any],
        item: str,
        price: float,
        budget_data: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation."""
        
        explanation_parts = []
        
        # Opening
        if result["affordable"]:
            explanation_parts.append(
                f"✓ Good news! You can afford the {item} (${price:.2f})."
            )
        else:
            explanation_parts.append(
                f"✗ Unfortunately, the {item} (${price:.2f}) exceeds your current budget."
            )
        
        # Financial breakdown
        if result.get("monthly_income", 0) > 0:
            explanation_parts.append(
                f"\nYour monthly income: ${result['monthly_income']:.2f}"
            )
            explanation_parts.append(
                f"Total expenses: ${result['total_expenses']:.2f}"
            )
            explanation_parts.append(
                f"Discretionary budget: ${result['discretionary_budget']:.2f}"
            )
            explanation_parts.append(
                f"Current savings: ${result['current_savings']:.2f}"
            )
        
        # Available funds
        if result.get("available_funds", 0) > 0:
            explanation_parts.append(
                f"\nTotal available (with {self.safety_buffer*100:.0f}% safety buffer): "
                f"${result['available_funds']:.2f}"
            )
        
        # Warnings
        if result.get("warnings"):
            explanation_parts.append("\n⚠️  Important considerations:")
            for warning in result["warnings"]:
                explanation_parts.append(f"  • {warning}")
        
        # Recommendation
        explanation_parts.append(f"\n💡 Recommendation: {result['recommendation']}")
        
        return "\n".join(explanation_parts)
    
    
    def _store_decision(self, item: str, price: float, result: Dict[str, Any]):
        """Store purchase decision in memory for future reference."""
        memory_text = (
            f"Purchase analysis: {item} (${price:.2f}) - "
            f"{'Affordable' if result['affordable'] else 'Not affordable'} "
            f"(confidence: {result['confidence']:.0%})"
        )
        
        self.memory.add(
            memory_text,
            metadata={
                "item": item,
                "price": price,
                "affordable": result["affordable"],
                "confidence": result["confidence"],
                "date": datetime.now().isoformat()
            }
        )
        print(f"[advisor] Stored decision in memory")
    
    
    def get_spending_history(self) -> list:
        """Retrieve past purchase decisions from memory."""
        return self.memory.get_all()
    
    
    def get_budget_summary(self) -> str:
        """Quick budget overview without purchase analysis."""
        analysis = self._analyze_budget()
        if not analysis["success"]:
            return f"Error: {analysis['error']}"
        
        # Use simple LLM call for summary
        llm = get_llm_client(None, os.getenv("RLM_ROOT_MODEL", ""))
        prompt = f"""Summarize this budget in 3-4 sentences:

{analysis['data']['raw_text'][:2000]}

Include: monthly income, total expenses, discretionary budget, and savings."""
        
        return llm.completion([{"role": "user", "content": prompt}])


# ============================================================================
# Interactive CLI
# ============================================================================

def interactive_mode():
    """Interactive budget advisor session."""
    print("\n" + "="*70)
    print("  💰 AI Budget Advisor")
    print("="*70)
    
    budget_file = input("\nBudget file path [C:/Users/bani_/Downloads/budget.xlsx]: ").strip()
    if not budget_file:
        budget_file = "C:/Users/bani_/Downloads/budget.xlsx"
    
    advisor = BudgetAdvisor(budget_file=budget_file)
    
    print("\n" + "─"*70)
    print("Commands:")
    print("  'buy <item> <price>' - Check if you can afford something")
    print("  'summary'            - Get budget overview")
    print("  'history'            - View past decisions")
    print("  'quit'               - Exit")
    print("─"*70)
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[advisor] Goodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("quit", "exit", "q"):
            print("[advisor] Goodbye!")
            break
        
        if user_input.lower() == "summary":
            print("\n" + advisor.get_budget_summary())
            continue
        
        if user_input.lower() == "history":
            history = advisor.get_spending_history()
            if not history:
                print("\nNo purchase history yet.")
            else:
                print(f"\n{len(history)} past decisions:")
                for i, item in enumerate(history, 1):
                    print(f"  {i}. {item.get('memory', str(item))[:100]}")
            continue
        
        # Parse "buy <item> <price>" command
        buy_match = re.match(r'buy\s+(.+?)\s+(\d+(?:\.\d+)?)', user_input, re.I)
        if buy_match:
            item = buy_match.group(1).strip()
            price = float(buy_match.group(2))
            
            result = advisor.can_afford(item, price, explain=True)
            
            print("\n" + "─"*70)
            print("ANALYSIS RESULT")
            print("─"*70)
            print(result.get("explanation", result["recommendation"]))
            print("─"*70)
            print(f"Confidence: {result['confidence']:.0%}")
            print(f"Available funds: ${result.get('available_funds', 0):.2f}")
            print("─"*70)
        else:
            print("\nUnknown command. Try: buy smartwatch 250")


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AI Budget Advisor")
    parser.add_argument("--budget", "-b", 
                       default="C:/Users/bani_/Downloads/budget.xlsx",
                       help="Path to budget.xlsx")
    parser.add_argument("--item", "-i", help="Item to purchase")
    parser.add_argument("--price", "-p", type=float, help="Price in dollars")
    parser.add_argument("--interactive", "-I", action="store_true",
                       help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.item and args.price:
        advisor = BudgetAdvisor(budget_file=args.budget)
        result = advisor.can_afford(args.item, args.price, explain=True)
        print("\n" + result.get("explanation", result["recommendation"]))
    else:
        print("Usage:")
        print("  python budget_advisor.py --interactive")
        print("  python budget_advisor.py --item smartwatch --price 250")


if __name__ == "__main__":
    main()