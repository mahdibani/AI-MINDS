"""
Recursive Language Model with REPL environment.

Patches vs original:
  1. _find_code_blocks accepts ```python / ```py / ```Python in addition to
     ```repl – small local models frequently use the wrong fence tag.
  2. REPL output (stdout + stderr) is always fed back to the model as a
     "user" message so it can react to errors instead of looping blindly.
  3. RLM_REPL.__init__ accepts an `extra_locals` dict that run.py uses to
     pre-inject the Downloads path as a clean forward-slash string variable,
     avoiding backslash-escape bugs entirely.
"""

from typing import Dict, List, Optional, Any, Union
import json
import re

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.fs_tools import FSTools, FS_TOOL_DEFINITIONS, dispatch_tool_call
from rlm.utils.tracing import tracer


class RLM_REPL(RLM):

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        recursive_model: str = "gpt-5-mini",
        max_iterations: int = 20,
        max_output_length: int = 500_000,
        depth: int = 0,
        allowed_roots: Optional[List[str]] = None,
        extra_locals: Optional[Dict[str, Any]] = None,   # ← NEW
    ):
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.max_iterations = max_iterations
        self.max_output_length = max_output_length
        self.depth = depth
        self.allowed_roots = allowed_roots
        self.extra_locals = extra_locals or {}            # ← NEW

        from rlm.utils.llm import get_llm_client
        self.llm = get_llm_client(api_key, model)

        self._root_llm_cost = 0.0
        self._sub_llm_cost  = 0.0
        self._root_llm_tokens = 0
        self._sub_llm_tokens  = 0
        self._root_llm_calls  = 0
        self._sub_llm_calls   = 0

        self.repl_env: Optional[REPLEnv] = None
        self.messages: List[Dict[str, str]] = []
        self.query: Optional[str] = None

        self._fs = FSTools(allowed_roots=allowed_roots)

    # ------------------------------------------------------------------
    # Context setup
    # ------------------------------------------------------------------

    def _setup_context(
        self,
        context: Union[List[str], str, List[Dict[str, str]]],
        query: str,
    ):
        print("_setup_context called")
        self.query = query
        self.messages = []

        from rlm.utils.prompts import build_system_prompt
        self.messages = build_system_prompt(self.model)

        context_data, context_str = self._convert_context(context)
        context_type, context_lengths, context_total_length = self._get_context_metadata(
            context, context_data, context_str
        )
        print(f"Metadata: type={context_type}, total={context_total_length}")

        def llm_query_fn(prompt: str) -> str:
            return self._recursive_llm_call(prompt)

        self.repl_env = REPLEnv(
            llm_query_fn=llm_query_fn,
            context_json=context_data,
            context_str=context_str,
            allowed_roots=self.allowed_roots,
        )

        # ── Inject extra locals (e.g. pre-normalised paths) ──────────────
        if self.extra_locals:
            self.repl_env.locals.update(self.extra_locals)
            print(f"REPL environment initialized (fs tools available, "
                  f"extra locals: {list(self.extra_locals.keys())})")
        else:
            print("REPL environment initialized (fs tools available)")

        from rlm.utils.prompts import add_context_metadata
        self.messages = add_context_metadata(
            self.messages, context_type, context_lengths, context_total_length
        )

    def _convert_context(self, context):
        if isinstance(context, dict):
            return context, None
        elif isinstance(context, str):
            return None, context
        elif isinstance(context, list):
            if len(context) > 0 and isinstance(context[0], dict):
                if "content" in context[0]:
                    return [msg.get("content", "") for msg in context], None
                return context, None
            return context, None
        return context, None

    def _get_context_metadata(self, context, context_data, context_str):
        if context_str is not None:
            return "str", [len(context_str)], len(context_str)
        elif context_data is not None:
            if isinstance(context_data, list):
                lengths = [len(str(item)) for item in context_data]
                return "list", lengths, sum(lengths)
            elif isinstance(context_data, dict):
                l = len(str(context_data))
                return "dict", [l], l
            else:
                l = len(str(context_data))
                return type(context_data).__name__, [l], l
        return "unknown", [0], 0

    # ------------------------------------------------------------------
    # Recursive LLM call (sub-LLM)
    # ------------------------------------------------------------------

    def _recursive_llm_call(self, prompt: str) -> str:
        from rlm.utils.llm import get_llm_client
        sub_llm = get_llm_client(self.api_key, self.recursive_model)
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        response, cost_info = sub_llm.completion_with_cost(messages)
        self._sub_llm_cost   += cost_info['cost']
        self._sub_llm_tokens += cost_info['tokens']
        self._sub_llm_calls  += 1
        return response

    # ------------------------------------------------------------------
    # Tool-calling layer
    # ------------------------------------------------------------------

    def _extract_tool_calls(self, response: str) -> Optional[List[Dict[str, Any]]]:
        pattern = r'```tool_call\s*\n(.*?)\n```'
        calls = []
        for match in re.finditer(pattern, response, re.DOTALL):
            raw = match.group(1).strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, list):
                    calls.extend(obj)
                elif isinstance(obj, dict) and "name" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass
        return calls if calls else None

    def _dispatch_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        tool_messages = []
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            print(f"[tool-call] {name}({args})")
            result_json = dispatch_tool_call(self._fs, name, args)
            tool_messages.append({
                "role": "tool",
                "name": name,
                "content": result_json,
            })
        return tool_messages

    # ------------------------------------------------------------------
    # REPL code-block helpers  (FIX 1: accept python/py/Python tags)
    # ------------------------------------------------------------------

    def _find_code_blocks(self, text: str) -> Optional[List[str]]:
        # Accept ```repl, ```python, ```py, ```Python, or plain ``` blocks
        pattern = r'```(?:repl|python|py|Python)?\s*\n(.*?)\n```'
        results = [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL)]
        return results if results else None

    def _find_final_answer(self, text: str) -> Optional[tuple]:
        m = re.search(r'^\s*FINAL_VAR\((.*?)\)', text, re.MULTILINE | re.DOTALL)
        if m:
            return ('FINAL_VAR', m.group(1).strip())
        m = re.search(r'^\s*FINAL\((.*?)\)', text, re.MULTILINE | re.DOTALL)
        if m:
            return ('FINAL', m.group(1).strip())
        return None

    def _execute_code(self, code: str) -> str:
        result = self.repl_env.code_execution(code)
        parts = []
        if result.stdout:
            parts.append(result.stdout)
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")

        important_vars = {}
        for key, value in result.locals.items():
            if not key.startswith('_') and key not in ['__builtins__', '__name__', '__doc__']:
                try:
                    if isinstance(value, (str, int, float, bool, list, dict, tuple)):
                        important_vars[key] = (
                            f"'{value[:100]}...'" if isinstance(value, str) and len(value) > 100
                            else repr(value)
                        )
                except Exception:
                    important_vars[key] = f"<{type(value).__name__}>"

        if important_vars:
            parts.append(f"REPL variables: {list(important_vars.keys())}")

        formatted = "\n".join(parts) if parts else "(no output)"
        if len(formatted) > self.max_output_length:
            formatted = formatted[:self.max_output_length] + f"... [truncated from {len(formatted)} chars]"
        return formatted

    def _process_code_execution_with_results(self, response: str):
        code_blocks = self._find_code_blocks(response)
        execution_results = []
        if code_blocks:
            for code in code_blocks:
                execution_result = self._execute_code(code)
                execution_results.append(execution_result)
                # FIX 2: always echo output so the model can see errors
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"Code executed:\n```python\n{code}\n```\n\n"
                        f"REPL output:\n{execution_result}\n\n"
                        "If the output shows an error or the listing is empty, "
                        "diagnose and fix it. If output looks good, build the report "
                        "and emit FINAL_VAR('report') or FINAL(...)."
                    ),
                })
        return self.messages, execution_results

    def _check_final_answer(self, response: str) -> Optional[str]:
        result = self._find_final_answer(response)
        if result is None:
            return None
        answer_type, content = result
        if answer_type == 'FINAL':
            return content
        elif answer_type == 'FINAL_VAR':
            variable_name = content.strip().strip('"').strip("'").strip()
            if self.repl_env and variable_name in self.repl_env.locals:
                return str(self.repl_env.locals[variable_name])
        return None

    # ------------------------------------------------------------------
    # Main completion loop
    # ------------------------------------------------------------------

    def completion(
        self,
        context: Union[List[str], str, List[Dict[str, str]]],
        query: str,
    ) -> Optional[str]:
        print("Starting RLM completion...")
        self._setup_context(context, query)

        for iteration in range(self.max_iterations):
            from rlm.utils.prompts import next_action_prompt
            user_prompt = next_action_prompt(query, iteration)

            response, cost_info = self.llm.completion_with_cost(
                self.messages + [user_prompt]
            )
            print(f"[iter {iteration}] response[:120]: {response[:120]}...")

            self._root_llm_cost   += cost_info['cost']
            self._root_llm_tokens += cost_info['tokens']
            self._root_llm_calls  += 1

            # ---- Tool calls ---------------------------------------------
            tool_calls = self._extract_tool_calls(response)
            if tool_calls:
                print(f"[iter {iteration}] dispatching {len(tool_calls)} tool call(s)")
                tool_messages = self._dispatch_tool_calls(tool_calls)
                self.messages.append({"role": "assistant", "content": response})
                self.messages.extend(tool_messages)

            # ---- REPL code blocks ---------------------------------------
            code_blocks = self._find_code_blocks(response)
            execution_results = []
            if code_blocks:
                self.messages, execution_results = self._process_code_execution_with_results(response)
            elif not tool_calls:
                self.messages.append({"role": "assistant", "content": "You responded with:\n" + response})

            # ---- Tracing ------------------------------------------------
            repl_state = {}
            if self.repl_env:
                repl_state = {
                    'context_loaded': 'context' in self.repl_env.locals,
                    'local_vars': list(self.repl_env.locals.keys()),
                    'globals': list(self.repl_env.globals.keys()),
                }
            tracer.log_turn(
                iteration=iteration,
                messages=self.messages,
                response=response,
                code_blocks=code_blocks or [],
                execution_results=execution_results,
                repl_state=repl_state,
                cost_info=cost_info,
            )

            # ---- Check for final answer ---------------------------------
            final_answer = self._check_final_answer(response)
            if final_answer:
                return final_answer

            # Check REPL locals for FINAL(...)
            if self.repl_env and hasattr(self.repl_env, 'locals'):
                for var_name, var_value in self.repl_env.locals.items():
                    if isinstance(var_value, str) and var_value.startswith('FINAL(') and var_value.endswith(')'):
                        return var_value[6:-1]

        print(f"Warning: RLM reached max iterations ({self.max_iterations}) without a final answer.")
        return None

    # ------------------------------------------------------------------
    # Cost summary & reset
    # ------------------------------------------------------------------

    def cost_summary(self) -> Dict[str, Any]:
        return {
            'total_cost':       self._root_llm_cost + self._sub_llm_cost,
            'root_llm_cost':    self._root_llm_cost,
            'sub_llm_cost':     self._sub_llm_cost,
            'root_llm_tokens':  self._root_llm_tokens,
            'sub_llm_tokens':   self._sub_llm_tokens,
            'root_llm_calls':   self._root_llm_calls,
            'sub_llm_calls':    self._sub_llm_calls,
        }

    def reset(self):
        self.repl_env = None
        self.messages = []
        self.query    = None
        self._root_llm_cost   = 0.0
        self._sub_llm_cost    = 0.0
        self._root_llm_tokens = 0
        self._sub_llm_tokens  = 0
        self._root_llm_calls  = 0
        self._sub_llm_calls   = 0