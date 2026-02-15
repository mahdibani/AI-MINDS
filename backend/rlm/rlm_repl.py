"""
Recursive Language Model with REPL environment.
Implements the core RLM algorithm from the paper, extended with
sandboxed filesystem access via both:
  - REPL functions (fs_list, fs_read, fs_write, fs_exists, fs_info)
  - LLM tool-calling (OpenAI-compatible function-calling protocol)
"""

from typing import Dict, List, Optional, Any, Union
import json
import re

from rlm import RLM
from rlm.repl import REPLEnv
from rlm.fs_tools import FSTools, FS_TOOL_DEFINITIONS, dispatch_tool_call
from rlm.utils.tracing import tracer


class RLM_REPL(RLM):
    """
    RLM implementation using REPL environment with sandboxed filesystem access.

    Two complementary ways for the LLM to interact with the filesystem:

    1. REPL functions  – written inside ```repl ... ``` blocks, executed
       directly in the REPL sandbox:
           fs_list('/workspace/src')
           content = fs_read('/workspace/README.md')
           fs_write('/workspace/output/result.txt', content)

    2. Tool calls – standard OpenAI-style function-calling. The LLM emits
       a JSON tool-call object; rlm_repl intercepts it, dispatches to
       FSTools, and feeds the result back as a tool message. This is useful
       when the LLM is better at structured tool calls than inline Python.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5",
        recursive_model: str = "gpt-5-mini",
        max_iterations: int = 20,
        max_output_length: int = 500_000,
        depth: int = 0,
        allowed_roots: Optional[List[str]] = None,
    ):
        """
        Initialize RLM with REPL + filesystem tools.

        Args:
            api_key:        API key for LLM provider.
            model:          Root LLM model name.
            recursive_model: Sub-LLM model name for recursive calls.
            max_iterations: Maximum root LLM iterations before timeout.
            max_output_length: Max REPL output length before truncation.
            depth:          Current recursion depth (0 = root).
            allowed_roots:  Whitelist of directories the LLM may access.
                            Defaults to ['/workspace', '/tmp/rlm', '~/workspace'].
        """
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.max_iterations = max_iterations
        self.max_output_length = max_output_length
        self.depth = depth
        self.allowed_roots = allowed_roots  # forwarded to REPLEnv → FSTools

        from rlm.utils.llm import get_llm_client
        self.llm = get_llm_client(api_key, model)

        # Cost / usage tracking
        self._root_llm_cost = 0.0
        self._sub_llm_cost  = 0.0
        self._root_llm_tokens = 0
        self._sub_llm_tokens  = 0
        self._root_llm_calls  = 0
        self._sub_llm_calls   = 0

        # State
        self.repl_env: Optional[REPLEnv] = None
        self.messages: List[Dict[str, str]] = []
        self.query: Optional[str] = None

        # Shared FSTools instance (also used by the tool-calling layer)
        self._fs = FSTools(allowed_roots=allowed_roots)

    # ------------------------------------------------------------------
    # Context setup
    # ------------------------------------------------------------------

    def _setup_context(
        self,
        context: Union[List[str], str, List[Dict[str, str]]],
        query: str,
    ):
        """Set up the REPL environment with context and filesystem tools."""
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
        """
        Extract tool calls from a model response.

        Supports two formats:
          1. Native OpenAI-style tool_calls in the response dict (when the
             LLM client surfaces them — future work).
          2. JSON embedded in the response text inside a
             ```tool_call ... ``` fenced block, e.g.:

             ```tool_call
             {"name": "fs_read", "arguments": {"path": "/workspace/README.md"}}
             ```

        Returns a list of {"name": str, "arguments": dict} dicts, or None.
        """
        # Look for ```tool_call ... ``` blocks
        pattern = r'```tool_call\s*\n(.*?)\n```'
        calls = []
        for match in re.finditer(pattern, response, re.DOTALL):
            raw = match.group(1).strip()
            try:
                obj = json.loads(raw)
                # Accept both single call and list-of-calls
                if isinstance(obj, list):
                    calls.extend(obj)
                elif isinstance(obj, dict) and "name" in obj:
                    calls.append(obj)
            except json.JSONDecodeError:
                pass
        return calls if calls else None

    def _dispatch_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Execute a list of tool calls against the sandboxed FSTools instance.

        Returns a list of message dicts (role=tool) to append to self.messages.
        """
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
    # REPL code-block helpers
    # ------------------------------------------------------------------

    def _find_code_blocks(self, text: str) -> Optional[List[str]]:
        pattern = r'```repl\s*\n(.*?)\n```'
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
            parts.append(f"\n{result.stdout}")
        if result.stderr:
            parts.append(f"\nError: {result.stderr}")

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
            parts.append(f"\nREPL variables: {list(important_vars.keys())}")

        formatted = "\n".join(parts) if parts else "No output"
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
                self.messages.append({
                    "role": "user",
                    "content": f"Code executed:\n```python\n{code}\n```\n\nREPL output:\n{execution_result}",
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
            variable_name = content.strip().strip('"').strip("'").strip('\n').strip('\r')
            if variable_name in self.repl_env.locals:
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
        """
        Generate a completion using RLM with REPL + filesystem tools.

        The loop handles three kinds of model output per iteration:
          1. ```repl ... ```  blocks  → executed in the REPL sandbox
          2. ```tool_call ... ``` blocks → dispatched to FSTools directly
          3. FINAL(...) / FINAL_VAR(...)  → terminates and returns answer

        Args:
            context: Context to process (can be arbitrarily long).
            query:   Query to answer.

        Returns:
            Final answer string, or None if max_iterations reached.
        """
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

            # ---- Handle tool calls (```tool_call``` blocks) --------------
            tool_calls = self._extract_tool_calls(response)
            if tool_calls:
                print(f"[iter {iteration}] dispatching {len(tool_calls)} tool call(s)")
                tool_messages = self._dispatch_tool_calls(tool_calls)
                # Add the assistant turn then all tool results
                self.messages.append({"role": "assistant", "content": response})
                self.messages.extend(tool_messages)

            # ---- Handle REPL code blocks ---------------------------------
            code_blocks = self._find_code_blocks(response)
            execution_results = []
            if code_blocks:
                self.messages, execution_results = self._process_code_execution_with_results(response)
            elif not tool_calls:
                # Plain text response, no code, no tool call
                self.messages.append({"role": "assistant", "content": "You responded with:\n" + response})

            # ---- Tracing -------------------------------------------------
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

            # ---- Check for final answer ----------------------------------
            final_answer = self._check_final_answer(response)
            if final_answer:
                tracer.log_turn(
                    iteration=iteration,
                    messages=self.messages,
                    response=response,
                    code_blocks=code_blocks or [],
                    execution_results=execution_results,
                    final_answer=final_answer,
                    repl_state=repl_state,
                    cost_info=cost_info,
                )
                return final_answer

            # Check REPL locals for FINAL(...)
            if self.repl_env and hasattr(self.repl_env, 'locals'):
                for var_name, var_value in self.repl_env.locals.items():
                    if isinstance(var_value, str) and var_value.startswith('FINAL(') and var_value.endswith(')'):
                        actual_answer = var_value[6:-1]
                        tracer.log_turn(
                            iteration=iteration,
                            messages=self.messages,
                            response=response,
                            code_blocks=code_blocks or [],
                            execution_results=execution_results,
                            final_answer=actual_answer,
                            repl_state=repl_state,
                            cost_info=cost_info,
                        )
                        return actual_answer

        print(f"Warning: RLM reached max iterations ({self.max_iterations}) without a final answer.")
        tracer.log_turn(
            iteration=self.max_iterations,
            messages=self.messages,
            response="",
            code_blocks=[],
            execution_results=[],
            final_answer=None,
            repl_state={
                'context_loaded': 'context' in (self.repl_env.locals if self.repl_env else {}),
                'local_vars': list(self.repl_env.locals.keys()) if self.repl_env else [],
            },
            cost_info={'cost': 0, 'tokens': 0},
        )
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