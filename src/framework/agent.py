import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent

from src.framework.agent_registry import AgentRegistry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class Agent:
    """
    A reusable Agent wrapper around langgraph's create_react_agent.
    """

    def __init__(
        self,
        name: str,
        model: BaseChatModel,
        tools: List[Any],
        system_prompt: str,
        response_format: Optional[Type[BaseModel]] = None,
        context_keys: Optional[List[str]] = None,
        output_key: Optional[str] = None
    ):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.response_format = response_format
        self.context_keys = context_keys or []
        self.output_key = output_key or "final_output"
        self.response_tool_name = "submit_final_report"
        
        try:
            description = self.system_prompt.split('\n')[0] if self.system_prompt else f"Agent {self.name}"
            AgentRegistry.register_agent(name=self.name, description=description)
        except Exception as exc:
            logger.warning("Failed to register agent %s: %s", self.name, exc)
        
        self.tools = list(tools)
        
        if self.response_format:
            self.response_tool = self._create_response_tool(self.response_format)
            self.tools.append(self.response_tool)
        
        self.graph = create_react_agent(self.model, self.tools)

    def _create_response_tool(self, schema: Type[T]) -> BaseTool:
        """Creates a tool that the LLM must call to submit its final answer."""
        def submit_final_report(**kwargs):
            """Call this tool to submit the final report/answer."""
            return "Output submitted successfully."

        return StructuredTool.from_function(
            func=submit_final_report,
            name=self.response_tool_name,
            description=f"Submit the final answer formatted as {schema.__name__}.",
            args_schema=schema,
        )

    def _get_system_message(self, state: Dict[str, Any]) -> str:
        """Build the dynamic system message with injected context and output instructions."""
        current_prompt = self.system_prompt

        if self.context_keys:
            current_prompt += "\n\n### Context Data:"
            for key in self.context_keys:
                val = state.get(key)
                if val:
                    if hasattr(val, "model_dump_json"):
                        val_str = val.model_dump_json()
                    elif isinstance(val, (dict, list)):
                        val_str = json.dumps(val, default=str)
                    else:
                        val_str = str(val)
                    current_prompt += f"\n- {key}: {val_str}"

        if self.response_format:
            current_prompt += (
                f"\n\n### Output Requirement:\n"
                f"You MUST end your turn by calling the tool '{self.response_tool_name}'. "
                f"Pass the final data into '{self.response_tool_name}'."
            )

        return current_prompt

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Run the agent graph and return the output keyed by ``output_key``."""
        logger.info("[%s] Starting run...", self.name)

        system_msg = self._get_system_message(state)
        user_request = state.get("user_request", "Proceed with the task.")

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=str(user_request)),
        ]
        inputs = {"messages": messages}

        try:
            result = self.graph.invoke(inputs)
            result_messages = result.get("messages", [])
            last_message = result_messages[-1] if result_messages else None

            if self.response_format and last_message:
                if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
                    for tool_call in last_message.tool_calls:
                        if tool_call["name"] == self.response_tool_name:
                            try:
                                parsed_output = self.response_format(**tool_call["args"])
                                return {self.output_key: parsed_output}
                            except Exception as exc:
                                logger.error("[%s] Validation error: %s", self.name, exc)
                                return {"error": f"Validation failed: {exc}", "raw": tool_call["args"]}

                logger.warning("[%s] '%s' tool was not called by the model.", self.name, self.response_tool_name)
                return {"error": "Tool not called", "content": last_message.content}

            return {"messages": result_messages}

        except Exception as exc:
            logger.error("[%s] Error: %s", self.name, exc, exc_info=True)
            raise
