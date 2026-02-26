import json
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent


from src.framework.agent_registry import AgentRegistry

import logging

# Configure logger
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
        except Exception as e:
            print(f"Warning: Failed to register agent {self.name}: {e}")
        
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
        """Builds the dynamic system message."""
        current_prompt = self.system_prompt

        # Inject Context
        if self.context_keys:
            current_prompt += "\n\n### Context Data:"
            for key in self.context_keys:
                val = state.get(key)
                if val:
                    # Convert to string/json
                    if hasattr(val, "json"):
                        val_str = val.json()
                    elif isinstance(val, (dict, list)):
                        val_str = json.dumps(val, default=str)
                    else:
                        val_str = str(val)
                    current_prompt += f"\n- {key}: {val_str}"
        
        # Inject Instructions for Response Format
        if self.response_format:
             current_prompt += (
                f"\n\n### Output Requirement:\n"
                f"You MUST end your turn by calling the tool '{self.response_tool_name}'. "
                f"Pass the final data into '{self.response_tool_name}'."
            )
            
        return current_prompt

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the agent graph with the given state."""
        logger.info(f"[{self.name}] Starting run...")
        
        # 1. Prepare Inputs
        system_msg = self._get_system_message(state)
        # Use user_request if available, otherwise just a trigger
        user_request = state.get("user_request", "Proceed with the task.")

        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=str(user_request))
        ]

        # create_react_agent expects a state with "messages"
        inputs = {"messages": messages}

        try:
            result = self.graph.invoke(inputs)
            # Result state usually has "messages"
            result_messages = result.get("messages", [])
            last_message = result_messages[-1] if result_messages else None

            # 2. Process Structured Output
            if self.response_format and last_message:
                # Check for tool_calls in the last message
                if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
                    for tool_call in last_message.tool_calls:
                        if tool_call["name"] == self.response_tool_name:
                            # Parse arguments
                            try:
                                # Validate against pydantic schema
                                parsed_output = self.response_format(**tool_call["args"])
                                return {self.output_key: parsed_output}
                            except Exception as e:
                                logger.error(f"[{self.name}] Validation Error: {e}")
                                return {"error": f"Validation failed: {e}", "raw": tool_call["args"]}
                
                logger.warning(f"[{self.name}] Warning: '{self.response_tool_name}' not called.")
                # Fallback return content if tool wasn't called
                return {"error": "Tool not called", "content": last_message.content}

            # 3. Default Output (if no structured response required)
            # We explicitly strictly return a dict that matches what SupervisorState expects if we know it.
            # But the Agent usage in the plan implies returning the output key update.
            # If no output key/response format, we might strictly return messages if that was the design.
            # However, looking at the graph, nodes return updates to state.
            
            # If no response format, assume we return messages or some other key?
            # For this simplified framework, let's assume if no response_format, we return nothing or messages.
            return {"messages": result_messages}

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}", exc_info=True)
            raise e
