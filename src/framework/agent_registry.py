from typing import Dict, List, TypedDict, Optional

class AgentMetadata(TypedDict):
    name: str
    description: str

class AgentRegistry:
    _agents: Dict[str, AgentMetadata] = {}

    @classmethod
    def register_agent(cls, name: str, description: str):
        """Register a new agent with its name and description (usually from system prompt)."""
        cls._agents[name] = {"name": name, "description": description}

    @classmethod
    def get_all_agents(cls) -> List[AgentMetadata]:
        """Retrieve metadata for all registered agents."""
        return list(cls._agents.values())

    @classmethod
    def get_agent(cls, name: str) -> Optional[AgentMetadata]:
        return cls._agents.get(name)
