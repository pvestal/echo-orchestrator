"""TechLead Orchestrator Agent using stateless execution pattern."""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.harness_models import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession


from src.agents.actions.orchestrator_hub import OrchestratorHub
from src.agents.actions.parsing.action_handler import ActionHandler
from src.agents.actions.state_managers import ScratchpadManager, TodoManager
from src.agents.env_interaction.entities.conversation_history import ConversationHistory
from src.agents.env_interaction.entities.turn import Turn
from src.agents.env_interaction.turn_executor import TurnExecutor
from src.agents.actions.parsing.parser import SimpleActionParser

from src.agents.utils.llm_client import (
    count_input_tokens,
    count_output_tokens,
    get_llm_response,
)

from src.agents.state.orchestrator_state import OrchestratorState
from src.agents.env_interaction.command_executor import (
    CommandExecutor,
    DockerExecutor,
)
from src.misc.log_setup import setup_file_logging
from src.misc.turn_logger import TurnLogger
from src.agents.system_msgs.system_msg_loader import load_orchestrator_system_message

logger = logging.getLogger(__name__)
setup_file_logging("INFO")


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent coordinating tasks and subagents."""
    
    @staticmethod
    def name() -> str:
        return "OrchestratorAgent"
        
    def __init__(
        self,
        system_message_path: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        **kwargs  # Accept additional keyword arguments from terminal bench
    ):
        """Initialize the orchestrator.
        
        Args:
            system_message_path: Path to system message file
            model: LiteLLM model to use (overrides env var)
            temperature: Temperature for LLM (overrides env var)
            api_key: API key for LiteLLM (overrides env var)
            api_base: API base URL for LiteLLM (overrides env var)
        """
        # Store LLM configuration
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        
        logger.info(f"OrchestratorAgent initialized with model={model}, temperature={temperature}")
        
        # Load system message
        self.system_message = self._load_system_message(system_message_path)
        
        # These will be initialized in setup()
        self.orchestrator_hub = None
        self.conversation_history = None
        self.action_parser = None
        self.action_handler = None
        self.executor = None
        self.state = None
        
        # Track orchestrator's messages for token counting
        self.orchestrator_messages = []
        
        # Turn logger (will be initialized in perform_task)
        self.turn_logger = None
        self.logging_dir = None
        
        super().__init__(**kwargs)
    
    def _load_system_message(self, path: Optional[str]) -> str:
        if path:
            # If explicit path provided, load from that file
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Use default system message loader
            return load_orchestrator_system_message()
    
    def setup(self, command_executor: CommandExecutor, logging_dir: Optional[Path] = None):
        """Setup the orchestrator with the necessary components.
        
        Args:
            command_executor: The command executor to use
            logging_dir: Optional directory for logging
        """
        
        # Initialize components with the provided executor
        self.orchestrator_hub = OrchestratorHub()
        self.conversation_history = ConversationHistory()
        
        # Store logging directory
        self.logging_dir = logging_dir
        
        # Create action components
        self.action_parser = SimpleActionParser()
        self.action_handler = ActionHandler(
            executor=command_executor,
            todo_manager=TodoManager(),
            scratchpad_manager=ScratchpadManager(),
            orchestrator_hub=self.orchestrator_hub,
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
            logging_dir=logging_dir,  # Pass logging dir for subagent logging
        )
        
        self.executor = TurnExecutor(
            action_parser=self.action_parser,
            action_handler=self.action_handler,
        )
        
        # Track state
        self.state = OrchestratorState(
            orchestrator_hub=self.orchestrator_hub,
            conversation_history=self.conversation_history
        )

    
    def execute_turn(self, instruction: str, turn_num: int) -> Dict[str, Any]:
        logging.info(f"\n🟡 ORCHESTRATOR TURN {turn_num} STARTING")
        
        # Build user message with current state
        user_message = f"## Current Task\n{instruction}\n\n{self.state.to_prompt()}"
        
        # Get LLM response
        logging.info(f"🟡 ORCHESTRATOR: Getting LLM response...")
        llm_response = self._get_llm_response(user_message)
        logging.info(f"🟡 ORCHESTRATOR: LLM response received, executing actions...")
        
        # Execute actions from LLM response
        result = self.executor.execute(llm_response)
        logging.info(f"🟡 ORCHESTRATOR: Actions executed - Count: {len(result.actions_executed)}")
        
        # Check if any subagent trajectories were returned
        if result.subagent_trajectories:
            logging.info(f"🟡 ORCHESTRATOR: Received {len(result.subagent_trajectories)} subagent report(s)")
            for task_id, trajectory in result.subagent_trajectories.items():
                logging.info(f"   - Task {task_id}: {trajectory.get('title', 'Unknown')}")
        else:
            logging.info(f"🟡 ORCHESTRATOR: No subagent reports in this turn")
        
        # Create turn for history
        turn = Turn(
            llm_output=llm_response,
            actions_executed=result.actions_executed,
            env_responses=result.env_responses,
            subagent_trajectories=result.subagent_trajectories
        )
        
        # Add to conversation history
        self.conversation_history.add_turn(turn)
        
        # Log this turn if logger is available
        if self.turn_logger:
            turn_data = {
                "instruction": instruction,
                "user_message": user_message,
                "llm_response": llm_response,
                "actions_executed": [str(action) for action in result.actions_executed],
                "env_responses": result.env_responses,
                "subagent_trajectories": result.subagent_trajectories,
                "done": result.done,
                "finish_message": result.finish_message,
                "has_error": result.has_error,
                "state_snapshot": self.state.to_dict()
            }
            self.turn_logger.log_turn(turn_num, turn_data)
        
        # Update done state
        if result.done:
            self.state.done = True
            self.state.finish_message = result.finish_message
            logging.info(f"🟡 ORCHESTRATOR: Task marked as DONE - {result.finish_message}")
        else:
            logging.info(f"🟡 ORCHESTRATOR TURN {turn_num} COMPLETE - Continuing...\n")
        
        return {
            'done': result.done,
            'finish_message': result.finish_message,
            'has_error': result.has_error,
            'actions_executed': len(result.actions_executed),
            'turn': turn
        }
    
    def _get_llm_response(self, user_message: str) -> str:
        # Build messages for this request
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message}
        ]
        
        # Track messages for token counting (add system message only once)
        if not self.orchestrator_messages:
            self.orchestrator_messages.append({"role": "system", "content": self.system_message})
        self.orchestrator_messages.append({"role": "user", "content": user_message})
        
        # Call centralized LLM client
        response = get_llm_response(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            api_key=self.api_key,
            api_base=self.api_base
        )
        
        # Track assistant response
        self.orchestrator_messages.append({"role": "assistant", "content": response})
        
        return response
    
    def run(self, instruction: str, max_turns: int = 50) -> Dict[str, Any]:
        """Run the orchestrator until completion or max turns.
        
        Args:
            instruction: The main task to complete
            max_turns: Maximum number of turns before stopping
            
        Returns:
            Final execution summary
        """
        turns_executed = 0
        
        while not self.state.done and turns_executed < max_turns:
            turns_executed += 1
            logger.info(f"Executing turn {turns_executed}")
            logging.info(f"\n{'='*60}")
            logging.info(f"ORCHESTRATOR MAIN LOOP - Turn {turns_executed}/{max_turns}")
            logging.info(f"{'='*60}")
            
            try:
                result = self.execute_turn(instruction, turns_executed)
                
                if result['done']:
                    logger.info(f"Task completed: {result['finish_message']}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in turn {turns_executed}: {e}")
                # Could add error to conversation history here
                
        return {
            'completed': self.state.done,
            'finish_message': self.state.finish_message,
            'turns_executed': turns_executed,
            'max_turns_reached': turns_executed >= max_turns
        }
    
    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        """Execute the orchestrator task using the stateless execution pattern.
        
        Args:
            instruction: The task instruction to execute
            session: TmuxSession containing the container
            logging_dir: Optional directory for logging
            
        Returns:
            AgentResult with execution details
        """
        # Get container name from session
        container_name = session.container.name
        if not container_name:
            raise ValueError("Container name is required for DockerExecutor")
        
        # Set up logging
        timestamped_markers: List[Tuple[float, str]] = []
        log_file = None
        
        if logging_dir:
            logging_dir = Path(logging_dir)
            logging_dir.mkdir(exist_ok=True, parents=True)
            
            # Create a unique log file name with timestamp
            log_file = logging_dir / f"agent_conversation_{int(time.time())}.json"
            logger.info(f"Logging conversation to: {log_file}")
        
        # Create docker executor for this session
        docker_executor = DockerExecutor(container_name=container_name)
        
        # Setup the orchestrator with the docker executor and logging directory
        self.setup(docker_executor, logging_dir)
        
        # Initialize turn logger if logging directory provided
        if logging_dir:
            self.turn_logger = TurnLogger(logging_dir, "orchestrator")

        failure_mode = FailureMode.NONE
        total_input_tokens = 0
        total_output_tokens = 0
        
        try:
            # Run the orchestrator
            result = self.run(instruction, max_turns=50)
            
            # Calculate total tokens from all subagent executions
            subagent_input_tokens = 0
            subagent_output_tokens = 0
            
            # Iterate through conversation history to find all subagent trajectories
            if self.conversation_history and self.conversation_history.turns:
                for turn in self.conversation_history.turns:
                    if turn.subagent_trajectories:
                        for task_id, trajectory_data in turn.subagent_trajectories.items():
                            subagent_input_tokens += trajectory_data.get('total_input_tokens', 0)
                            subagent_output_tokens += trajectory_data.get('total_output_tokens', 0)
                            logger.info(f"Subagent {task_id} tokens - Input: {trajectory_data.get('total_input_tokens', 0)}, Output: {trajectory_data.get('total_output_tokens', 0)}")
            
            # Calculate orchestrator's own token usage
            orchestrator_input_tokens = count_input_tokens(self.orchestrator_messages, self.model)
            orchestrator_output_tokens = count_output_tokens(self.orchestrator_messages, self.model)
            
            # Add orchestrator's own token usage
            total_input_tokens = subagent_input_tokens + orchestrator_input_tokens
            total_output_tokens = subagent_output_tokens + orchestrator_output_tokens
            
            logger.info(f"Orchestrator tokens - Input: {orchestrator_input_tokens}, Output: {orchestrator_output_tokens}")
            logger.info(f"Total tokens (orchestrator + subagents) - Input: {total_input_tokens}, Output: {total_output_tokens}")
            
            # Determine failure mode based on result
            if result['completed']:
                failure_mode = FailureMode.NONE
            elif result['max_turns_reached']:
                failure_mode = FailureMode.AGENT_TIMEOUT
            else:
                failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
                
        except Exception as e:
            logger.exception(f"Error during orchestrator execution: {e}")
            failure_mode = FailureMode.UNKNOWN_AGENT_ERROR
        
        # Save conversation log if logging directory provided
        if log_file:
            data = {
                "instruction": instruction,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "failure_mode": failure_mode.value,
                "timestamped_markers": timestamped_markers,
                "final_state": self.state.to_dict() if self.state else "",
            }
            
            # Get path and save the log
            path = log_file.resolve().parent
            with open(path / 'conversation_log.json', 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Conversation log exported to: {path / 'conversation_log.json'}")

        return AgentResult(
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            failure_mode=failure_mode,
            timestamped_markers=timestamped_markers,
        )
