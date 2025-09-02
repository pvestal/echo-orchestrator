"""Stateless executor for single-turn agent execution with state management."""

import logging

from src.agents.actions.parsing.action_handler import ActionHandler
from src.agents.actions.parsing.parser import SimpleActionParser
from src.agents.actions.entities.actions import (
    FinishAction,
)
from src.agents.env_interaction.entities.execution_result import ExecutionResult

logger = logging.getLogger(__name__)


class TurnExecutor:
    """Executes a single turn of agent interaction"""
    
    def __init__(
        self,
        action_parser: SimpleActionParser,
        action_handler: ActionHandler,
    ):
        self.action_parser = action_parser
        self.action_handler = action_handler


    def execute(self, llm_output: str) -> ExecutionResult:
        """Execute actions from LLM output and return result.
        
        Args:
            llm_output: Raw output from the LLM
            
        Returns:
            ExecutionResult containing executed actions and responses
        """
        # Parse actions from LLM output
        actions, parsing_errors, found_action_attempt = self.action_parser.parse_response(
            llm_output
        )

        if not found_action_attempt:
            logger.warning("No actions attempted in response")
            return ExecutionResult(
                actions_executed=[],
                env_responses=["No actions were attempted."],
                has_error=True,
                done=True
            )
        
        # Track execution
        actions_executed = []
        env_responses = []
        has_error = False
        finish_message = None
        done = False
        
        # Handle parsing errors
        if parsing_errors:
            has_error = True
            # Include parsing errors in env responses
            for error in parsing_errors:
                env_responses.append(f"[PARSE ERROR] {error}")
            
            # If no valid actions were parsed, return early
            if not actions:
                return ExecutionResult(
                    actions_executed=[],
                    env_responses=env_responses,
                    has_error=True,
                    done=False
                )
        
        # Execute each action
        for action in actions:
            try:
                # Execute the action
                output, is_error = self.action_handler.handle_action(action)
                actions_executed.append(action)
                
                if is_error:
                    has_error = True
                
                env_responses.append(output)
                
                # Check for finish
                if isinstance(action, FinishAction):
                    finish_message = action.message
                    done = True
                    logger.info(f"Task finished: {finish_message}")
                    break
                    
            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                env_responses.append(f"[ERROR] Action execution failed: {str(e)}")
                has_error = True
        
        # Collect any subagent trajectories from this execution
        subagent_trajectories = self.action_handler.get_and_clear_subagent_trajectories()
        
        return ExecutionResult(
            actions_executed=actions_executed,
            env_responses=env_responses,
            has_error=has_error,
            finish_message=finish_message,
            done=done,
            subagent_trajectories=subagent_trajectories if subagent_trajectories else None
        )
