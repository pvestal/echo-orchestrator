"""Subagent implementation for executing delegated tasks."""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from src.misc.turn_logger import TurnLogger
from src.agents.actions.parsing.action_handler import ActionHandler
from src.agents.actions.state_managers import ScratchpadManager, TodoManager
from src.agents.actions.parsing.parser import SimpleActionParser
from src.agents.env_interaction.command_executor import CommandExecutor
from src.agents.actions.entities.actions import ReportAction
from src.agents.actions.entities.subagent_report import ContextItem, SubagentMeta, SubagentReport
from src.agents.env_interaction.turn_executor import TurnExecutor
from src.agents.utils.llm_client import count_input_tokens, count_output_tokens, get_llm_response
from src.agents.system_msgs.system_msg_loader import load_coder_system_message, load_explorer_system_message



logger = logging.getLogger(__name__)


@dataclass
class SubagentTask:
    """Task specification for a subagent."""
    agent_type: str  # "explorer" or "coder"
    title: str
    description: str
    ctx_store_ctxts: Dict[str, str]  # Resolved context content from store
    bootstrap_ctxts: List[Dict[str, str]]  # List of {"path": str, "content": str, "reason": str}


class Subagent:
    """Executes a specific task delegated by the orchestrator."""
    
    def __init__(
        self,
        task: SubagentTask,
        executor: CommandExecutor,
        max_turns: int = 30,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        logging_dir: Optional[Path] = None,
        task_id: Optional[str] = None
    ):
        """Initialize the subagent.
        
        Args:
            task: The task specification
            executor: Command executor (shared with orchestrator)
            max_turns: Maximum turns before forcing completion
            model: LiteLLM model to use (overrides env var)
            temperature: Temperature for LLM (overrides env var)
            api_key: API key for LiteLLM (overrides env var)
            api_base: API base URL for LiteLLM (overrides env var)
            logging_dir: Optional directory for logging turns
            task_id: Optional task ID for log file naming
        """
        self.task = task
        self.max_turns = max_turns
        
        # Store LLM configuration
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        
        # Initialize components (own state, shared executor)
        self.action_parser = SimpleActionParser()
        self.action_handler = ActionHandler(
            executor=executor,
            todo_manager=TodoManager(),
            scratchpad_manager=ScratchpadManager(),
            orchestrator_hub=None,  # Subagents don't have access to hub
        )
        
        self.executor_stateless = TurnExecutor(
            action_parser=self.action_parser,
            action_handler=self.action_handler
        )
        
        # Load system message based on agent type
        self.system_message = self._load_system_message()
        
        # Track completion
        self.report: Optional[SubagentReport] = None
        self.messages: List[Dict[str, str]] = []
        
        # Initialize turn logger if logging directory provided
        self.turn_logger = None
        if logging_dir:
            prefix = f"subagent_{task_id}" if task_id else f"subagent_{task.agent_type}"
            self.turn_logger = TurnLogger(logging_dir, prefix)
    
    def _load_system_message(self) -> str:
        """Load system message based on agent type."""
        if self.task.agent_type == "explorer":
            return load_explorer_system_message()
        elif self.task.agent_type == "coder":
            return load_coder_system_message()
        else:
            raise ValueError(f"Unknown agent type: {self.task.agent_type}")
        
    def _build_task_prompt(self) -> str:
        """Build the initial task prompt with all context."""
        sections = []
        
        # Task description
        sections.append(f"# Task: {self.task.title}\n")
        sections.append(f"{self.task.description}\n")
        
        # Include resolved contexts
        if self.task.ctx_store_ctxts:
            sections.append("## Provided Context\n")
            for ctx_id, content in self.task.ctx_store_ctxts.items():
                sections.append(f"### Context: {ctx_id}\n")
                sections.append(f"{content}\n")
        
        # Include bootstrap files/dirs
        if self.task.bootstrap_ctxts:
            sections.append("## Relevant Files/Directories\n")
            for item in self.task.bootstrap_ctxts:
                sections.append(f"- {item['path']}: {item['reason']}\n")
        
        sections.append("\nBegin your investigation/implementation now.")
        
        return "\n".join(sections)
    
    def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Get response from LLM using centralized client."""
        # Call centralized LLM client
        return get_llm_response(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            max_tokens=4096,
            api_key=self.api_key,
            api_base=self.api_base
        )
    
    @property
    def total_input_tokens(self) -> int:
        """Calculate total input tokens from all messages."""
        return count_input_tokens(self.messages, self.model)
    
    @property
    def total_output_tokens(self) -> int:
        """Calculate total output tokens from all messages."""
        return count_output_tokens(self.messages, self.model)
    
    def _check_for_report(self, actions: List) -> Optional[SubagentReport]:
        """Check if any action is a ReportAction and convert to SubagentReport."""
        for action in actions:
            if isinstance(action, ReportAction):
                # Convert to SubagentReport
                contexts = [
                    ContextItem(id=ctx["id"], content=ctx["content"])
                    for ctx in action.contexts
                ]
                return SubagentReport(
                    contexts=contexts,
                    comments=action.comments,
                    meta=SubagentMeta(
                        trajectory=self.messages.copy() if hasattr(self, 'messages') else None,
                        total_input_tokens=0,  # Will be set in run()
                        total_output_tokens=0  # Will be set in run()
                    )
                )
        return None
    
    def run(self) -> SubagentReport:
        """Execute the task and return the report."""
        # Initialize message history
        self.messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self._build_task_prompt()}
        ]
        
        for turn_num in range(self.max_turns):
            logger.debug(f"Subagent {self.task.agent_type} executing turn {turn_num + 1}")
            
            try:
                # Get LLM response
                llm_response = self._get_llm_response(self.messages)

                logger.debug(f"--- Subagent Turn {turn_num + 1} ---")
                logger.debug(f"LLM Response:\n{llm_response}")
                
                # Add assistant response to message history
                self.messages.append({"role": "assistant", "content": llm_response})
                
                # Execute actions
                result = self.executor_stateless.execute(llm_response)
                
                # Add environment responses to message history first
                env_response = "\n".join(result.env_responses)
                self.messages.append({"role": "user", "content": env_response})
                logger.debug(f"Environment Response:\n{env_response}")
                
                # Log this turn if logger is available
                if self.turn_logger:
                    turn_data = {
                        "task_type": self.task.agent_type,
                        "task_title": self.task.title,
                        "llm_response": llm_response,
                        "actions_executed": [str(action) for action in result.actions_executed],
                        "env_responses": result.env_responses,
                        "messages_count": len(self.messages)
                    }
                    self.turn_logger.log_turn(turn_num + 1, turn_data)
                
                # Check for report action
                report = self._check_for_report(result.actions_executed)
                if report:
                    logging.info(f"\n🔵 SUBAGENT REPORT DETECTED - Turn {turn_num + 1}")
                    logging.info(f"   Agent Type: {self.task.agent_type}")
                    logging.info(f"   Task Title: {self.task.title}")
                    logging.info(f"   Comments: {report.comments[:200]}..." if len(report.comments) > 200 else f"   Comments: {report.comments}")
                    logging.info(f"   Contexts returned: {len(report.contexts)}")
                    
                    self.report = report
                    logging.debug(f"  Report set")
                    # Add metadata
                    report.meta.num_turns = turn_num + 1
                    report.meta.total_input_tokens = self.total_input_tokens
                    report.meta.total_output_tokens = self.total_output_tokens
                    logger.debug(f"Subagent completed with report: {report.comments}")
                    logger.debug(f"Token usage - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}")
                    
                    logging.info(f"🔵 SUBAGENT RETURNING REPORT TO ORCHESTRATOR\n")
                    return report
                    
            except Exception as e:
                logger.error(f"Error in subagent turn {turn_num + 1}: {e}")
                # Add error to message history and continue
                self.messages.append({"role": "user", "content": f"Error occurred: {str(e)}. Please continue."})
        
        # Max turns reached - force the agent to create a report
        logger.warning("Subagent reached max turns without reporting - forcing report")
        
        # Log the force report attempt if logger is available
        if self.turn_logger:
            turn_data = {
                "task_type": self.task.agent_type,
                "task_title": self.task.title,
                "event": "forcing_report",
                "reason": "max_turns_reached"
            }
            self.turn_logger.log_turn(self.max_turns + 1, turn_data)
        
        # Create a forceful message requiring only a report action
        force_report_msg = (
            "\n\n⚠️ CRITICAL: MAXIMUM TURNS REACHED ⚠️\n"
            "You have reached the maximum number of allowed turns.\n"
            "You MUST now submit a report using ONLY the <report> action.\n"
            "NO OTHER ACTIONS ARE ALLOWED.\n\n"
            "Instructions:\n"
            "1. Use ONLY the <report> action\n"
            "2. Include ALL contexts you have discovered so far\n"
            "3. In the comments section:\n"
            "   - Summarize what you have accomplished\n"
            "   - If the task is incomplete, explain what remains to be done\n"
            "   - Describe what you were about to do next and why\n\n"
            "SUBMIT YOUR REPORT NOW."
        )
        
        # Append the force report message to the last user message (env_response)
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] += force_report_msg
        else:
            # Fallback: add as new message if last message isn't user
            self.messages.append({"role": "user", "content": force_report_msg.strip()})
        
        # Get final response with report
        try:
            llm_response = self._get_llm_response(self.messages)
            self.messages.append({"role": "assistant", "content": llm_response})
            
            # Execute to extract the report
            result = self.executor_stateless.execute(llm_response)
            
            # Check for report action
            report = self._check_for_report(result.actions_executed)
            if report:
                logging.info(f"\n🔵 SUBAGENT FORCED REPORT DETECTED - After {self.max_turns} turns")
                logging.info(f"   Agent Type: {self.task.agent_type}")
                logging.info(f"   Task Title: {self.task.title}")
                logging.info(f"   Comments: {report.comments[:200]}..." if len(report.comments) > 200 else f"   Comments: {report.comments}")
                logging.info(f"   Contexts returned: {len(report.contexts)}")
                
                report.meta.num_turns = self.max_turns + 1
                report.meta.total_input_tokens = self.total_input_tokens
                report.meta.total_output_tokens = self.total_output_tokens
                logger.debug(f"Token usage - Input: {self.total_input_tokens}, Output: {self.total_output_tokens}")
                
                logging.info(f"🔵 SUBAGENT RETURNING FORCED REPORT TO ORCHESTRATOR\n")
                
                # Log final summary if logger is available
                if self.turn_logger:
                    summary_data = {
                        "task_type": self.task.agent_type,
                        "task_title": self.task.title,
                        "completed": True,
                        "num_turns": report.meta.num_turns,
                        "total_input_tokens": report.meta.total_input_tokens,
                        "total_output_tokens": report.meta.total_output_tokens,
                        "contexts_returned": len(report.contexts),
                        "comments": report.comments
                    }
                    self.turn_logger.log_final_summary(summary_data)
                
                return report
        except Exception as e:
            logger.error(f"Error forcing report: {e}")
        
        # Fallback if agent still doesn't provide report
        logging.warning(f"\n🔴 SUBAGENT FALLBACK - No report provided after {self.max_turns} turns")
        logging.warning(f"   Agent Type: {self.task.agent_type}")
        logging.warning(f"   Task Title: {self.task.title}")
        logging.warning(f"   Creating fallback report\n")
        
        return SubagentReport(
            contexts=[],
            comments=f"Task incomplete - reached maximum turns ({self.max_turns}) without proper completion. Agent failed to provide report when requested.",
            meta=SubagentMeta(
                trajectory=self.messages.copy(), 
                num_turns=self.max_turns,
                total_input_tokens=self.total_input_tokens,
                total_output_tokens=self.total_output_tokens
            )
        )