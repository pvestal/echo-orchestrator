"""Action handler for executing parsed actions in the RLLM environment."""

import logging
from typing import Dict, Callable, Tuple, Optional, Any

from src.agents.env_interaction.command_executor import CommandExecutor
from src.agents.actions.orchestrator_hub import OrchestratorHub
from src.agents.actions.entities.actions import (
    Action,
    BatchTodoAction,
    AddNoteAction,
    ViewAllNotesAction,
    ReadAction,
    WriteAction,
    EditAction,
    MultiEditAction,
    GrepAction,
    GlobAction,
    LSAction,
    FileMetadataAction,
    WriteTempScriptAction,
    BashAction,
    FinishAction,
    TaskCreateAction,
    AddContextAction,
    LaunchSubagentAction,
    ReportAction,
)
from src.agents.actions.state_managers import TodoManager, ScratchpadManager
from src.agents.actions.file_manager import FileManager
from src.agents.actions.search_manager import SearchManager

logger = logging.getLogger(__name__)


def format_tool_output(tool_name: str, content: str) -> str:
    """Format tool output in XML format.
    
    Args:
        tool_name: Name of the tool (e.g., 'todo', 'file', 'search')
        content: The raw content to wrap
        
    Returns:
        XML-formatted output string
    """
    tag_name = f"{tool_name}_output"
    return f"<{tag_name}>\n{content}\n</{tag_name}>"


class ActionHandler:
    """Handles execution of different action types."""
    
    @staticmethod
    def truncate_content(content: str, max_length: int = 15) -> str:
        """Truncate content for display to reduce tokens."""
        return content[:max_length] + "..." if len(content) > max_length else content
    
    def __init__(
        self,
        executor: CommandExecutor,
        todo_manager: Optional[TodoManager] = None,
        scratchpad_manager: Optional[ScratchpadManager] = None,
        orchestrator_hub: Optional[OrchestratorHub] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        logging_dir: Optional[Any] = None,
    ):
        self.executor = executor
        self.todo_manager = todo_manager or TodoManager()
        self.scratchpad_manager = scratchpad_manager or ScratchpadManager()
        self.file_manager = FileManager(executor)
        self.search_manager = SearchManager(executor)
        self.orchestrator_hub = orchestrator_hub or OrchestratorHub()
        
        # Store LLM configuration for subagents
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.api_base = api_base
        self.logging_dir = logging_dir
        
        # Track subagent trajectories for current execution
        self.subagent_trajectories: Dict[str, Dict[str, Any]] = {}
        
        # Map action types to handler methods
        self._handlers: Dict[type, Callable] = {
            BatchTodoAction: self._handle_batch_todo,
            AddNoteAction: self._handle_add_note,
            ViewAllNotesAction: self._handle_view_all_notes,
            ReadAction: self._handle_read_file,
            WriteAction: self._handle_write_file,
            EditAction: self._handle_edit_file,
            MultiEditAction: self._handle_multi_edit_file,
            GrepAction: self._handle_grep,
            GlobAction: self._handle_glob,
            LSAction: self._handle_ls,
            FileMetadataAction: self._handle_file_metadata,
            WriteTempScriptAction: self._handle_write_temp_script,
            BashAction: self._handle_bash,
            FinishAction: self._handle_finish,
            TaskCreateAction: self._handle_task_create,
            AddContextAction: self._handle_add_context,
            LaunchSubagentAction: self._handle_launch_subagent,
            ReportAction: self._handle_report,
        }
    
    def handle_action(self, action: Action) -> Tuple[str, bool]:
        """Handle an action and return (response, is_error)."""
        handler = self._handlers.get(type(action))
        if handler:
            return handler(action)
        content = f"[ERROR] Unknown action type: {type(action).__name__}"
        return format_tool_output("unknown", content), True
    
    # Individual todo action handlers removed - only BatchTodoAction is used now
    
    
    
    
    def _handle_batch_todo(self, action: BatchTodoAction) -> Tuple[str, bool]:
        """Handle batch todo operations."""
        results = []
        has_error = False
        
        for op in action.operations:
            if op.action == "add":
                task_id = self.todo_manager.add_task(op.content)
                truncated_content = self.truncate_content(op.content)
                results.append(f"Added todo [{task_id}]: {truncated_content}")
            
            elif op.action == "complete":
                task = self.todo_manager.get_task(op.task_id)
                if not task:
                    results.append(f"[ERROR] Task {op.task_id} not found")
                    has_error = True
                elif task["status"] == "completed":
                    results.append(f"Task {op.task_id} is already completed")
                else:
                    self.todo_manager.complete_task(op.task_id)
                    truncated_content = self.truncate_content(task['content'])
                    results.append(f"Completed task [{op.task_id}]: {truncated_content}")
            
            elif op.action == "delete":
                task = self.todo_manager.get_task(op.task_id)
                if not task:
                    results.append(f"[ERROR] Task {op.task_id} not found")
                    has_error = True
                else:
                    self.todo_manager.delete_task(op.task_id)
                    truncated_content = self.truncate_content(task['content'])
                    results.append(f"Deleted task [{op.task_id}]: {truncated_content}")
            
            elif op.action == "view_all":
                # This is handled after all operations
                pass
        
        # Join results
        response = "\n".join(results)
        
        # Add todo list if requested
        if action.view_all:
            response += f"\n\n{self.todo_manager.view_all()}"
        
        return format_tool_output("todo", response), has_error
    
    def _handle_add_note(self, action: AddNoteAction) -> Tuple[str, bool]:
        """Handle adding a note to scratchpad."""
        if not action.content:
            return format_tool_output("scratchpad", "[ERROR] Cannot add empty note"), True
        
        note_idx = self.scratchpad_manager.add_note(action.content)
        response = f"Added note {note_idx + 1} to scratchpad"
        return format_tool_output("scratchpad", response), False
    
    def _handle_view_all_notes(self, action: ViewAllNotesAction) -> Tuple[str, bool]:
        """Handle viewing all notes."""
        return format_tool_output("scratchpad", self.scratchpad_manager.view_all()), False
    
    def _handle_read_file(self, action: ReadAction) -> Tuple[str, bool]:
        """Handle reading a file."""
        content, is_error = self.file_manager.read_file(
            action.file_path, action.offset, action.limit
        )
        return format_tool_output("file", content), is_error
    
    def _handle_write_file(self, action: WriteAction) -> Tuple[str, bool]:
        """Handle writing a file."""
        content, is_error = self.file_manager.write_file(
            action.file_path, action.content
        )
        return format_tool_output("file", content), is_error
    
    def _handle_edit_file(self, action: EditAction) -> Tuple[str, bool]:
        """Handle editing a file."""
        content, is_error = self.file_manager.edit_file(
            action.file_path, action.old_string, action.new_string, action.replace_all
        )
        return format_tool_output("file", content), is_error
    
    def _handle_multi_edit_file(self, action: MultiEditAction) -> Tuple[str, bool]:
        """Handle multiple edits to a file."""
        edits = [(e.old_string, e.new_string, e.replace_all) for e in action.edits]
        content, is_error = self.file_manager.multi_edit_file(
            action.file_path, edits
        )
        return format_tool_output("file", content), is_error
    
    def _handle_grep(self, action: GrepAction) -> Tuple[str, bool]:
        """Handle grep search."""
        content, is_error = self.search_manager.grep(
            action.pattern, action.path, action.include
        )
        return format_tool_output("search", content), is_error
    
    def _handle_glob(self, action: GlobAction) -> Tuple[str, bool]:
        """Handle glob search."""
        content, is_error = self.search_manager.glob(
            action.pattern, action.path
        )
        return format_tool_output("search", content), is_error
    
    def _handle_ls(self, action: LSAction) -> Tuple[str, bool]:
        """Handle ls command."""
        content, is_error = self.search_manager.ls(
            action.path, action.ignore
        )
        return format_tool_output("search", content), is_error
    
    def _handle_file_metadata(self, action: FileMetadataAction) -> Tuple[str, bool]:
        """Handle file metadata request."""
        content, is_error = self.file_manager.get_metadata(action.file_paths)
        return format_tool_output("file", content), is_error
    
    def _handle_write_temp_script(self, action: WriteTempScriptAction) -> Tuple[str, bool]:
        """Handle writing a temporary script file.
        
        This uses the same underlying file write functionality but is specifically
        intended for temporary scripts used during exploration/testing.
        """
        # Use the existing file write functionality
        content, is_error = self.file_manager.write_file(
            action.file_path, action.content
        )
        return format_tool_output("file", content), is_error
    
    def _handle_bash(self, action: BashAction) -> Tuple[str, bool]:
        """Handle bash command execution."""
        try:
            if action.block:
                output, exit_code = self.executor.execute(
                    action.cmd, 
                    timeout=action.timeout_secs
                )
            else:
                # Non-blocking execution
                self.executor.execute_background(action.cmd)
                output = "Command started in background"
                exit_code = 0
            
            is_error = exit_code != 0
            return format_tool_output("bash", output), is_error
            
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            return format_tool_output("bash", error_msg), True
    
    def _handle_finish(self, action: FinishAction) -> Tuple[str, bool]:
        """Handle finish action."""
        response = f"Task marked as complete: {action.message}"
        return format_tool_output("finish", response), False
    
    def _handle_task_create(self, action: TaskCreateAction) -> Tuple[str, bool]:
        """Handle task creation."""
        try:
            task_id = self.orchestrator_hub.create_task(
                agent_type=action.agent_type,
                title=action.title,
                description=action.description,
                context_refs=action.context_refs,
                context_bootstrap=action.context_bootstrap
            )
            
            response = f"Created task {task_id}: {action.title}"
            
            # Auto-launch if requested
            if action.auto_launch:
                launch_action = LaunchSubagentAction(task_id=task_id)
                launch_response, launch_error = self._handle_launch_subagent(launch_action)
                response += f"\n{launch_response}"
                return format_tool_output("task", response), launch_error
            
            return format_tool_output("task", response), False
            
        except Exception as e:
            error_msg = f"[ERROR] Failed to create task: {str(e)}"
            return format_tool_output("task", error_msg), True
    
    def _handle_add_context(self, action: AddContextAction) -> Tuple[str, bool]:
        """Handle adding context to store."""
        try:
            success = self.orchestrator_hub.add_context(
                context_id=action.id,
                content=action.content,
                reported_by=action.reported_by,
                task_id=action.task_id
            )
            
            if success:
                response = f"Added context '{action.id}' to store"
            else:
                response = f"[WARNING] Context '{action.id}' already exists in store"
                
            return format_tool_output("context", response), not success
            
        except Exception as e:
            error_msg = f"[ERROR] Failed to add context: {str(e)}"
            return format_tool_output("context", error_msg), True
    
    def _handle_launch_subagent(self, action: LaunchSubagentAction) -> Tuple[str, bool]:
        """Handle launching a subagent for a task."""
        # Import here to avoid circular import
        from src.agents.subagent import Subagent, SubagentTask
        
        task = self.orchestrator_hub.get_task(action.task_id)
        if not task:
            error_msg = f"[ERROR] Task {action.task_id} not found"
            return format_tool_output("subagent", error_msg), True
        
        # Resolve context references
        context_store_ctxts = self.orchestrator_hub.get_contexts_for_task(task.context_refs)
        bootstrap_ctxts = []
        
        if task.context_bootstrap: 
            for item in task.context_bootstrap:
                path = item.path
                reason = item.reason
                is_dir = path.endswith("/")
                if is_dir:
                    ls_result, _ = self.search_manager.ls(path, ignore=[])
                    bootstrap_ctxts.append({"path": path, "content": ls_result, "reason": reason})
                else:
                    file_result, _ = self.file_manager.read_file(path, offset=0, limit=1000)
                    bootstrap_ctxts.append({"path": path, "content": file_result, "reason": reason})


        subagent_task = SubagentTask(
            agent_type=task.agent_type,
            title=task.title,
            description=task.description,
            ctx_store_ctxts=context_store_ctxts,
            bootstrap_ctxts=bootstrap_ctxts
        )

        subagent = Subagent(
            task=subagent_task,
            executor=self.executor,
            model=self.model,
            temperature=self.temperature,
            api_key=self.api_key,
            api_base=self.api_base,
            logging_dir=self.logging_dir,
            task_id=action.task_id
        )
        
        logger.info(f"Launching {task.agent_type} subagent for task: {task.title}")
        report = subagent.run()
        
        # Store trajectory and token counts for this turn if available
        if report.meta:
            self.subagent_trajectories[action.task_id] = {
                'agent_type': task.agent_type,
                'title': task.title,
                'trajectory': report.meta.trajectory if report.meta.trajectory else None,
                'total_input_tokens': report.meta.total_input_tokens,
                'total_output_tokens': report.meta.total_output_tokens
            }

        result = self.orchestrator_hub.process_subagent_result(
            action.task_id,
            report
        )

        # Format response for Orchestrator
        response_lines = [
            f"Subagent completed task {action.task_id}",
            f"Contexts stored: {', '.join(result['context_ids_stored'])}",
        ]
        
        if result['comments']:
            response_lines.append(f"Comments: {result['comments']}")
        
        response = "\n".join(response_lines)
        return format_tool_output("subagent", response), False
    
    def _handle_report(self, action: ReportAction) -> Tuple[str, bool]:
        return format_tool_output("report", "Report submission successful"), False
    
    def get_and_clear_subagent_trajectories(self) -> Dict[str, Dict[str, Any]]:
        """Get collected subagent trajectories and clear the internal store."""
        trajectories = self.subagent_trajectories.copy()
        self.subagent_trajectories.clear()
        return trajectories
        