"""Task Manager for coordinating between Orchestrator and subagents."""

import json
import logging
from typing import Dict, List, Optional, Any

from datetime import datetime

from src.agents.actions.entities.context import Context
from src.agents.actions.entities.subagent_report import SubagentReport
from src.agents.actions.entities.task import ContextBootstrapItem, Task, TaskStatus


logger = logging.getLogger(__name__)


class OrchestratorHub:
    """Central coordination hub for Orchestrator, allowing the agent to manage tasks and the context store."""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.context_store: Dict[str, Context] = {}
        self.task_counter = 0
        
    def create_task(
        self,
        agent_type: str,
        title: str,
        description: str,
        context_refs: List[str],
        context_bootstrap: List[dict]
    ) -> str:
        self.task_counter += 1
        task_id = f"task_{self.task_counter:03d}"
        
        # Convert bootstrap dicts to objects
        bootstrap_items = [
            ContextBootstrapItem(path=item['path'], reason=item['reason'])
            for item in context_bootstrap
        ]
        
        task = Task(
            task_id=task_id,
            agent_type=agent_type,
            title=title,
            description=description,
            context_refs=context_refs,
            context_bootstrap=bootstrap_items
        )
        
        self.tasks[task_id] = task
        logger.info(f"Created task {task_id}: {title}")
        
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)
    
    def update_task_status(self, task_id: str, status: TaskStatus) -> bool:
        """Update the status of a task.
        
        Returns:
            True if successful, False if task not found
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.warning(f"Task {task_id} not found")
            return False
        
        task.status = status
        if status == TaskStatus.COMPLETED:
            task.completed_at = datetime.now().isoformat()
            
        logger.info(f"Updated task {task_id} status to {status.value}")
        return True
    
    def view_all_tasks(self) -> str:
        """Return formatted view of all tasks with their statuses."""
        if not self.tasks:
            return "No tasks created yet."
        
        lines = ["Tasks:"]
        for task_id, task in self.tasks.items():
            status_symbol = {
                TaskStatus.CREATED: "○",
                TaskStatus.COMPLETED: "●",
                TaskStatus.FAILED: "✗"
            }.get(task.status, "?")
            
            lines.append(f"  {status_symbol} [{task_id}] {task.title} ({task.agent_type})")
            lines.append(f"      Status: {task.status.value}")
            
            if task.context_refs:
                lines.append(f"      Context refs: {', '.join(task.context_refs)}")

            if task.context_bootstrap:
                bootstrap_str = ', '.join([item.path for item in task.context_bootstrap])
                lines.append(f"      Bootstrap: {bootstrap_str}")

            if task.result:
                lines.append(f"      Result: {json.dumps(task.result)}")
            if task.completed_at:
                lines.append(f"      Completed at: {task.completed_at}")
                
        return "\n".join(lines)
    
    def add_context(
        self,
        context_id: str,
        content: str,
        reported_by: str,
        task_id: Optional[str] = None
    ) -> bool:
        if context_id in self.context_store:
            logger.warning(f"Context {context_id} already exists")
            return False
        
        context = Context(
            id=context_id,
            content=content,
            reported_by=reported_by,
            task_id=task_id
        )
        
        self.context_store[context_id] = context
        logger.info(f"Added context {context_id} to store")
        return True

    def get_contexts_for_task(self, context_refs: List[str]) -> Dict[str, str]:
        """Get multiple contexts by their IDs.
        
        Returns:
            Dictionary mapping context_id to content
        """
        contexts = {}
        for ref in context_refs:
            context = self.context_store.get(ref)
            if context:
                contexts[ref] = context.content
            else:
                logger.warning(f"Context {ref} not found")
                
        return contexts
    
    def view_context_store(self) -> str:
        """Return formatted summary of all stored contexts."""
        if not self.context_store:
            return "Context store is empty."
        
        lines = ["Context Store:"]
        for context_id, context in self.context_store.items():
            # Get first line of content for summary
            lines.append(f"  Id: [{context_id}]")
            lines.append(f"     Content: {context.content}")
            lines.append(f"     Reported by: {context.reported_by}")
            
            if context.task_id:
                lines.append(f"    Task: {context.task_id}")
                
        return "\n".join(lines)
    
    def process_subagent_result(
        self,
        task_id: str,
        report: SubagentReport
    ) -> Dict[str, Any]:
        """Process subagent output and extract contexts.
        
        Args:
            task_id: The task that was executed
            raw_output: Raw output from subagent containing contexts and comments
            
        Returns:
            Processed result for Orchestrator containing context_ids and comments
        """
        # Extract contexts and store them
        stored_context_ids = []
        
        for ctx in report.contexts:
            if ctx.id and ctx.content:
                # Add context to store
                success = self.add_context(
                    context_id=ctx.id,
                    content=ctx.content,
                    reported_by=task_id,
                    task_id=task_id
                )
                
                if success:
                    stored_context_ids.append(ctx.id)
                else:
                    # Context ID already exists, might want to update or skip
                    logger.warning(f"Context {ctx.id} already exists, skipping")
        
        # Prepare result for Orchestrator
        result = {
            'task_id': task_id,
            'context_ids_stored': stored_context_ids,
            'comments': report.comments
        }
        
        # Don't include trajectory in task result as it's already in conversation history
        
        # Update task with result
        task = self.get_task(task_id)
        if task:
            task.result = result
            self.update_task_status(task_id, TaskStatus.COMPLETED)
        
        return result
