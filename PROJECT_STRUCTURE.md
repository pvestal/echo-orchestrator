# Project Structure Documentation

## Directory Structure

```
multi-agent-coding-system/
├── src/
│   ├── agents/
│   │   ├── orchestrator_agent.py       # Main orchestrator agent coordinating tasks
│   │   ├── subagent.py                 # Subagent implementation for delegated tasks
│   │   ├── actions/                    # Action handling and parsing
│   │   │   ├── entities/               # Data models for actions and context
│   │   │   │   ├── actions.py          # Pydantic action definitions
│   │   │   │   ├── context.py          # Context data model
│   │   │   │   ├── subagent_report.py  # Subagent report structure
│   │   │   │   └── task.py             # Task management models
│   │   │   ├── parsing/                # Action parsing logic
│   │   │   │   ├── parser.py           # XML-to-action parser
│   │   │   │   └── action_handler.py   # Action execution handler
│   │   │   ├── orchestrator_hub.py     # Central coordination hub
│   │   │   ├── file_manager.py         # File operations management
│   │   │   ├── search_manager.py       # Search operations (grep, glob, ls)
│   │   │   └── state_managers.py       # Todo and scratchpad state management
│   │   ├── env_interaction/            # Environment interaction layer
│   │   │   ├── command_executor.py     # Docker/command execution
│   │   │   ├── turn_executor.py        # Single-turn execution logic
│   │   │   └── entities/               # Execution-related models
│   │   │       ├── conversation_history.py
│   │   │       ├── execution_result.py
│   │   │       └── turn.py
│   │   ├── state/                      # State management
│   │   │   └── orchestrator_state.py   # Orchestrator state tracking
│   │   ├── system_msgs/                # System messages for agents
│   │   │   ├── system_msg_loader.py    # Message loading utilities
│   │   │   └── md_files/               # Markdown system messages
│   │   └── utils/                      # Utilities
│   │       └── llm_client.py           # LLM API client wrapper
│   └── misc/                           # Miscellaneous utilities
│       ├── log_setup.py                # Logging configuration
│       └── turn_logger.py              # Turn-based logging
├── tests/                              # Test suite
│   ├── test_action_parser.py          # Parser unit tests
│   ├── test_lead_architect_real.py    # Integration tests
│   └── test_subagent_real.py          # Subagent tests
├── pyproject.toml                      # Project dependencies and metadata
└── README.md                           # Project overview
```

## Core Components

### 1. **OrchestratorAgent** (`src/agents/orchestrator_agent.py`)
The main coordinator agent that manages the entire task execution workflow.

**Key Responsibilities:**
- Receives high-level task instructions
- Manages conversation history and state
- Creates and delegates tasks to subagents
- Processes subagent reports and maintains context store
- Tracks token usage across all agents

**Key Methods:**
- `perform_task()`: Entry point for terminal bench integration
- `execute_turn()`: Single turn execution with LLM interaction
- `run()`: Main execution loop until task completion

### 2. **Subagent** (`src/agents/subagent.py`)
Specialized agents (Explorer or Coder) that execute specific delegated tasks.

**Types:**
- **Explorer Agent**: Investigates codebase, searches for information
- **Coder Agent**: Implements features, fixes bugs, writes code

**Key Features:**
- Autonomous execution within defined scope
- Context collection and reporting back to orchestrator
- Forced reporting mechanism when max turns reached
- Independent state management

### 3. **OrchestratorHub** (`src/agents/actions/orchestrator_hub.py`)
Central coordination hub managing tasks and context sharing.

**Core Functions:**
- **Task Management**: Create, track, and update task status
- **Context Store**: Centralized storage for discovered information
- **Subagent Results Processing**: Extract and store contexts from reports

**Data Flow:**
1. Orchestrator creates tasks with context references
2. Subagents execute tasks and return reports with contexts
3. Hub stores contexts for future task reference
4. Contexts are shared across subagents via the hub

### 4. **Action System** (`src/agents/actions/`)
Comprehensive action framework for agent-environment interaction.

**Action Categories:**
- **File Operations**: Read, Write, Edit, MultiEdit
- **Search Operations**: Grep, Glob, LS
- **Task Management**: TaskCreate, LaunchSubagent, Report
- **State Management**: Todo operations, Scratchpad notes
- **Environment**: Bash command execution
- **Control**: Finish action for task completion

**Parser Pipeline:**
1. XML tag extraction from LLM response
2. YAML content parsing within tags
3. Pydantic validation and action object creation
4. Action execution via handlers

### 5. **Execution Layer** (`src/agents/env_interaction/`)
Handles the actual execution of actions and environment interaction.

**Components:**
- **TurnExecutor**: Stateless single-turn execution
- **CommandExecutor**: Docker container command execution
- **ExecutionResult**: Structured results with error handling

## Key Architecture Patterns

### 1. **Multi-Agent Orchestration Pattern**
The system employs a hierarchical multi-agent architecture where:
- **Orchestrator** acts as the task manager and coordinator
- **Subagents** are specialized workers for specific task types
- Communication happens through structured reports and context sharing

### 2. **Stateless Turn-Based Execution**
Each agent operates in discrete turns:
- LLM generates actions based on current state
- Actions are parsed and executed
- Environment responses update the state
- Process repeats until task completion or max turns

### 3. **Context Store Pattern**
Centralized information management:
- Subagents discover and report contexts
- Orchestrator Hub stores contexts with unique IDs
- Future tasks can reference stored contexts
- Enables knowledge sharing across agent boundaries

### 4. **Action-First Design**
All agent capabilities are expressed as discrete actions:
- Declarative action definitions using Pydantic models
- XML-based action syntax in LLM responses
- Strict validation before execution
- Clear separation between parsing and execution

### 5. **Forced Completion Pattern**
Ensures task termination:
- Max turn limits for both orchestrator and subagents
- Forced reporting mechanism when limits reached
- Fallback report generation if agent fails to comply
- Prevents infinite loops and resource exhaustion

### 6. **Token Tracking Architecture**
Comprehensive token usage monitoring:
- Each agent tracks its own message history
- Token counting at orchestrator and subagent levels
- Aggregated reporting for total system usage
- Supports both input and output token metrics

### 7. **Error Resilience Pattern**
Multiple layers of error handling:
- Parsing errors captured and reported to agent
- Execution errors handled gracefully
- Fallback mechanisms for critical operations
- Continuation strategies to prevent task failure

### 8. **Dependency Injection Pattern**
Components are loosely coupled:
- CommandExecutor injected into agents
- Shared executor across orchestrator and subagents
- Configurable LLM clients with override parameters
- Modular action handlers and parsers

### 9. **System Message Versioning**
Flexible system message management:
- Versioned system messages for each agent type
- Dynamic loading based on agent requirements
- Separation of prompting logic from code
- Easy updates without code changes

### 10. **Structured Logging Pattern**
Comprehensive logging and observability:
- Turn-based logging for debugging
- Separate log files for orchestrator and subagents
- JSON-formatted conversation logs
- Timestamped markers for performance analysis