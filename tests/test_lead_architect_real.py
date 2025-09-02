#!/usr/bin/env python3
"""Test script for LeadArchitectAgent with real LiteLLM calls."""

import os
import sys
import subprocess
import time
import uuid

from src.agents.env_interaction.command_executor import DockerExecutor
from src.agents.orchestrator_agent import OrchestratorAgent

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))



def test_lead_architect_simple_task(model: str, temperature=0.1):
    """Test orchestrator with a simple task."""
    
    print(f"\n{'='*60}")
    print(f"Testing with model: {model}")
    print(f"Temperature: {temperature}")
    print('='*60)
    
    # Generate unique container name
    container_name = f"test_orchestrator_{uuid.uuid4().hex[:8]}"
    
    try:
        # Start Docker container
        print(f"Starting Docker container: {container_name}")
        subprocess.run([
            'docker', 'run', '-d', '--rm',
            '--name', container_name,
            'ubuntu:latest',
            'sleep', 'infinity'
        ], check=True, capture_output=True)
        
        # Wait for container to be ready
        time.sleep(2)
        
        # Create command executor with Docker
        executor = DockerExecutor(container_name)
        
        # Create orchestrator with specified model
        orchestrator = OrchestratorAgent(
            model=model,
            temperature=temperature,
        )
        
        # Setup the orchestrator with the executor
        orchestrator.setup(executor)
        
        # Define the task
        instruction = (
            "Create a file called 'hello.txt' with content 'Hello world! Ending with a newline.' "
        )
        
        # Run the task
        print("\nExecuting task...")
        result = orchestrator.run(instruction, max_turns=15)
        
        # Display results
        print(f"\n{'='*40}")
        print("EXECUTION RESULT:")
        print(f"{'='*40}")
        print(f"Completed: {result['completed']}")
        print(f"Finish message: {result['finish_message']}")
        print(f"Turns executed: {result['turns_executed']}")
        print(f"Max turns reached: {result['max_turns_reached']}")
        
        # Check if files were created in container
        print(f"\n{'='*40}")
        print("VERIFICATION:")
        print(f"{'='*40}")
        
        # Check hello.txt
        output, exit_code = executor.execute("cat hello.txt")
        if exit_code == 0:
            print(f"✓ hello.txt created successfully!")
            print(f"  Content: {repr(output)}")
            if output.endswith('\n'):
                print("  Content ends with a newline.")
            else:
                print("  Content does NOT end with a newline.")
        else:
            print("✗ hello.txt was not created")
        
        # Show final state summary
        if orchestrator.state:
            print(f"\n{'='*40}")
            print("FINAL STATE:")
            print(f"{orchestrator.state.to_dict()}")
        
        return result
        
    finally:
        # Clean up Docker container
        try:
            print(f"\nStopping and removing container: {container_name}")
            subprocess.run(['docker', 'stop', container_name], 
                         capture_output=True, timeout=10)
        except Exception as e:
            print(f"Error cleaning up container: {e}")

def main():
    """Main test runner."""
    
    # Test with different models if you want
    models_to_test = [
        ("anthropic/claude-sonnet-4-20250514", 0.1),
        # ("openrouter/qwen/qwen3-coder", 0.1),
        # ("openrouter/z-ai/glm-4.5", 0.1),
        # ("openrouter/deepseek/deepseek-chat-v3.1", 0.1),
    ]
    
    # Check for environment variables
    if not os.getenv("LITE_LLM_API_KEY") and not os.getenv("LITELLM_API_KEY"):
        print("Warning: No API key found in LITE_LLM_API_KEY or LITELLM_API_KEY")
        print("You may need to set one of these environment variables")
    
    print("\n" + "="*60)
    print("TESTING LEAD ARCHITECT AGENT")
    print("="*60)
    
    # Test 1: Simple task
    print("\n### TEST 1: Simple File Creation ###")
    results = []
    for model, temp in models_to_test:
        try:
            result = test_lead_architect_simple_task(model=model, temperature=temp)
            results.append((model, "Test 1", "SUCCESS", result))
        except Exception as e:
            print(f"\n✗ Error with model {model}: {e}")
            results.append((model, "Test 1", "FAILED", str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    for model, test_name, status, _ in results:
        print(f"{model} - {test_name}: {status}")


if __name__ == "__main__":
    main()