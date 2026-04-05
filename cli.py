import os
import uuid
import click
import readline  # Added for interactive history support
from langgraph.checkpoint.sqlite import SqliteSaver
from demo_agent import builder

# --- Configuration ---
DB_PATH = "state.sqlite"

def get_graph(checkpointer):
    """Re-compiles the graph with our persistent checkpointer."""
    return builder.compile(checkpointer=checkpointer, interrupt_before=["action"])

def run_agent_turn(graph, config, message=None):
    """Executes a single turn of the agent, handling interrupts."""
    input_data = {"messages": [("user", message)]} if message else None
    
    # Execution loop
    for event in graph.stream(input_data, config, stream_mode="values"):
        # We can add streaming logic here if needed
        pass

    # Check for interrupts (Human-in-the-loop)
    snapshot = graph.get_state(config)
    while snapshot.next:
        click.echo(f"\n[!] INTERRUPT: Agent is about to execute: {snapshot.next}")
        if click.confirm("Do you want to proceed?"):
            for event in graph.stream(None, config, stream_mode="values"):
                pass
            snapshot = graph.get_state(config) # Check if there's another interrupt
        else:
            click.echo("[*] Action cancelled/paused.")
            break

    # Final output display
    final_state = graph.get_state(config)
    if final_state.values and "messages" in final_state.values:
        last_msg = final_state.values["messages"][-1]
        if last_msg.type == "ai":
            click.echo(f"\nAssistant: {last_msg.content}")

@click.command()
@click.argument('message', required=False)
@click.option('--resume', 'session_id', help='Resume a session with the given ID.')
@click.option('--list-sessions', is_flag=True, help='List existing session IDs.')
def main(message, session_id, list_sessions):
    """CLI Wrapper for the Task Management Agent."""
    
    with SqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        graph = get_graph(checkpointer)
        
        if list_sessions:
            click.echo("Existing sessions (thread_ids) in state.sqlite:")
            import sqlite3
            try:
                with sqlite3.connect(DB_PATH) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
                    threads = cursor.fetchall()
                    if not threads:
                        click.echo(" (No sessions found)")
                    for t in threads:
                        click.echo(f" - {t[0]}")
            except sqlite3.OperationalError:
                click.echo(" (No sessions found - database not yet initialized)")
            return

        # Handle Session ID logic
        thread_id = session_id if session_id else str(uuid.uuid4())
        click.echo(f"[*] Session ID: {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}

        if message:
            # Single-shot mode
            run_agent_turn(graph, config, message)
        else:
            # Interactive REPL mode
            click.echo("[*] Entering interactive mode. Type 'exit' or 'quit' to stop.")
            while True:
                try:
                    user_input = input(f"({thread_id[:8]}) > ").strip()
                    if user_input.lower() in ["exit", "quit"]:
                        break
                    if not user_input:
                        continue
                    run_agent_turn(graph, config, user_input)
                except EOFError:
                    break
                except KeyboardInterrupt:
                    click.echo("\nInterrupted by user.")
                    break

if __name__ == "__main__":
    main()
