import os
import uuid
import click
from langgraph.checkpoint.sqlite import SqliteSaver
from demo_agent import builder  # Importing builder for decoupling

# --- Configuration ---
DB_PATH = "state.sqlite"

def get_graph(checkpointer):
    """
    Re-compiles the graph with our persistent checkpointer.
    This allows the CLI to control persistence independently of the agent definition.
    """
    # We use the same interrupt logic as the original demo_agent
    return builder.compile(checkpointer=checkpointer, interrupt_before=["action"])

@click.command()
@click.argument('message', required=False)
@click.option('--resume', 'session_id', help='Resume a session with the given ID.')
@click.option('--list-sessions', is_flag=True, help='List existing session IDs.')
def main(message, session_id, list_sessions):
    """CLI Wrapper for the Task Management Agent."""
    
    # Initialize SQLite checkpointer
    # Using a context manager for the checkpointer connection
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
                # Table doesn't exist yet
                click.echo(" (No sessions found - database not yet initialized)")
            return

        if not message and not session_id:
            click.echo("Error: Please provide a message or use --resume <id>.")
            return

        # Handle Session ID logic
        is_new = False
        if session_id:
            thread_id = session_id
            click.echo(f"[*] Resuming session: {thread_id}")
        else:
            thread_id = str(uuid.uuid4())
            is_new = True
            click.echo(f"[*] Starting new session: {thread_id}")

        config = {"configurable": {"thread_id": thread_id}}

        # If resuming without a message, we might be triggering an interrupt
        input_data = None
        if message:
            input_data = {"messages": [("user", message)]}
        
        # Execution loop to handle interrupts (Human-in-the-loop)
        # We use stream to see what's happening
        for event in graph.stream(input_data, config, stream_mode="values"):
            if "messages" in event:
                last_msg = event["messages"][-1]
                # Only print assistant/tool messages, or everything if it's the final output
                if hasattr(last_msg, "content") and last_msg.content:
                    # We avoid re-printing the user message we just sent if it's the first step
                    if not (is_new and last_msg == event["messages"][0] and message):
                        pass 

        # Check for interrupts
        snapshot = graph.get_state(config)
        if snapshot.next:
            click.echo("\n[!] INTERRUPT: The agent is about to execute a database modification.")
            click.echo(f"Pending actions: {snapshot.next}")
            if click.confirm("Do you want to proceed?"):
                # Proceed by passing None as input to the next node
                for event in graph.stream(None, config, stream_mode="values"):
                    pass
                click.echo("[*] Actions executed.")
            else:
                click.echo(f"[*] Action paused. You can resume this session later with: --resume {thread_id}")
                return

        # Final output display
        final_state = graph.get_state(config)
        if final_state.values and "messages" in final_state.values:
            last_msg = final_state.values["messages"][-1]
            if last_msg.type == "ai":
                click.echo(f"\nAssistant: {last_msg.content}")

if __name__ == "__main__":
    main()
