from agno.agent import Agent
from agno.models.openai import OpenAIChat
from rich.console import Console

regular_agent = Agent(model=OpenAIChat(id="gpt-4o-mini"), markdown=True)
console = Console()
reasoning_agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    reasoning=True,
    markdown=True,
    structured_outputs=True,
)

task = "单词 “supercalifragilisticexpialidocious” 中有多少个 “r”？"

console.rule("[bold green]Regular Agent[/bold green]")
regular_agent.print_response(task, stream=True)
console.rule("[bold yellow]Reasoning Agent[/bold yellow]")
reasoning_agent.print_response(task, stream=True, show_full_reasoning=True)