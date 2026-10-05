"""
Command dispatcher and handler for CLI modes.
"""

from dataclasses import dataclass
import shlex
from typing import Callable, Dict, List, Optional, Tuple
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.config.env_manager import EnvManager
from app.core.orchestrator import BackendOrchestrator
from app.providers.manager import ProviderManager
from app.routing.system_scanner import SystemScanner

console = Console()


@dataclass
class CommandHelp:
    name: str
    args: str
    description: str
    scope: str  # "all", "menu", "chat", "model_select"


class CommandHandler:
    """Processes slash commands dynamically based on current CLI scope."""

    COMMANDS: List[CommandHelp] = [
        CommandHelp("/help", "", "Show available commands for current mode and debug guidance.", "all"),
        CommandHelp("/list", "", "List providers (in menu) or available models (in chat/selection).", "all"),
        CommandHelp("/list-providers", "", "List all configured inference providers and their status.", "menu"),
        CommandHelp("/add-api", "<provider> <key>", "Add or update an API key in .env (e.g. /add-api openai sk-...)", "all"),
        CommandHelp("/edit-api", "provider=<tag> key=<key>", "Edit an API key (e.g. /edit-api provider=groq key=gsk_...)", "all"),
        CommandHelp("/set-endpoint", "<provider> <url>", "Set base URL for a provider (e.g. /set-endpoint ollama http://localhost:11434)", "all"),
        CommandHelp("/specs", "", "Scan and display system hardware specs (CPU, RAM, GPU/VRAM).", "all"),
        CommandHelp("/debug", "", "Inspect internal state, token counts, routing rules, and recent logs.", "all"),
        CommandHelp("/route", "", "View or adjust smart model routing tier assignments.", "chat"),
        CommandHelp("/model", "<model_name>", "Override active model or switch to a specific model.", "chat"),
        CommandHelp("/compact", "", "Manually force context compaction to free memory.", "chat"),
        CommandHelp("/clear", "", "Clear conversation history for the current session.", "chat"),
        CommandHelp("/back", "", "Go back to the previous menu / provider selection.", "chat"),
        CommandHelp("/exit", "", "Exit the application (/bye, /close, /quit).", "all"),
    ]

    def __init__(
        self,
        env_manager: EnvManager,
        provider_manager: ProviderManager,
        orchestrator: BackendOrchestrator
    ):
        self.env_manager = env_manager
        self.provider_manager = provider_manager
        self.orchestrator = orchestrator

    def is_command(self, text: str) -> bool:
        return text.strip().startswith("/")

    def show_help(self, scope: str = "all") -> None:
        """Renders an elegant help table filtered by current scope."""
        table = Table(title=f"Available Commands ({scope.upper()} Mode)", show_header=True, header_style="bold cyan")
        table.add_column("Command", style="bold yellow")
        table.add_column("Arguments", style="green")
        table.add_column("Description", style="white")

        for cmd in self.COMMANDS:
            if cmd.scope == "all" or cmd.scope == scope or scope == "all":
                table.add_row(cmd.name, cmd.args, cmd.description)

        console.print(table)
        console.print("[dim]Tip: You can use commands at any point to debug or reconfigure the backend.[/dim]\n")

    def show_specs(self) -> None:
        """Displays scanned hardware specifications."""
        specs = SystemScanner.scan()
        table = Table(title="Hardware & System Specifications", show_header=False)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="white")

        table.add_row("CPU", f"{specs.cpu_model} ({specs.cpu_cores_physical} physical / {specs.cpu_cores_logical} logical cores)")
        table.add_row("System RAM", f"{specs.ram_available_gb:.1f} GB free / {specs.ram_total_gb:.1f} GB total")
        table.add_row("GPU", specs.gpu_name or "No Dedicated NVIDIA GPU detected (using CPU/Integrated)")
        if specs.vram_total_gb:
            table.add_row("VRAM", f"{specs.vram_free_gb:.1f} GB free / {specs.vram_total_gb:.1f} GB total")
        table.add_row("Recommended Max Local Model", f"[bold green]{specs.recommended_max_param_size}[/bold green]")

        console.print(table)
        console.print()

    def show_debug(self) -> None:
        """Displays internal debugging state."""
        st = self.orchestrator.session_state
        cm = self.orchestrator.context_manager
        table = Table(title="Backend Internal Debug Diagnostics", show_header=False)
        table.add_column("Property", style="bold cyan")
        table.add_column("State", style="white")

        table.add_row("Session Title", st.title)
        table.add_row("Active Provider", st.active_provider_id)
        table.add_row("Routing Mode", f"Override: {st.override_model}" if st.override_model else "Smart Routing Active")
        if self.orchestrator.tier_assignment:
            ta = self.orchestrator.tier_assignment
            table.add_row("Tier - Basic", ta.basic_model)
            table.add_row("Tier - Complex", ta.complex_model)
            table.add_row("Tier - Vision", ta.vision_model or "None (OCR Fallback Active)")
            table.add_row("Tier - Default", ta.default_model)

        table.add_row("Conversation Turns", str(st.turn_count))
        table.add_row("Messages in Context", str(len(cm.messages)))
        table.add_row("Estimated Active Tokens", str(cm.get_total_tokens()))
        table.add_row("Has Compacted Summary", "Yes" if cm.compacted_summary else "No")

        console.print(table)
        console.print()

    async def handle_command(self, cmd_text: str, current_scope: str = "menu") -> Tuple[bool, Optional[str]]:
        """
        Executes a command.
        Returns (handled: bool, action_signal: Optional[str]).
        action_signals: "exit", "back", "clear", None
        """
        parts = cmd_text.strip().split()
        if not parts:
            return True, None

        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ["/exit", "/bye", "/close", "/quit"]:
            return True, "exit"

        if cmd == "/back":
            return True, "back"

        if cmd == "/help":
            self.show_help(current_scope)
            return True, None

        if cmd == "/specs" or cmd == "/scan":
            self.show_specs()
            return True, None

        if cmd == "/debug":
            self.show_debug()
            return True, None

        if cmd in ["/add-api", "/edit-api"]:
            if not args:
                console.print("[red]Usage: /add-api <provider> <key> or /edit-api provider=<provider> key=<key>[/red]")
                return True, None

            provider_arg = ""
            key_arg = ""

            if "=" in args[0]:
                for a in args:
                    if a.startswith("provider="):
                        provider_arg = a.split("=", 1)[1]
                    elif a.startswith("key="):
                        key_arg = a.split("=", 1)[1]
            else:
                provider_arg = args[0]
                key_arg = args[1] if len(args) > 1 else ""

            if not provider_arg or not key_arg:
                console.print("[red]Missing provider or key argument![/red]")
                return True, None

            success, msg = self.env_manager.update_key(provider_arg, key_arg)
            if success:
                console.print(f"[green]{msg}[/green]")
            else:
                console.print(f"[red]{msg}[/red]")
            return True, None

        if cmd == "/set-endpoint":
            if len(args) < 2:
                console.print("[red]Usage: /set-endpoint <provider> <url>[/red]")
                return True, None
            provider_arg, url_arg = args[0], args[1]
            success, msg = self.env_manager.update_key(provider_arg, url_arg)
            console.print(f"[green]{msg}[/green]")
            return True, None

        if cmd in ["/list", "/list-providers", "/providers"]:
            if current_scope == "menu":
                reports = await self.provider_manager.scan_and_test_all()
                table = Table(title="Configured Providers Status", show_header=True, header_style="bold cyan")
                table.add_column("Provider", style="yellow")
                table.add_column("Type", style="cyan")
                table.add_column("Status", style="white")
                table.add_column("Details", style="dim")

                for r in reports:
                    status_str = "[green]Connected[/green]" if r.connected else ("[red]Failed[/red]" if r.configured else "[dim]Not Configured[/dim]")
                    table.add_row(r.name, "Local" if r.is_local else "Cloud", status_str, r.status_message)
                console.print(table)
            elif current_scope in ["chat", "model_select"]:
                if self.orchestrator.tier_assignment:
                    ta = self.orchestrator.tier_assignment
                    provider = self.provider_manager.get_provider(self.orchestrator.session_state.active_provider_id)
                    models = await provider.list_models() if provider else []

                    table = Table(title=f"Models for Provider '{self.orchestrator.session_state.active_provider_id}'", show_header=True)
                    table.add_column("Model Name", style="bold yellow")
                    table.add_column("Parameters / Size", style="cyan")
                    table.add_column("Smart Routing Tier", style="bold green")
                    table.add_column("Capabilities", style="white")

                    for m in models:
                        tiers = []
                        if m.id == ta.basic_model:
                            tiers.append("[Basic]")
                        if m.id == ta.complex_model:
                            tiers.append("[Complex]")
                        if ta.vision_model and m.id == ta.vision_model:
                            tiers.append("[Vision]")
                        if m.id == ta.default_model:
                            tiers.append("[Default]")
                        tier_tag = " ".join(tiers) if tiers else "[dim]-[/dim]"

                        table.add_row(m.name, m.parameter_size, tier_tag, m.display_tag)
                    console.print(table)
            return True, None

        if cmd == "/route":
            if self.orchestrator.tier_assignment:
                ta = self.orchestrator.tier_assignment
                table = Table(title="Current Smart Routing Configuration", show_header=True)
                table.add_column("Tier", style="bold cyan")
                table.add_column("Assigned Model", style="bold yellow")
                table.add_column("Reasoning", style="dim")

                table.add_row("Basic Tier", ta.basic_model, ta.reasoning.get("basic", "Fast everyday text"))
                table.add_row("Complex Tier", ta.complex_model, ta.reasoning.get("complex", "Deep reasoning/code"))
                table.add_row("Vision Tier", ta.vision_model or "None (OCR Enabled)", ta.reasoning.get("vision", "Multimodal images"))
                table.add_row("Default Tier", ta.default_model, ta.reasoning.get("default", "Standard fallback"))

                console.print(table)
            return True, None

        if cmd == "/model":
            if not args:
                console.print("[red]Usage: /model <model_name> or /model auto[/red]")
                return True, None
            m_name = args[0]
            if m_name.lower() in ["auto", "smart", "reset"]:
                self.orchestrator.router.clear_override()
                self.orchestrator.session_state.override_model = None
                console.print("[green]Reset model override. Smart Routing is now ACTIVE.[/green]")
            else:
                self.orchestrator.router.set_override_model(m_name)
                self.orchestrator.session_state.override_model = m_name
                console.print(f"[green]Override set: All subsequent prompts will route directly to '{m_name}'.[/green]")
            return True, None

        if cmd == "/compact":
            res = self.orchestrator.context_manager.force_compact()
            if res.compacted:
                console.print(f"[green]Context compacted: {res.messages_condensed} messages condensed ({res.original_tokens} -> {res.new_tokens} tokens).[/green]")
            else:
                console.print("[yellow]Context is already minimal, no compaction needed.[/yellow]")
            return True, None

        if cmd == "/clear":
            self.orchestrator.context_manager.clear_history()
            console.print("[green]Chat history cleared.[/green]")
            return True, None

        console.print(f"[red]Unknown command '{cmd}'. Type /help to see available commands.[/red]")
        return True, None
