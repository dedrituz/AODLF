"""
Interactive Rich CLI Interface for the AODLF Prototype.
Implements the complete interactive lifecycle, provider selection, smart routing confirmation/veto,
model load verification with informative animations, and streaming chat.
"""

import asyncio
import sys
import time
from typing import Dict, List, Optional, Tuple
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from app.cli.animations import LoadingAnimation
from app.cli.commands import CommandHandler
from app.config.env_manager import EnvManager
from app.config.settings import (
    SUPPORTED_PROVIDERS,
    TIMEOUT_CLOUD_SECONDS,
    TIMEOUT_LOCAL_SECONDS,
)
from app.core.orchestrator import BackendOrchestrator
from app.multimodal.doc_loader import DocumentLoader, PathValidationError
from app.providers.base import ModelInfo
from app.providers.manager import ProviderConnectionReport, ProviderManager
from app.routing.smart_assignment import SmartModelAssigner, TierAssignment
from app.routing.system_scanner import SystemScanner
from app import __version__

console = Console()


class CLIApp:
    """Interactive command-line application interface."""

    def __init__(self):
        self.env_manager = EnvManager()
        self.provider_manager = ProviderManager(env_manager=self.env_manager)
        self.orchestrator = BackendOrchestrator(provider_manager=self.provider_manager)
        self.command_handler = CommandHandler(
            env_manager=self.env_manager,
            provider_manager=self.provider_manager,
            orchestrator=self.orchestrator
        )
        self.running = True

    def print_banner(self):
        banner_text = Text()
        banner_text.append("====================================================================\n", style="bold cyan")
        banner_text.append(f"   AUTOMATED OPEN-DOMAIN LEARNING FRAMEWORK (AODLF) - PROTOTYPE v{__version__}\n", style="bold yellow")
        banner_text.append("   Intelligent Tiered Routing | Multimodal Media Engine | Local & Cloud\n", style="dim")
        banner_text.append("====================================================================", style="bold cyan")
        console.print(Panel(banner_text, border_style="cyan"))

    async def initialize(self) -> Optional[List[ProviderConnectionReport]]:
        """
        Step 1: Initialization.
        Parses .env, checks keys, checks local backend, and diagnoses connectivity.
        """
        console.print("[cyan]Initializing backend and scanning providers...[/cyan]")
        reports = await self.provider_manager.scan_and_test_all()

        connected_reports = [r for r in reports if r.connected]
        configured_failed = [r for r in reports if r.configured and not r.connected]
        placeholder_reports = [r for r in reports if r.empty_or_placeholder]
        local_reports = [r for r in reports if r.is_local and r.connected]

        console.print()
        if connected_reports:
            console.print(
                Panel(
                    f"[bold green]✓ Successful Initialization![/bold green]\n"
                    f"Connected to [bold]{len(connected_reports)}[/bold] inference provider(s).\n"
                    f"[dim]You can type [bold yellow]/help[/bold yellow] at any time for debug commands and diagnostics.[/dim]",
                    title="Initialization Status",
                    border_style="green"
                )
            )
        else:
            # Connection failure alert
            warn_text = Text()
            warn_text.append("⚠ ALERT: Backend is running without a live connection to any LLM!\n\n", style="bold red")

            if configured_failed:
                warn_text.append("The following providers were configured, but connection failed:\n", style="yellow")
                for cr in configured_failed:
                    warn_text.append(f"  • {cr.name}: {cr.status_message}\n", style="red")

            if placeholder_reports:
                warn_text.append("\nIncomplete or empty API key entries detected in .env:\n", style="yellow")
                for pr in placeholder_reports:
                    warn_text.append(f"  • {pr.name}: {pr.status_message}\n", style="dim")

            if not local_reports:
                warn_text.append("\nNo active local backend (Ollama, llama.cpp) was detected.\n", style="yellow")

            warn_text.append("\nOptions to resolve:\n", style="bold white")
            warn_text.append("  1. Use [bold yellow]/add-api <provider> <key>[/bold yellow] or [bold yellow]/edit-api provider=<tag> key=<key>[/bold yellow] to set keys.\n")
            warn_text.append("  2. Use [bold yellow]/set-endpoint <provider> <url>[/bold yellow] to set custom endpoint.\n")
            warn_text.append("  3. Start Ollama locally ('ollama serve') at http://localhost:11434.\n")
            warn_text.append("  4. Type [bold yellow]/help[/bold yellow] for full command list.", style="dim")

            console.print(Panel(warn_text, title="Connection Warning", border_style="red"))

        return reports

    async def select_provider_menu(self) -> Optional[str]:
        """
        Step 2: Provider selection menu.
        """
        while self.running:
            reports = await self.provider_manager.scan_and_test_all()
            table = Table(title="Select LLM Inference Provider", show_header=True, header_style="bold cyan")
            table.add_column("#", style="bold yellow", width=4)
            table.add_column("Provider", style="bold white")
            table.add_column("Type", style="cyan")
            table.add_column("Status", style="white")
            table.add_column("Models", style="green")

            active_choices: Dict[str, str] = {}
            idx = 1

            # Connected providers first
            sorted_reports = sorted(reports, key=lambda r: (not r.connected, not r.is_local))

            for r in sorted_reports:
                status_str = "[green]✓ Ready[/green]" if r.connected else ("[red]✗ Failed[/red]" if r.configured else "[dim]Not Configured[/dim]")
                models_str = f"{r.models_count} model(s)" if r.connected else "-"
                type_str = "Local" if r.is_local else "Cloud"
                table.add_row(str(idx), r.name, type_str, status_str, models_str)
                active_choices[str(idx)] = r.provider_id
                active_choices[r.provider_id.lower()] = r.provider_id
                idx += 1

            console.print(table)
            console.print("[dim]Enter number or provider ID. Or use commands: /add-api, /set-endpoint, /specs, /debug, /exit[/dim]")

            choice = Prompt.ask("\n[bold cyan]Select Provider[/bold cyan]").strip()

            if self.command_handler.is_command(choice):
                handled, signal = await self.command_handler.handle_command(choice, current_scope="menu")
                if signal == "exit":
                    self.running = False
                    return None
                continue

            if choice in active_choices:
                selected_pid = active_choices[choice]
                # Check if provider is connected
                provider = self.provider_manager.get_provider(selected_pid)
                if not provider:
                    console.print(f"[red]Could not load provider '{selected_pid}'.[/red]")
                    continue

                success, msg = await provider.test_connection()
                if not success:
                    console.print(f"[red]Cannot connect to {selected_pid.title()}: {msg}[/red]")
                    retry = Prompt.ask("Do you want to enter key/endpoint manually? (y/n)", choices=["y", "n"], default="y")
                    if retry == "y":
                        if SUPPORTED_PROVIDERS[selected_pid].requires_key:
                            key = Prompt.ask(f"Enter API Key for {selected_pid}")
                            self.env_manager.update_key(selected_pid, key)
                        else:
                            url = Prompt.ask(f"Enter Base URL for {selected_pid}", default="http://localhost:11434")
                            self.env_manager.update_key(selected_pid, url)
                    continue

                return selected_pid

            console.print(f"[red]Invalid selection '{choice}'. Please choose a valid provider number or ID.[/red]")

    async def select_and_configure_routing(self, provider_id: str) -> Optional[Tuple[TierAssignment, Optional[str]]]:
        """
        Step 3: Model Listing, Smart Routing & Tier Customization / Veto.
        """
        provider = self.provider_manager.get_provider(provider_id)
        if not provider:
            return None

        models = await provider.list_models()
        if not models:
            console.print(f"[red]No models found for provider '{provider_id}'.[/red]")
            return None

        # Display available models table
        table = Table(title=f"Available Models ({provider_id.upper()})", show_header=True, header_style="bold cyan")
        table.add_column("#", style="bold yellow", width=4)
        table.add_column("Model ID", style="bold white")
        table.add_column("Parameters", style="cyan")
        table.add_column("Context Window", style="magenta")
        table.add_column("Pricing / Cost", style="green")
        table.add_column("Capabilities", style="white")

        model_map: Dict[str, ModelInfo] = {}
        for idx, m in enumerate(models, 1):
            pricing_str = "Local / Free ($0.00)" if m.is_local else f"${m.cost_per_1m_input:.2f}/M in, ${m.cost_per_1m_output:.2f}/M out"
            table.add_row(str(idx), m.id, m.parameter_size, f"{m.context_length:,} tokens", pricing_str, m.display_tag)
            model_map[str(idx)] = m
            model_map[m.id.lower()] = m

        console.print(table)
        console.print()

        # Hardware Spec Scanning recommendation
        specs = None
        if provider_id == "ollama" or SUPPORTED_PROVIDERS[provider_id].is_local:
            scan_prompt = Prompt.ask(
                "[cyan]Would you like to scan local hardware specifications to optimize model tiers?[/cyan]",
                choices=["y", "n"],
                default="y"
            )
            if scan_prompt == "y":
                specs = SystemScanner.scan()
                console.print(f"[dim]{specs.summary}[/dim]\n")

        # Smart routing calculation
        tier_assign = SmartModelAssigner.assign_tiers(models, provider_id, specs)

        # Ask user: Smart Routing or Specific Model Override
        console.print("[bold yellow]Routing Options:[/bold yellow]")
        console.print("  [bold green]1. Automatic Smart Routing[/bold green] (Dynamically delegates Basic, Complex, and Vision prompts)")
        console.print("  [bold cyan]2. Specific Model Selection[/bold cyan] (Override routing and use a single fixed model)")
        console.print()

        routing_choice = Prompt.ask(
            "[bold cyan]Select mode[/bold cyan] (1 for Smart Routing, 2 or model name for Single Model)",
            default="1"
        ).strip()

        if self.command_handler.is_command(routing_choice):
            await self.command_handler.handle_command(routing_choice, current_scope="model_select")
            return None

        # If user picked a single model directly
        if routing_choice != "1" and (routing_choice == "2" or routing_choice in model_map):
            if routing_choice == "2":
                m_choice = Prompt.ask("Enter model number or name from the list above").strip()
                selected_model = model_map.get(m_choice.lower(), models[0]).id
            else:
                selected_model = model_map[routing_choice].id

            console.print(f"[green]✓ Selected single model override: [bold]{selected_model}[/bold][/green]")
            return tier_assign, selected_model

        # Smart Routing Confirmation
        while True:
            confirm_table = Table(title="Proposed Smart Routing Tier Assignments", show_header=True)
            confirm_table.add_column("Tier", style="bold cyan")
            confirm_table.add_column("Assigned Model", style="bold yellow")
            confirm_table.add_column("Rationale", style="white")

            confirm_table.add_row("Basic Tier", tier_assign.basic_model, tier_assign.reasoning.get("basic", "Lightweight queries"))
            confirm_table.add_row("Complex Tier", tier_assign.complex_model, tier_assign.reasoning.get("complex", "Deep reasoning/code"))
            confirm_table.add_row(
                "Vision Tier",
                tier_assign.vision_model or "None (OCR Fallback)",
                tier_assign.reasoning.get("vision", "Image & visual reasoning")
            )
            confirm_table.add_row("Default Tier", tier_assign.default_model, tier_assign.reasoning.get("default", "Standard fallback"))

            console.print(confirm_table)
            console.print()

            user_action = Prompt.ask(
                "[bold cyan]Accept smart model routing selection?[/bold cyan] ([green]y[/green]=Accept / [yellow]v[/yellow]=Veto & Customize / [red]s[/red]=Select Single Model)",
                choices=["y", "v", "s", "cancel"],
                default="y"
            )

            if user_action == "y":
                return tier_assign, None

            if user_action == "s":
                m_choice = Prompt.ask("Enter model number or name for single model").strip()
                selected_model = model_map.get(m_choice.lower(), models[0]).id
                return tier_assign, selected_model

            if user_action in ["v", "veto"]:
                # Sequential customization
                console.print("\n[yellow]--- Customize Routing Tiers (Press Enter to keep current, or type model name / 'cancel') ---[/yellow]")
                try:
                    b_in = Prompt.ask(f"1. Basic Model", default=tier_assign.basic_model).strip()
                    if b_in.lower() == "cancel":
                        console.print("[dim]Customization cancelled. Reverting to system specifications.[/dim]")
                        continue
                    tier_assign.basic_model = model_map.get(b_in.lower(), ModelInfo(id=b_in, name=b_in, provider_id=provider_id)).id

                    c_in = Prompt.ask(f"2. Complex Model", default=tier_assign.complex_model).strip()
                    if c_in.lower() == "cancel":
                        continue
                    tier_assign.complex_model = model_map.get(c_in.lower(), ModelInfo(id=c_in, name=c_in, provider_id=provider_id)).id

                    v_default = tier_assign.vision_model or "none"
                    v_in = Prompt.ask(f"3. Vision Model (optional, type 'none' for OCR only)", default=v_default).strip()
                    if v_in.lower() == "cancel":
                        continue
                    tier_assign.vision_model = None if v_in.lower() in ["none", "no", ""] else model_map.get(v_in.lower(), ModelInfo(id=v_in, name=v_in, provider_id=provider_id)).id

                    d_in = Prompt.ask(f"4. Default Model", default=tier_assign.default_model).strip()
                    if d_in.lower() == "cancel":
                        continue
                    tier_assign.default_model = model_map.get(d_in.lower(), ModelInfo(id=d_in, name=d_in, provider_id=provider_id)).id

                    console.print("[green]Custom routing saved![/green]\n")
                except Exception as e:
                    console.print(f"[red]Error in customization: {e}[/red]")
                    continue

    async def warmup_and_verify_model(
        self,
        provider_id: str,
        model_name: str
    ) -> bool:
        """
        Step 4: Model load and warmup with informative loading animation.
        Enforces timeouts: 30s for cloud, 180s (3m) for local backend.
        Allows user to retry.
        """
        provider = self.provider_manager.get_provider(provider_id)
        if not provider:
            return False

        is_local = provider_id == "ollama" or SUPPORTED_PROVIDERS.get(provider_id, ModelInfo(id="", name="", provider_id="")).is_local
        timeout = TIMEOUT_LOCAL_SECONDS if is_local else TIMEOUT_CLOUD_SECONDS

        while True:
            console.print(f"\n[cyan]Connecting to model [bold]{model_name}[/bold] ({'Local backend' if is_local else 'Cloud API'})...[/cyan]")

            async def do_warmup():
                return await provider.warmup_model(model_name, timeout_seconds=timeout)

            try:
                success, msg = await LoadingAnimation.run_with_spinner(
                    message=f"Loading & verifying model '{model_name}'",
                    coroutine_func=do_warmup,
                    timeout_seconds=timeout
                )
                if success:
                    console.print(f"[bold green]✓ {msg}[/bold green]\n")
                    return True
                else:
                    console.print(f"[bold red]✗ Model Load Error: {msg}[/bold red]")
            except Exception as e:
                console.print(f"[bold red]✗ Verification Failed: {str(e)}[/bold red]")

            retry = Prompt.ask("[yellow]Would you like to retry connecting?[/yellow] (y/n)", choices=["y", "n"], default="y")
            if retry != "y":
                return False

    async def run_chat_loop(self):
        """
        Step 5: Interactive Chat Mode.
        Supports text input, full-path attachments, OCR/Vision routing,
        streaming response, thought deltas, compaction, and slash commands.
        """
        session_st = self.orchestrator.session_state
        ta = self.orchestrator.tier_assignment

        # Alert if vision model is not present
        if not session_st.has_vision_model:
            console.print(
                Panel(
                    "[yellow]ℹ Multimodal Notice: No vision model is configured for this session.\n"
                    "All attached images will be automatically processed through the lightweight OCR pipeline.[/yellow]",
                    border_style="yellow"
                )
            )

        console.print(
            Panel(
                f"[bold green]Chat Mode Active[/bold green] | Provider: [cyan]{session_st.active_provider_id.upper()}[/cyan] | "
                f"Mode: [yellow]{f'Override ({session_st.override_model})' if session_st.override_model else 'Smart Routing'}[/yellow]\n"
                f"[dim]Attach files using full path in quotes (e.g. \"C:\\path\\to\\file.pdf\"). Commands: /help, /list, /debug, /route, /compact, /back, /exit[/dim]",
                title=f"Session: {session_st.title}",
                border_style="green"
            )
        )

        while self.running:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Exiting chat session...[/dim]")
                break

            if not user_input:
                continue

            # Command handling
            if self.command_handler.is_command(user_input):
                handled, signal = await self.command_handler.handle_command(user_input, current_scope="chat")
                if signal == "exit":
                    self.running = False
                    break
                elif signal == "back":
                    break
                continue

            # Stream processing
            console.print()
            thinking_buffer = ""
            content_buffer = ""
            in_thinking = False
            last_metadata: Dict[str, Any] = {}

            console.print("[dim]Assistant is processing...[/dim]")

            async for chunk in self.orchestrator.process_turn_stream(user_input=user_input):
                if chunk.event_type == "status":
                    console.print(f"[dim italic]⚡ {chunk.content}[/dim italic]")
                elif chunk.event_type == "thought_delta":
                    if not in_thinking:
                        in_thinking = True
                        console.print("\n[bold magenta]Thinking Process:[/bold magenta]")
                    console.print(f"[magenta]{chunk.content}[/magenta]", end="")
                    thinking_buffer += chunk.content
                    sys.stdout.flush()
                elif chunk.event_type == "content_delta":
                    if in_thinking:
                        in_thinking = False
                        console.print("\n\n[bold cyan]Assistant:[/bold cyan]")
                    elif not content_buffer:
                        console.print("\n[bold cyan]Assistant:[/bold cyan]")
                    console.print(chunk.content, end="")
                    content_buffer += chunk.content
                    sys.stdout.flush()
                elif chunk.event_type == "metadata":
                    last_metadata = chunk.metadata
                elif chunk.event_type == "error":
                    console.print(f"\n[bold red]Error: {chunk.content}[/bold red]")

            console.print()  # newline after stream

            # Render metadata & suggestions
            if last_metadata:
                elapsed = last_metadata.get("elapsed_seconds", 0)
                tok_sec = last_metadata.get("tokens_per_second", 0)
                routing_info = last_metadata.get("routing", {})
                model_used = routing_info.get("model", "")
                tier_used = routing_info.get("tier", "")

                meta_line = f"[dim]Time: {elapsed}s | Speed: {tok_sec} tok/s | Model: {model_used} [{tier_used}][/dim]"
                console.print(meta_line)

                suggestions = last_metadata.get("suggestions", [])
                if suggestions:
                    console.print("\n[bold yellow]Suggested Follow-Ups:[/bold yellow]")
                    for idx, q in enumerate(suggestions, 1):
                        console.print(f"  [dim]{idx}.[/dim] [white]{q}[/white]")

    async def main_loop(self):
        """Main application lifecycle controller."""
        self.print_banner()

        # Step 1: Initialize & Check Connections
        await self.initialize()

        while self.running:
            # Step 2: Select Provider
            provider_id = await self.select_provider_menu()
            if not provider_id or not self.running:
                break

            # Step 3: Configure Smart Routing / Model Select
            routing_result = await self.select_and_configure_routing(provider_id)
            if not routing_result:
                continue

            tier_assignment, override_model = routing_result

            # Step 4: Model Warmup / Load
            target_model = override_model or tier_assignment.default_model
            loaded = await self.warmup_and_verify_model(provider_id, target_model)
            if not loaded:
                console.print("[yellow]Returning to provider selection...[/yellow]\n")
                continue

            # Configure Orchestrator
            self.orchestrator.configure_session(
                provider_id=provider_id,
                tier_assignment=tier_assignment,
                override_model=override_model
            )

            # Step 5: Chat Loop
            await self.run_chat_loop()

        console.print("\n[bold green]Thank you for using AODLF! Goodbye.[/bold green]")


def start_cli():
    """CLI application entrypoint."""
    app = CLIApp()
    try:
        asyncio.run(app.main_loop())
    except KeyboardInterrupt:
        console.print("\n[dim]Process interrupted by user. Exiting.[/dim]")


if __name__ == "__main__":
    start_cli()
