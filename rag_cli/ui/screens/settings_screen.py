"""Settings screen for RAG CLI application"""
import os
from typing import Any
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static, Input, Select, Label, Switch


class SettingsScreen(ModalScreen):
    """Modal screen for application settings"""

    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def on_mount(self) -> None:
        """Load current settings when screen is mounted"""
        # Get settings from the settings manager (which reflects current RAG state)
        app = self.app  # type: ignore
        settings = app.settings_manager.settings  # type: ignore

        # Update input fields with current values
        self.query_one("#model-input", Input).value = str(
            settings.get("ollama_model", "llama3.2:3b")
        )
        self.query_one("#temp-input", Input).value = str(
            settings.get("temperature", 0.0)
        )
        self.query_one("#chunk-input", Input).value = str(
            settings.get("chunk_size", 1000)
        )
        self.query_one("#overlap-input", Input).value = str(
            settings.get("chunk_overlap", 200)
        )
        self.query_one("#retrieval-input", Input).value = str(
            settings.get("retrieval_k", 3)
        )
        
        # Update show context switch
        self.query_one("#show-context-switch", Switch).value = settings.get("show_context", False)

        # Update LLM provider settings
        provider_select = self.query_one("#provider-select", Select)
        provider_select.value = settings.get("llm_provider", "ollama")

        # Show API key status (from environment or settings)
        api_key_value = settings.get("api_key", "")
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
            api_key_value = "(Using environment variable)"
        self.query_one("#api-key-input", Input).value = str(api_key_value)
        self.query_one("#api-base-input", Input).value = str(
            settings.get("api_base_url", "")
        )
        self.query_one("#openai-model-input", Input).value = str(
            settings.get("openai_model", "gpt-3.5-turbo")
        )
        self.query_one("#anthropic-model-input", Input).value = str(
            settings.get("anthropic_model", "claude-3-haiku-20240307")
        )
        
        # Update reranker settings
        self.query_one("#use-reranker-switch", Switch).value = settings.get("use_reranker", False)
        self.query_one("#reranker-model-input", Input).value = str(
            settings.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        self.query_one("#reranker-top-k-input", Input).value = str(
            settings.get("reranker_top_k", 3)
        )
        
        # Update query expansion settings
        self.query_one("#use-query-expansion-switch", Switch).value = settings.get("use_query_expansion", False)
        self.query_one("#query-expansion-model-input", Input).value = str(
            settings.get("query_expansion_model", "llama3.2:3b")
        )
        self.query_one("#expansion-queries-input", Input).value = str(
            settings.get("expansion_queries", 3)
        )

        # Update provider-specific field visibility
        self._update_provider_fields(settings.get("llm_provider", "ollama"))

        # Ensure Ollama fields are visible by default if provider is ollama
        if settings.get("llm_provider", "ollama") == "ollama":
            try:
                ollama_fields = self.query(".ollama-field")
                for field in ollama_fields:
                    field.add_class("visible")
            except Exception:
                pass

    def compose(self) -> ComposeResult:
        with Container(id="settings-container"):
            yield Static("⚙️ Settings", id="settings-title")

            with VerticalScroll():
                # LLM Provider Settings
                yield Label("LLM Provider:")
                yield Select(
                    [
                        ("Ollama (Local)", "ollama"),
                        ("OpenAI API", "openai"),
                        ("Anthropic API", "anthropic"),
                    ],
                    value="ollama",
                    id="provider-select",
                )

                yield Static("")  # Spacer

                # Ollama-specific settings
                yield Label(
                    "Ollama Model:", id="ollama-model-label", classes="ollama-field"
                )
                yield Input(
                    value="llama3.2",
                    placeholder="Ollama model name",
                    id="model-input",
                    classes="ollama-field",
                )

                # API-specific settings (initially hidden)
                yield Label("API Key:", id="api-key-label", classes="api-field")
                yield Input(
                    value="",
                    placeholder="Your API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)",
                    password=True,
                    id="api-key-input",
                    classes="api-field",
                )

                yield Label(
                    "API Base URL (optional):", id="api-base-label", classes="api-field"
                )
                yield Input(
                    value="",
                    placeholder="Custom API endpoint (or set OPENAI_API_BASE env var)",
                    id="api-base-input",
                    classes="api-field",
                )

                yield Label(
                    "OpenAI Model:", id="openai-model-label", classes="openai-field"
                )
                yield Input(
                    value="gpt-3.5-turbo",
                    placeholder="OpenAI model name",
                    id="openai-model-input",
                    classes="openai-field",
                )

                yield Label(
                    "Anthropic Model:",
                    id="anthropic-model-label",
                    classes="anthropic-field",
                )
                yield Input(
                    value="claude-3-haiku-20240307",
                    placeholder="Anthropic model name",
                    id="anthropic-model-input",
                    classes="anthropic-field",
                )

                yield Static("")  # Spacer

                # General settings
                yield Label("Temperature (0.0-1.0):")
                yield Input(value="0.0", placeholder="0.0", id="temp-input")

                yield Label("Chunk Size:")
                yield Input(value="1000", placeholder="1000", id="chunk-input")

                yield Label("Chunk Overlap:")
                yield Input(value="200", placeholder="200", id="overlap-input")

                yield Label("Retrieval Count (k):")
                yield Input(value="3", placeholder="3", id="retrieval-input")

                with Horizontal():
                    yield Switch(value=True, id="auto-save-switch")
                    yield Label("Auto-save chat history")

                with Horizontal():
                    yield Switch(value=False, id="dark-mode-switch")
                    yield Label("Dark mode")

                with Horizontal():
                    yield Switch(value=False, id="show-context-switch")
                    yield Label("Show retrieved context with responses")

                yield Static("")  # Spacer
                yield Label("🔍 Reranking Settings", classes="section-header")
                
                with Horizontal():
                    yield Switch(value=False, id="use-reranker-switch")
                    yield Label("Use reranking to improve retrieval quality")

                yield Label("Reranker Model:", classes="reranker-field")
                yield Input(
                    value="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    placeholder="Reranker model name",
                    id="reranker-model-input",
                    classes="reranker-field",
                )

                yield Label("Reranker Top K:", classes="reranker-field")
                yield Input(
                    value="3",
                    placeholder="Number of documents after reranking",
                    id="reranker-top-k-input",
                    classes="reranker-field",
                )

                yield Static("")  # Spacer
                yield Label("🔄 Query Expansion Settings", classes="section-header")
                
                with Horizontal():
                    yield Switch(value=False, id="use-query-expansion-switch")
                    yield Label("Use query expansion to improve retrieval")

                yield Label("Query Expansion Model:", classes="query-expansion-field")
                yield Input(
                    value="llama3.2:3b",
                    placeholder="Small model for query expansion",
                    id="query-expansion-model-input",
                    classes="query-expansion-field",
                )

                yield Label("Number of Expanded Queries:", classes="query-expansion-field")
                yield Input(
                    value="3",
                    placeholder="How many query variations to generate",
                    id="expansion-queries-input",
                    classes="query-expansion-field",
                )

                yield Static("")  # Spacer

                with Horizontal(classes="button-row"):
                    yield Button("Save", variant="primary", id="save-settings")
                    yield Button("Cancel", id="cancel-settings")

    def _update_provider_fields(self, provider: str) -> None:
        """Update visibility of provider-specific fields"""
        # Hide all provider-specific fields first
        for field_class in [
            "api-field",
            "openai-field",
            "anthropic-field",
            "ollama-field",
        ]:
            try:
                fields = self.query(f".{field_class}")
                for field in fields:
                    field.remove_class("visible")
            except Exception:
                pass

        # Show/hide Ollama fields using proper Textual visibility
        try:
            ollama_fields = self.query(".ollama-field")
            for field in ollama_fields:
                if provider == "ollama":
                    field.add_class("visible")
                else:
                    field.remove_class("visible")
        except Exception:
            pass

        # Show API fields for non-Ollama providers
        if provider in ["openai", "anthropic"]:
            try:
                api_fields = self.query(".api-field")
                for field in api_fields:
                    field.add_class("visible")
            except Exception:
                pass

        # Show provider-specific model fields
        if provider == "openai":
            try:
                openai_fields = self.query(".openai-field")
                for field in openai_fields:
                    field.add_class("visible")
            except Exception:
                pass
        elif provider == "anthropic":
            try:
                anthropic_fields = self.query(".anthropic-field")
                for field in anthropic_fields:
                    field.add_class("visible")
            except Exception:
                pass

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle provider selection change"""
        if event.select.id == "provider-select":
            self._update_provider_fields(str(event.value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses in settings screen"""
        if event.button.id == "cancel-settings":
            self.dismiss()
        elif event.button.id == "save-settings":
            # Get all input values
            provider_select = self.query_one("#provider-select", Select)
            model_input = self.query_one("#model-input", Input)
            temp_input = self.query_one("#temp-input", Input)
            chunk_input = self.query_one("#chunk-input", Input)
            overlap_input = self.query_one("#overlap-input", Input)
            retrieval_input = self.query_one("#retrieval-input", Input)
            api_key_input = self.query_one("#api-key-input", Input)
            api_base_input = self.query_one("#api-base-input", Input)
            openai_model_input = self.query_one("#openai-model-input", Input)
            anthropic_model_input = self.query_one("#anthropic-model-input", Input)
            show_context_switch = self.query_one("#show-context-switch", Switch)
            use_reranker_switch = self.query_one("#use-reranker-switch", Switch)
            reranker_model_input = self.query_one("#reranker-model-input", Input)
            reranker_top_k_input = self.query_one("#reranker-top-k-input", Input)
            use_query_expansion_switch = self.query_one("#use-query-expansion-switch", Switch)
            query_expansion_model_input = self.query_one("#query-expansion-model-input", Input)
            expansion_queries_input = self.query_one("#expansion-queries-input", Input)

            # Validate and parse values
            try:
                new_settings = {
                    "llm_provider": provider_select.value or "ollama",
                    "ollama_model": model_input.value or "llama3.2:3b",
                    "temperature": float(temp_input.value or "0.0"),
                    "chunk_size": int(chunk_input.value or "1000"),
                    "chunk_overlap": int(overlap_input.value or "200"),
                    "retrieval_k": int(retrieval_input.value or "3"),
                    "api_key": api_key_input.value or "",
                    "api_base_url": api_base_input.value or "",
                    "openai_model": openai_model_input.value or "gpt-3.5-turbo",
                    "anthropic_model": anthropic_model_input.value or "claude-3-haiku-20240307",
                    "show_context": show_context_switch.value,
                    "use_reranker": use_reranker_switch.value,
                    "reranker_model": reranker_model_input.value or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "reranker_top_k": int(reranker_top_k_input.value or "3"),
                    "use_query_expansion": use_query_expansion_switch.value,
                    "query_expansion_model": query_expansion_model_input.value or "llama3.2:3b",
                    "expansion_queries": int(expansion_queries_input.value or "3"),
                }

                # Save settings to file first
                app = self.app  # type: ignore
                if app.settings_manager.save_settings(new_settings):  # type: ignore
                    # Update the RAG system with new settings (after they're saved)
                    app.rag.update_settings(**new_settings)  # type: ignore

                    # Update the stats display
                    app.update_stats()  # type: ignore

                    # Show success notification
                    app.notify("Settings saved successfully!")
                else:
                    app.notify(
                        "Settings applied but could not save to file",
                        severity="warning",
                    )

                self.dismiss()

            except ValueError as e:
                app.notify(f"Invalid settings: {str(e)}", severity="error")
            except Exception as e:
                app.notify(f"Error saving settings: {str(e)}", severity="error")

    async def action_dismiss(self, result: Any | None = None) -> None:
        """Handle escape key to close modal"""
        self.dismiss()

    def on_key(self, event) -> None:
        """Handle key events, specifically escape key to close modal"""
        if event.key == "escape":
            self.dismiss()
            event.prevent_default()

    def action_quit(self) -> None:
        """Quit the application from settings screen."""
        app = self.app  # type: ignore
        app.exit()