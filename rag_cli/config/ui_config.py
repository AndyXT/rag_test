"""UI-related configuration constants"""

# Theme settings
THEME_SETTINGS = {
    "default": {
        "primary": "#007ACC",
        "secondary": "#40E0D0",
        "background": "#0C0C0C",
        "surface": "#1E1E1E",
        "error": "#FF6B6B",
        "warning": "#FFD93D",
        "success": "#6BCF7F",
        "info": "#4ECDC4"
    },
    "dark": {
        "primary": "#BB86FC",
        "secondary": "#03DAC6",
        "background": "#121212",
        "surface": "#1E1E1E",
        "error": "#CF6679",
        "warning": "#FFB74D",
        "success": "#81C784",
        "info": "#64B5F6"
    },
    "light": {
        "primary": "#1976D2",
        "secondary": "#00ACC1",
        "background": "#FAFAFA",
        "surface": "#FFFFFF",
        "error": "#D32F2F",
        "warning": "#F57C00",
        "success": "#388E3C",
        "info": "#0288D1"
    }
}

# UI Layout settings
LAYOUT_SETTINGS = {
    "sidebar_width": 30,
    "sidebar_min_width": 20,
    "sidebar_max_width": 50,
    "chat_input_height": 3,
    "chat_input_max_height": 10,
    "message_wrap_width": 80,
    "modal_width": 60,
    "modal_height": 80
}

# Keybindings
KEYBINDINGS = {
    "quit": "ctrl+q",
    "new_chat": "ctrl+n",
    "settings": "ctrl+s",
    "help": "ctrl+h",
    "toggle_sidebar": "ctrl+b",
    "focus_input": "ctrl+i",
    "clear_input": "ctrl+l",
    "submit": "enter",
    "multiline_submit": "ctrl+enter",
    "history_up": "up",
    "history_down": "down",
    "toggle_context": "ctrl+t",
    "export_chat": "ctrl+e",
    "search": "ctrl+f"
}

# Display settings
DISPLAY_SETTINGS = {
    "show_timestamps": True,
    "show_model_info": True,
    "show_token_count": False,
    "show_relevance_scores": False,
    "show_source_preview": True,
    "max_source_preview_chars": 200,
    "syntax_highlighting": True,
    "markdown_rendering": True,
    "auto_scroll": True,
    "message_grouping": True
}

# Animation settings
ANIMATION_SETTINGS = {
    "enabled": True,
    "typing_speed": 50,  # chars per second
    "fade_duration": 0.3,
    "slide_duration": 0.2,
    "progress_spinner": True
}

# Status messages
STATUS_MESSAGES = {
    "loading": "Loading...",
    "thinking": "Thinking...",
    "searching": "Searching documents...",
    "processing": "Processing response...",
    "ready": "Ready",
    "error": "Error occurred",
    "offline": "Offline mode"
}

# Modal titles
MODAL_TITLES = {
    "settings": "Settings",
    "help": "Help & Keyboard Shortcuts",
    "about": "About RAG CLI",
    "export": "Export Chat",
    "import": "Import Documents",
    "database": "Database Management",
    "history": "Chat History"
}

# Emoji mappings
EMOJI_MAPPINGS = {
    "user": "👤",
    "assistant": "🤖",
    "system": "⚙️",
    "error": "❌",
    "warning": "⚠️",
    "success": "✅",
    "info": "ℹ️",
    "document": "📄",
    "folder": "📁",
    "search": "🔍",
    "settings": "⚙️",
    "database": "🗄️"
}

# Progress indicators
PROGRESS_INDICATORS = {
    "spinner": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    "bar": ["▱", "▰"],
    "dots": ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
}

# Message formatting
MESSAGE_FORMATTING = {
    "user_prefix": "You",
    "assistant_prefix": "Assistant",
    "system_prefix": "System",
    "timestamp_format": "%H:%M:%S",
    "date_format": "%Y-%m-%d",
    "max_message_length": 10000,
    "truncate_indicator": "... (truncated)",
    "code_block_theme": "monokai"
}

# Notification settings
NOTIFICATION_SETTINGS = {
    "enabled": True,
    "duration": 3,  # seconds
    "position": "bottom-right",  # top-left, top-right, bottom-left, bottom-right
    "sound": False,
    "stack": True,
    "max_stack": 3
}