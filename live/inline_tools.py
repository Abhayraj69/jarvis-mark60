"""Declarations of the tools handled inside the Live session itself."""

TOOL_DECLARATIONS = [
    # ── Inline tools ─────────────────────────────────────────────────────────
    # These stay here (rather than in an actions/*.py TOOL dict) because their
    # handling is woven into live-session state — vision capture/injection,
    # camera stream, memory writes, the monitor engine, and shutdown. All other
    # tools live in their own action file and are auto-discovered by
    # core.action_loader (see JarvisLive.__init__).
    #
    # Keep these terse: every character here is sent on every turn, and the
    # behavioural rules ("say nothing after shutdown", "save silently") live
    # ONCE in core/prompt.txt, not here. tests/test_tool_budget.py enforces
    # the size limits.
    {
        "name": "system_status",
        "description": "Live CPU, RAM, GPU, temperature, uptime and process count.",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        # The time in the system prompt is frozen at connect, and a resumed
        # session can run for hours — it said "12:18" at 12:20.
        "name": "get_time",
        "description": "The current local time and date. Only needed when no [CLOCK] note "
                       "has arrived yet; otherwise answer from the latest [CLOCK].",
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "screen_process",
        "description": (
            "Capture the screen or webcam so you can see it — you have no vision "
            "without this. The image is sent to you right after; answer from it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "screen (default) | camera"},
                "text":  {"type": "STRING", "description": "The question about the image"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "close_camera",
        "description": "Close the live webcam view.",
        "parameters": {"type": "OBJECT", "properties": {}, "required": []},
    },
    {
        "name": "manage_monitor",
        "description": (
            "Topics checked once a day for new developments ('monitor X', "
            "'track X'). No crypto, financial or trading topics."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "add | remove | list"},
                "topic":  {"type": "STRING", "description": "Topic, e.g. 'AI news'"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "shutdown_jarvis",
        "description": (
            "Sleep until 'Hey Jarvis'. ONLY when the user tells YOU to sleep "
            "or says bye to you. Never for goodbyes to others, background/TV "
            "audio, or 'stop' about a task."
        ),
        "parameters": {"type": "OBJECT", "properties": {}},
    },
    {
        "name": "save_memory",
        "description": (
            "Store a durable personal fact: name, city, job, preference, "
            "relationship, project, plan. Not for one-off commands or searches."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": "identity | preferences | projects | relationships | wishes | notes",
                },
                "key":   {"type": "STRING", "description": "snake_case key, e.g. sister_name"},
                "value": {"type": "STRING", "description": "Concise value, in English"},
            },
            "required": ["category", "key", "value"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Search everything stored about the user, including the "
            "[ALSO REMEMBERED] keys the prompt had no room for. Local and instant."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "Name or topic; empty = list all"},
            },
            "required": [],
        },
    },
    {
        # NON_BLOCKING: the Live model keeps talking (its one-sentence
        # acknowledgement) while the reasoning core works; the answer arrives
        # later as a scheduled FunctionResponse — see _start_think.
        "name": "think",
        "behavior": "NON_BLOCKING",
        "description": (
            "Your reasoning core, for real analysis only: multi-step plans, detailed "
            "comparisons, advice with tradeoffs, long explanations. Answer simple "
            "questions and quick maths yourself."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING",
                          "description": "The full request, with every specific the user gave"},
                "include_screen": {"type": "BOOLEAN",
                                   "description": "Attach a screenshot (the request is about what's on screen)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "undo",
        "description": (
            "Reverse YOUR last change: a file moved/renamed/created/written or a "
            "setting changed. Not the app's Ctrl+Z (that is computer_settings undo)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "undo (default) | list"},
            },
            "required": [],
        },
    },
]
