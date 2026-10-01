"""What each Live session is built from: system prompt, tools and connect
config. Mixed into JarvisLive (main.py).
"""

import platform as _platform
from google.genai import types
from memory.memory_manager import load_memory, format_memory_for_prompt
from memory.config_manager import (
    get_media_resolution, get_proactive_audio_enabled, get_thinking_enabled,
    get_turn_tuning, get_voice,
)
from live.inline_tools import TOOL_DECLARATIONS
from live.prompting import (
    _describe_limits, _describe_tools, _load_api_config, _load_system_prompt,
    _render_prompt,
)


class SessionConfigMixin:
    """What each Live session is built from: system prompt, tools and connect config."""

    def _assemble_system_prompt(self) -> str:
        """Build the full system-instruction text: current time, identity
        (assistant name / how to address the user), the memory block, then
        the static JARVIS protocol from prompt.txt.

        Extracted out of _build_config() so Local Mode's tool-calling loop
        (_run_local_loop) gets the exact same system prompt content the
        Gemini Live path builds, instead of a second, drifting copy of this
        assembly logic. _build_config()'s own output is unchanged by this
        split — it just calls this instead of inlining the same code."""
        from datetime import datetime

        # Load customization from config
        try:
            _cfg = _load_api_config()
            self._asst_name = (_cfg.get("assistant_name") or "JARVIS").strip()
            _user_name = (_cfg.get("user_name") or "").strip()
        except Exception:
            self._asst_name = "JARVIS"
            _user_name = ""

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[SESSION START]\n"
            f"This session started: {time_str}\n"
            f"That is NOT the current time. [CLOCK] notes arrive every minute: answer the "
            f"time or date from the latest one, never read the tag aloud, never reply to it. "
            f"Call get_time only if there is no [CLOCK] note yet.\n\n"
        )

        # Identity injection — overrides any hardcoded name in prompt.txt
        _addr = (f"ADDRESS: Call the user '{_user_name}' — sparingly, at most "
                 f"once per reply."
                 if _user_name
                 else "ADDRESS: \"sir\", at most once per reply.")
        identity_ctx = (
            f"[IDENTITY]\n"
            f"You are {self._asst_name}.\n"
            f"{_addr}\n\n"
        )

        # Everything the model is told about *itself* is derived here, not
        # written into prompt.txt: the name comes from config, the platform from
        # the host, the capability list from the registries that were just
        # discovered. Rename the assistant, add a plugin or move to another OS
        # and this follows without anyone editing a prompt.
        values = {
            "assistant_name": self._asst_name,
            "platform": f"{_platform.system()} {_platform.release()}".strip(),
        }
        # The capability/limit lists only cost tokens if the prompt asks for
        # them — the tool declarations already carry every description.
        if "{capabilities}" in sys_prompt or "{limits}" in sys_prompt:
            _all_decls = self._all_tool_declarations()
            _names = {(d.get("name") if isinstance(d, dict) else getattr(d, "name", ""))
                      for d in _all_decls}
            values["capabilities"] = _describe_tools(_all_decls)
            values["limits"] = _describe_limits(
                has_vision="screen_process" in _names, has_mic=True)
        sys_prompt = _render_prompt(sys_prompt, values)

        parts = [time_ctx, identity_ctx]
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)
        return "\n".join(parts)

    def _all_tool_declarations(self) -> list[dict]:
        """The same Gemini-shaped tool list _build_config() feeds to the Live
        API — TOOL_DECLARATIONS + every discovered action + every enabled
        plugin — for Local Mode to convert into OpenAI/Ollama's tool format
        (see core.tool_schema)."""
        return (
            TOOL_DECLARATIONS
            + self._action_registry.get_tool_declarations()
            + self._plugin_registry.get_tool_declarations()
            + self._connector_declarations
        )

    # Defaults for the Live session's context-window compression (see
    # _build_config). Override per install with
    #   "live_context": {"trigger_tokens": 12000, "target_tokens": 6000}
    # in config/api_keys.json. Bounded so a typo can't disable compression
    # or squeeze the window below what one tool result needs.
    _CTX_TRIGGER_DEFAULT = 12_000
    _CTX_TARGET_DEFAULT  = 6_000

    def _live_context_limits(self) -> tuple[int, int]:
        """(trigger_tokens, target_tokens) for context-window compression."""
        try:
            raw = (_load_api_config().get("live_context") or {})
        except Exception:
            raw = {}
        def _num(key, default, lo, hi):
            try:
                v = int(raw.get(key, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(v, hi))
        trigger = _num("trigger_tokens", self._CTX_TRIGGER_DEFAULT, 4_000, 120_000)
        target  = _num("target_tokens",  self._CTX_TARGET_DEFAULT,  2_000, trigger - 1_000)
        return trigger, target

    def _build_config(self) -> types.LiveConnectConfig:
        system_instruction = self._assemble_system_prompt()

        cfg = dict(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction=system_instruction,
            tools=[{"function_declarations": self._all_tool_declarations()}],
            # Hand back the handle captured from the last session_resumption
            # update. `handle=None` is exactly the old behaviour (ask for
            # handles, start fresh), so the first connect of a run is unchanged.
            session_resumption=types.SessionResumptionConfig(
                handle=self._resume_handle
            ),
            # Sliding-window compression: session never dies from a full context
            # window — JARVIS can stay in one conversation for hours.
            # trigger/target are set explicitly: with the defaults, compression
            # only kicked in near the model's limit, so telemetry showed a
            # median of ~27k input tokens per turn (p90 50k) — most of it
            # stale history. Compressing at 12k down to 6k keeps every turn
            # small, which is both faster and measurably better at picking
            # the right tool. Tunable from api_keys.json "live_context".
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=self._live_context_limits()[0],
                sliding_window=types.SlidingWindow(
                    target_tokens=self._live_context_limits()[1],
                ),
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=get_voice()
                    )
                )
            ),
        )
        if self._enhanced_live:
            # Proactive audio: JARVIS stays silent when speech isn't addressed
            # to it (background chatter, talking to someone else in the room).
            # (Affective dialog was dropped: gemini-3.1-flash-live does not
            #  support it, and it never reliably detected tone in practice.
            #  To restore it on a 2.5 native-audio model, add back:
            #  cfg["enable_affective_dialog"] = True )
            if get_proactive_audio_enabled():
                cfg["proactivity"] = types.ProactivityConfig(proactive_audio=True)

        if self._tuned_live:
            cfg.update(self._tuning_config())

        return types.LiveConnectConfig(**cfg)

    def _tuning_config(self) -> dict:
        """The optional knobs, kept apart so one bad field can be dropped wholesale.

        Every one of these is a preview-API field. If a future model release
        stops accepting any of them the connection fails at setup, so the run
        loop turns `_tuned_live` off and reconnects on the plain config rather
        than leaving the user with an assistant that will not start.
        """
        out: dict = {}

        # How long the server waits through a pause before deciding your turn is
        # over. This — not the size of the prompt — is what most of the delay
        # before a reply actually is, and the default has to suit everybody, so
        # it is necessarily cautious.
        turn = get_turn_tuning()
        if turn.get("enabled", True):
            detect = types.AutomaticActivityDetection(
                silence_duration_ms=turn["silence_ms"],
                prefix_padding_ms=turn["prefix_ms"],
            )
            if turn["end_sensitivity"] == "high":
                detect.end_of_speech_sensitivity = types.EndSensitivity.END_SENSITIVITY_HIGH
            elif turn["end_sensitivity"] == "low":
                detect.end_of_speech_sensitivity = types.EndSensitivity.END_SENSITIVITY_LOW
            if turn["start_sensitivity"] == "high":
                detect.start_of_speech_sensitivity = types.StartSensitivity.START_SENSITIVITY_HIGH
            elif turn["start_sensitivity"] == "low":
                detect.start_of_speech_sensitivity = types.StartSensitivity.START_SENSITIVITY_LOW
            out["realtime_input_config"] = types.RealtimeInputConfig(
                automatic_activity_detection=detect)

        # Screenshots and camera frames are tokenised at this resolution and then
        # stay in the session's context. 'medium' keeps on-screen text legible
        # for a fraction of a full-resolution frame.
        res = get_media_resolution()
        if res != "default":
            out["media_resolution"] = {
                "low":    types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high":   types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }[res]

        # Thinking is left at the server default deliberately. Forcing the budget
        # to zero was measured on gemini-3.1-flash-live over interleaved trials
        # and did not make the first word arrive sooner — this model does not
        # appear to deliberate on the Live path, so pinning the field only adds a
        # way for a future release to behave differently. Set "thinking_enabled"
        # in config/api_keys.json to true to let it reason instead.
        if get_thinking_enabled():
            out["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)

        return out
