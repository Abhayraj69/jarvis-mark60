"""Input guard (core/input_guard.py) and its use in computer_control and
send_message: nothing is typed or sent in someone else's window without a
yes on screen."""
import unittest
from unittest.mock import patch

from core import confirm, input_guard
from core.input_guard import Settings, decide


class DecideTest(unittest.TestCase):
    s = Settings()

    def test_typing_into_an_allowed_app(self):
        self.assertEqual(decide("type", {"text": "hi"}, "Visual Studio Code", "main.py", self.s), "allow")

    def test_typing_into_an_unlisted_app_asks(self):
        self.assertEqual(decide("type", {"text": "hi"}, "Finder", "", self.s), "confirm_app")

    def test_typing_a_draft_in_whatsapp_is_fine(self):
        self.assertEqual(decide("type", {"text": "on my way"}, "WhatsApp", "", self.s), "allow")

    def test_enter_in_whatsapp_asks(self):
        self.assertEqual(decide("press", {"key": "enter"}, "WhatsApp", "", self.s), "confirm_send")

    def test_send_hotkey_in_slack_asks(self):
        self.assertEqual(decide("hotkey", {"keys": "cmd+enter"}, "Slack", "", self.s), "confirm_send")

    def test_web_whatsapp_is_caught_by_window_title(self):
        self.assertEqual(decide("press", {"key": "return"}, "Google Chrome", "(3) WhatsApp", self.s),
                         "confirm_send")

    def test_enter_in_a_browser_search_is_fine(self):
        self.assertEqual(decide("press", {"key": "enter"}, "Google Chrome", "Google", self.s), "allow")

    def test_clicking_send_in_a_messaging_app_asks(self):
        self.assertEqual(decide("screen_click", {"description": "the Send button"}, "Telegram", "", self.s),
                         "confirm_send")

    def test_mouse_and_reads_are_never_gated(self):
        for action in ("click", "scroll", "screenshot", "copy", "screen_find"):
            self.assertEqual(decide(action, {}, "WhatsApp", "", self.s), "allow", action)

    def test_unknown_focus_does_not_block(self):
        self.assertEqual(decide("type", {"text": "x"}, "", "", self.s), "allow")

    def test_switched_off(self):
        self.assertEqual(decide("press", {"key": "enter"}, "WhatsApp", "", Settings(enabled=False)), "allow")

    def test_settings_text_lists(self):
        with patch("memory.config_manager.get_plugin_config",
                   return_value={"allowed_apps": "Finder, Preview", "messaging_apps": ""}):
            s = input_guard.load_settings()
        self.assertEqual(s.allowed_apps, ("Finder", "Preview"))
        self.assertEqual(s.messaging_apps, tuple(input_guard.DEFAULT_MESSAGING_APPS))


class _Gate:
    """Bind core/confirm.py to a fake HUD for the duration of a test."""

    def __enter__(self):
        self.shown = []
        confirm.bind(lambda t, d: self.shown.append((t, d)), lambda: None, lambda m: None)
        return self

    def __exit__(self, *exc):
        confirm.bind(None, None, None)
        confirm._pending = None


class ComputerControlTest(unittest.TestCase):
    def _run(self, params, app, title=""):
        from actions import computer_control as cc
        with patch.object(input_guard, "frontmost", return_value=(app, title)), \
             patch.object(input_guard, "load_settings", return_value=Settings()), \
             patch.object(input_guard, "acting"), \
             patch.object(cc, "_perform", return_value="performed") as perform, \
             patch.object(cc, "_focus_window") as focus:
            result = cc.computer_control(params)
        return result, perform, focus

    def test_enter_in_whatsapp_waits_for_confirm(self):
        with _Gate() as g:
            result, perform, _ = self._run({"action": "press", "key": "enter"}, "WhatsApp")
        self.assertTrue(result.startswith("[CONFIRMATION_PENDING]"))
        perform.assert_not_called()
        self.assertEqual(g.shown[0][0], "Send in WhatsApp?")

    def test_confirm_refocuses_then_sends(self):
        from actions import computer_control as cc
        with _Gate():
            self._run({"action": "press", "key": "enter"}, "WhatsApp")
            pending = confirm._pending
            with patch.object(cc, "_perform", return_value="Pressed: enter") as perform, \
                 patch.object(cc, "_focus_window") as focus:
                pending.run()
        focus.assert_called_once_with("WhatsApp")
        perform.assert_called_once()

    def test_typing_in_vscode_goes_straight_through(self):
        result, perform, _ = self._run({"action": "type", "text": "hello"}, "Code")
        self.assertEqual(result, "performed")
        perform.assert_called_once()


class SendMessageTest(unittest.TestCase):
    def test_send_message_waits_for_confirm(self):
        from actions import send_message as sm
        with _Gate() as g, \
             patch.object(sm, "_PYAUTOGUI", True), \
             patch.object(input_guard, "load_settings", return_value=Settings()), \
             patch.object(sm, "_deliver") as deliver:
            result = sm.send_message({"receiver": "Mom", "message_text": "Home by 8", "platform": "whatsapp"})
        self.assertTrue(result.startswith("[CONFIRMATION_PENDING]"))
        deliver.assert_not_called()
        self.assertEqual(g.shown[0], ("Send to Mom on Whatsapp?", "Home by 8"))

    def test_can_be_switched_off(self):
        from actions import send_message as sm
        with patch.object(sm, "_PYAUTOGUI", True), \
             patch.object(input_guard, "load_settings", return_value=Settings(confirm_messages=False)), \
             patch.object(sm, "_deliver", return_value="Message sent to Mom.") as deliver:
            result = sm.send_message({"receiver": "Mom", "message_text": "Home by 8"})
        self.assertEqual(result, "Message sent to Mom.")
        deliver.assert_called_once()


if __name__ == "__main__":
    unittest.main()
