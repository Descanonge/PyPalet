

from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label


class Help:
    """Mighty Bringer of Help."""

    def run_help(self):
        pass


class Settings:
    """Superb Bringer of Settings."""

    def run_settings(self):
        pass


class SettingsButton(Button):
    pass


class TopBar(BoxLayout):
    pass


Builder.load_string("""
<TopBar@BoxLayout>:
    orientation: "horizontal"
    height: 40
    size_hint_y: None
    SettingsButton:
    TitleLabel:
    HelpButton:


<SettingsButton@Button>:
    text: "S"
    width: self.height
    size_hint_x: None
    on_press: app.transition_screen("settings", "right")
<TitleLabel@Label>:
    text: "pyPalet"
    halign: "center"
    valign: "center"
<HelpButton@Button>:
    text: "?"
    width: self.height
    size_hint_x: None
""")
