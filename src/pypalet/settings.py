

from kivy.lang import Builder
from kivy.uix.screenmanager import Screen


class SettingsScreen(Screen):
    pass


Builder.load_string("""
<SettingsScreen>:
    name: "settings"
    BoxLayout:
        orientation: "vertical"
        height: 40
        padding: self.width / 4., self.height / 4.
        Button:
            size_hint_x: None
            width: self.height
            on_press: app.transition_screen(app.last_screen, "left")
""")
