
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen

from ipalet.topbar import TopBar


class TitleScreen(Screen):
    pass


class TitleCard(BoxLayout):
    pass


class StartButtonLayout(BoxLayout):
    pass


class LogoLabel(Label):
    pass


class StartButtonFile(Button):
    def on_press(self):
        App.get_running_app().image_file = 'images/IMG_0240.JPG'
        sm = self.get_root_window().children[0]
        sm.transition.direction = 'left'
        sm.current = 'image'


class StartButtonPhoto(Button):
    pass


Builder.load_string("""
<TitleScreen>:
    name: "title"
    BoxLayout:
        orientation: "vertical"
        TopBar:
        TitleCard:

<TitleCard@BoxLayout>:
    orientation: "vertical"
    spacing: 50
    LogoLabel:
        size_hint_y: .6
    StartButtonLayout:

<StartButtonLayout@BoxLayout>:
    orientation: "horizontal"
    spacing: self.height / 10
    padding: self.width / 4., self.height / 4.
    StartButtonFile:
        text: "File"
    StartButtonPhoto:
        text: "Photo"

""")
