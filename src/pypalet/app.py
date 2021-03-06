
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty, ObjectProperty

from pypalet.title_screen import TitleScreen
from pypalet.settings import SettingsScreen
from pypalet.image import ImageScreen

from pypalet.finder import Finder, Params

from skimage.io import imread


class ResultsScreen(Screen):
    pass


class PyPaletApp(App):
    image_file = StringProperty("")
    finder = ObjectProperty(None)
    params = ObjectProperty(None)
    last_screen = StringProperty("title")

    def on_image_file(self, *args):
        self.finder = Finder(imread(self.image_file), self.params)
        self.transition_screen('image', 'left')

    def transition_screen(self, to: str, direction='left'):
        self.root.transition.direction = direction
        self.last_screen = self.root.current
        self.root.current = to

    def build(self):
        self.params = Params()

        root = Builder.load_string("""
ScreenManager:
    TitleScreen:
    SettingsScreen:
    ImageScreen:
        """)
        return root
