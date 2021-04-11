
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.image import Image

from pypalet.finder import Finder


class ImageScreen(Screen):
    def on_enter(self):
        app = App.get_running_app()
        app.finder.locate_board()
        image = self.children[0].children[1].children[0]
        img = image.data
        print(img)


Builder.load_string("""
<ImageScreen>:
    name: "image"
    BoxLayout:
        orientation: "vertical"
        TopBar:
        RelativeLayout:
            Image:
                keep_data: True
                source: app.image_file
                pos: self.pos
                size: self.size
            Button:
                size_hint: .2, .2
                text: app.image_file
""")
