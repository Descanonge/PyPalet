
from kivy.core import core_select_lib
from kivy.uix.image import Image
from kivy.properties import NumericProperty, ListProperty, BooleanProperty

providers = ()
providers += (('opencv', 'camera_opencv', 'CameraOpenCV'), )
CoreCamera = core_select_lib('camera', (providers))


class Camera(Image):
    '''Camera class. See module documentation for more information.
    '''
    play = BooleanProperty(True)
    index = NumericProperty(-1)
    resolution = ListProperty([-1, -1])

    def __init__(self, **kwargs):
        self._camera = None
        super(Camera, self).__init__(**kwargs)
        if self.index == -1:
            self.index = 0
        on_index = self._on_index
        fbind = self.fbind
        fbind('index', on_index)
        fbind('resolution', on_index)
        on_index()

    def on_tex(self, *args):
        self.canvas.ask_update()

    def _on_index(self, *largs):
        self._camera = None
        if self.index < 0:
            return
        if self.resolution[0] < 0 or self.resolution[1] < 0:
            self._camera = CoreCamera(index=self.index, stopped=True)
        else:
            self._camera = CoreCamera(index=self.index,
                                      resolution=self.resolution, stopped=True)
        self._camera.bind(on_load=self._camera_loaded)
        if self.play:
            self._camera.start()
            self._camera.bind(on_texture=self.on_tex)

    def _camera_loaded(self, *largs):
        if self._camera.texture is not None:
            self.texture = self._camera.texture
            self.texture_size = list(self.texture.size)

    def on_play(self, instance, value):
        if not self._camera:
            return
        if value:
            self._camera.start()
        else:
            self._camera.stop()


def take_photo(*args):
    pass


def choose_file(*args):
    pass
