from collections import OrderedDict
from io import BytesIO
from optparse import OptionParser
import os
from threading import Lock

from flask import Flask, abort, make_response, render_template, url_for
from render_config import SVS_Render_Config

if os.name == 'nt':
    _dll_path = os.getenv('OPENSLIDE_PATH')
    if _dll_path is not None:
        if hasattr(os, 'add_dll_directory'):
            # Python >= 3.8
            with os.add_dll_directory(_dll_path):
                import openslide
        else:
            # Python < 3.8
            _orig_path = os.environ.get('PATH', '')
            os.environ['PATH'] = _orig_path + ';' + _dll_path
            import openslide

            os.environ['PATH'] = _orig_path
else:
    import openslide

from openslide import OpenSlide, OpenSlideError
from openslide.deepzoom import DeepZoomGenerator

SLIDES_DIR = 'check'
BASE_PATH = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir, 'dataset', 'data'))
SLIDES_PATH = os.path.join(BASE_PATH, SLIDES_DIR)
print(SLIDES_PATH)

app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('DEEPZOOM_MULTISERVER_SETTINGS', silent=True)


class _SlideCache:
    def __init__(self, cache_size, dz_opts):
        self.cache_size = cache_size
        self.dz_opts = dz_opts
        self._lock = Lock()
        self._cache = OrderedDict()

    def get(self, path):
        with self._lock:
            if path in self._cache:
                # Move to end of LRU
                slide = self._cache.pop(path)
                self._cache[path] = slide
                return slide

        osr = OpenSlide(path)
        slide = DeepZoomGenerator(osr, **self.dz_opts)
        try:
            mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
            slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            slide.mpp = 0

        with self._lock:
            if path not in self._cache:
                if len(self._cache) == self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[path] = slide
        return slide


class _Directory:
    def __init__(self, basedir, relpath=''):
        self.name = os.path.basename(relpath)
        self.children = []
        for name in sorted(os.listdir(os.path.join(basedir, relpath))):
            cur_relpath = os.path.join(relpath, name)
            cur_path = os.path.join(basedir, cur_relpath)
            print(name, OpenSlide.detect_format(cur_path))
            if os.path.isdir(cur_path):
                cur_dir = _Directory(basedir, cur_relpath)
                if cur_dir.children:
                    self.children.append(cur_dir)
            elif OpenSlide.detect_format(cur_path):
                self.children.append(_SlideFile(cur_relpath))


class _SlideFile:
    def __init__(self, relpath):
        self.name = os.path.basename(relpath)
        self.url_path = relpath


@app.before_first_request
def _setup():
    app.basedir = os.path.abspath(app.config['SLIDE_DIR'])
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = {v: app.config[k] for k, v in config_map.items()}
    app.cache = _SlideCache(app.config['SLIDE_CACHE_SIZE'], opts)


def _get_slide(path):
    path = os.path.abspath(os.path.join(app.basedir, path))
    if not path.startswith(app.basedir + os.path.sep):
        # Directory traversal
        abort(404)
    if not os.path.exists(path):
        abort(404)
    try:
        slide = app.cache.get(path)
        slide.filename = os.path.basename(path)
        return slide
    except OpenSlideError:
        abort(404)


@app.route('/')
def index():
    return render_template('files.html', root_dir=_Directory(app.basedir))


@app.route('/<path:path>')
def slide(path):
    slide = _get_slide(path)
    slide_url = url_for('dzi', path=path)
    return render_template(
        'slide-fullpage.html',
        slide_url=slide_url,
        slide_filename=slide.filename,
        slide_mpp=slide.mpp,
    )


@app.route('/<path:path>.dzi')
def dzi(path):
    slide = _get_slide(path)
    format = app.config['DEEPZOOM_FORMAT']
    resp = make_response(slide.get_dzi(format))
    resp.mimetype = 'application/xml'
    return resp


@app.route('/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(path, level, col, row, format):
    slide = _get_slide(path)
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = slide.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = BytesIO()
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


if __name__ == '__main__':

    render_config = dict(
        SLIDE_DIR = SLIDES_PATH,
        DEEPZOOM_OVERLAP = 1,
        DEEPZOOM_LIMIT_BOUNDS = True,
        DEEPZOOM_FORMAT = 'jpeg',
        DEEPZOOM_TILE_SIZE = 254,
        DEEPZOOM_TILE_QUALITY = 75,
        ADDRESS = '127.0.0.1',
        port = 5000
    )

    app.config.from_object(SVS_Render_Config)
    app.config['SLIDE_DIR'] = SLIDES_PATH
    print(app.config)
    app.run(
        host=render_config.get('ADDRESS', '127.0.0.1'),
        port=render_config.get('port', 5000), 
        threaded=True
    )