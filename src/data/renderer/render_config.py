
"""
Default configurations for the flask-based
renderer for .svs files
"""
class SVS_Render_Config:
    SLIDE_CACHE_SIZE = 10
    DEEPZOOM_OVERLAP = 1
    DEEPZOOM_LIMIT_BOUNDS = True
    DEEPZOOM_FORMAT = 'jpeg'
    DEEPZOOM_TILE_SIZE = 254
    DEEPZOOM_TILE_QUALITY = 75
    ADDRESS = '127.0.0.1'
    port = 5000