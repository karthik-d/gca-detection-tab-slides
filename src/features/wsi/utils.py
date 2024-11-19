import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import color

# If True, display additional NumPy array stats (min, max, mean, is_binary).
ADDITIONAL_NP_STATS = False


def pil_to_np_rgb(pil_img):
  """
  Convert a PIL Image to a NumPy array.
  Note that RGB PIL (w, h) -> NumPy (h, w, 3).
  Args:
    pil_img: The PIL Image.
  Returns:
    The PIL image converted to a NumPy array.
  """
  t = Time()
  rgb = np.asarray(pil_img)
  np_info(rgb, "RGB", t.elapsed())
  return rgb


def np_to_pil(np_img):
  """
  Convert a NumPy array to a PIL Image.
  Args:
    np_img: The image represented as a NumPy array.
  Returns:
     The NumPy array converted to a PIL Image.
  """
  if np_img.dtype == "bool":
    np_img = np_img.astype("uint8") * 255
  elif np_img.dtype == "float64":
    np_img = (np_img * 255).astype("uint8")
  return Image.fromarray(np_img)


def np_info(np_arr, name=None, elapsed=None):
  """
  Display information (shape, type, max, min, etc) about a NumPy array.
  Args:
    np_arr: The NumPy array.
    name: The (optional) name of the array.
    elapsed: The (optional) time elapsed to perform a filtering operation.
  """

  if name is None:
    name = "NumPy Array"
  if elapsed is None:
    elapsed = "---"

  if ADDITIONAL_NP_STATS is False:
    print("%-20s | Time: %-14s  Type: %-7s Shape: %s" % (name, str(elapsed), np_arr.dtype, np_arr.shape))
  else:
    # np_arr = np.asarray(np_arr)
    max = np_arr.max()
    min = np_arr.min()
    mean = np_arr.mean()
    is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
    print("%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s" % (
      name, str(elapsed), min, max, mean, is_binary, np_arr.dtype, np_arr.shape))


def display_img(np_img, text=None, font_path="/Library/Fonts/Arial Bold.ttf", size=48, color=(255, 0, 0),
                background=(255, 255, 255), border=(0, 0, 0), bg=False):
  """
  Convert a NumPy array to a PIL image, add text to the image, and display the image.
  Args:
    np_img: Image as a NumPy array.
    text: The text to add to the image.
    font_path: The path to the font to use.
    size: The font size
    color: The font color
    background: The background color
    border: The border color
    bg: If True, add rectangle background behind text
  """
  result = np_to_pil(np_img)
  # if gray, convert to RGB for display
  if result.mode == 'L':
    result = result.convert('RGB')
  draw = ImageDraw.Draw(result)
  if text is not None:
    font = ImageFont.truetype(font_path, size)
    if bg:
      (x, y) = draw.textsize(text, font)
      draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
    draw.text((2, 0), text, color, font=font)
  result.show()


def mask_rgb(rgb, mask):
  """
  Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.
  Args:
    rgb: RGB image as a NumPy array.
    mask: An image mask to determine which pixels in the original image should be displayed.
  Returns:
    NumPy array representing an RGB image with mask applied.
  """
  t = Time()
  result = rgb * np.dstack([mask, mask, mask])
  np_info(result, "Mask RGB", t.elapsed())
  return result


class Time:
  """
  Class for displaying elapsed time.
  """

  def __init__(self):
    self.start = datetime.datetime.now()

  def elapsed_display(self):
    time_elapsed = self.elapsed()
    print("Time elapsed: " + str(time_elapsed))

  def elapsed(self):
    self.end = datetime.datetime.now()
    time_elapsed = self.end - self.start
    return time_elapsed


## ADDED functions

def roi_labelling_order(box_extents):
	"""
	Returns a sort key for boxes, with the descending precedence [vertical_posn, horizontal_posn]
	For use with np.argsort() - named fields
	Leverages Python's tuple sort logic
	- box_extents : [X_min, X_max, Y_min, Y_max]
	"""
	sort_key = np.array(
		(box_extents[0], box_extents[2]),
		dtype=[('vertical', 'i2'),('horizontal', 'i2')]
	)
	return sort_key


def rotate_clockwise_90(np_img):
  """
  90-deg clockwise rotation to make the slide vertical
  Series of 2 operations - transpose row and col dimensions, invert vals in each row
  """
  np_result = np.transpose(np_img, [1, 0, 2])
  np_result = np_result[:,::-1,:]
  return np_result


def rgba_to_rgb(rgba_img, background=(1,1,1), channel_axis=2):
  """
  Use alpha-blending to convert RGBA to RGB image
  Image is contained in an np array
  """
  target_dtype = rgba_img.dtype
  # rgb_img = color.rgba2rgb(rgba_img, background=background, channel_axis=channel_axis)
  rgb_img = color.rgba2rgb(rgba_img, background=background)
  # Convert back to 8-bit int from float if source data was so
  if target_dtype=='uint8':
    rgb_img *= 255
    rgb_img = rgb_img.astype(target_dtype)
  return rgb_img


def rotate_bounding_box_anticlockwise_90(box, img_dims):
  """
  Transforms bounding-box coordinates [X_min, X_max, Y_min, Y_max]
  when image is rotated counter-clockwise by 90-deg about its center
  - img_dims: X and Y dimensions in a tuple - (X,Y) after rotating img by anticlockwise-90
  - box: Box attributes as described above
  """
  x_dim, y_dim = img_dims 
  new_box = [
    box[0],
    box[1],
    x_dim - box[3],
    x_dim - box[2]
  ]
  return new_box


def scale_value_between_dimensions(value, from_dim, to_dim):
  """
  Map a single value on a linear scale from from_dim to to_dim
  Use this to locate pixels on different levels on the same slide
  """
  scale = to_dim / from_dim 
  scaled_val = round(scale * value)
  return int(scaled_val)


def small_to_large_mapping(small_pixel, large_dimensions):
  """
  Map a pixel coordinates on a linear scale onto the WSI
  """
  small_x, small_y = small_pixel
  large_w, large_h = large_dimensions
  large_x = round((large_w / SCALE_FACTOR) / math.floor(large_w / SCALE_FACTOR) * (SCALE_FACTOR * small_x))
  large_y = round((large_h / SCALE_FACTOR) / math.floor(large_h / SCALE_FACTOR) * (SCALE_FACTOR * small_y))
  return large_x, large_y