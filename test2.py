import vkdispatch as vd
import vkdispatch.codegen as vc
from vkdispatch.codegen.abreviations import *

devs = vd.get_devices()

for d in devs:
    print(d.get_info_string(True))