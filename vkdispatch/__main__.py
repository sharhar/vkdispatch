import vkdispatch as vd


def main():
    for dev in vd.get_devices():
        print(dev)
