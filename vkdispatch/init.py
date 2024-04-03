import vkdispatch_native

vkdispatch_native.init(False)

device_type_id_to_str_dict = {
    0: "Other",
    1: "Integrated GPU",
    2: "Discrete GPU",
    3: "Virtual GPU",
    4: "CPU",
}

class DeviceInfo:
    def __init__(self, dev_index: int, version_variant: int, version_major: int, version_minor: int,
                version_patch: int, driver_version: int, vendor_id: int, device_id: int,
                device_type: int, device_name: str):
        self.dev_index = dev_index

        self.version_variant = version_variant
        self.version_major = version_major
        self.version_minor = version_minor
        self.version_patch = version_patch

        self.driver_version = driver_version
        self.vendor_id = vendor_id
        self.device_id = device_id

        self.device_type = device_type

        self.device_name = device_name
    
    def __repr__(self) -> str:
        result = f"Device {self.dev_index}: {self.device_name}\n"

        result += f"\tVulkan Version={self.version_major}.{self.version_minor}.{self.version_patch}\n"
        result += f"\tDevice Type={device_type_id_to_str_dict[self.device_type]}\n"
        
        if(self.version_variant != 0):
            result += f"\tVariant={self.version_variant}\n"
        
        #result += f"\tDriver Version={self.driver_version}\n"
        #result += f"\tVendor ID={self.vendor_id}\n"
        #result += f"\tDevice ID={self.device_id}\n"

        return result

def get_devices():
    return [DeviceInfo(ii, *dev_obj) for ii, dev_obj in enumerate(vkdispatch_native.get_devices())]