from typing import Dict, Optional

class HandleRegistry:
    def __init__(self):
        self.registry: Dict[int, object] = {}
        self.next_handle: int = 1

    def new_handle(self, obj: object) -> int:
        handle = self.next_handle
        self.registry[handle] = obj
        self.next_handle += 1
        return handle

    def get(self, handle: int) -> Optional[object]:
        return self.registry.get(int(handle))

    def pop(self, handle: int) -> Optional[object]:
        return self.registry.pop(int(handle), None)


class CUDAHandle:
    handle: int

    def __init__(self, registry: HandleRegistry):
        self.handle = registry.new_handle(self)