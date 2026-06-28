from pydantic import BaseModel


class DeviceInput(BaseModel):
    device_brand: str
    os: str
    screen_size: float
    four_g: str
    five_g: str
    rear_camera_mp: float
    front_camera_mp: float
    internal_memory: float
    ram: float
    battery: float
    weight: float
    release_year: int
    days_used: int
    normalized_new_price: float