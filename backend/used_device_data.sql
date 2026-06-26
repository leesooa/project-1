CREATE DATABASE IF NOT EXISTS used_device_price_db
DEFAULT CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE used_device_price_db;

CREATE TABLE IF NOT EXISTS used_device_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    device_brand VARCHAR(50),
    os VARCHAR(30),
    screen_size FLOAT,
    four_g VARCHAR(10),
    five_g VARCHAR(10),
    rear_camera_mp FLOAT,
    front_camera_mp FLOAT,
    internal_memory FLOAT,
    ram FLOAT,
    battery FLOAT,
    weight FLOAT,
    release_year INT,
    days_used INT,
    normalized_used_price FLOAT,
    normalized_new_price FLOAT
);

SHOW TABLES;