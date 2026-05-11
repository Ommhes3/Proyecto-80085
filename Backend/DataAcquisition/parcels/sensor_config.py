SENSOR_CONFIG = {
    "rice": {
        "temperature": {
            "sensor_type": "DHT22 / SHT31",
            "unit": "°C",
            "sampling_interval_seconds": 60
        },
        "humidity": {
            "sensor_type": "DHT22 / SHT31",
            "unit": "%",
            "sampling_interval_seconds": 60
        },
        "rainfall": {
            "sensor_type": "Pluviómetro",
            "unit": "mm/día",
            "sampling_interval_seconds": 300
        },
        "solar_radiation": {
            "sensor_type": "BH1750",
            "unit": "W/m²",
            "sampling_interval_seconds": 60
        },
        "soil_ph": {
            "sensor_type": "Sensor de pH suelo",
            "unit": "pH",
            "sampling_interval_seconds": 1800
        }
    },

    "sugar_cane": {
        "temperature": {
            "sensor_type": "DHT22 / SHT31",
            "unit": "°C",
            "sampling_interval_seconds": 60
        },
        "humidity": {
            "sensor_type": "DHT22 / SHT31",
            "unit": "%",
            "sampling_interval_seconds": 60
        },
        "solar_radiation": {
            "sensor_type": "BH1750",
            "unit": "W/m²",
            "sampling_interval_seconds": 60
        },
        "soil_ph": {
            "sensor_type": "Sensor de pH suelo",
            "unit": "pH",
            "sampling_interval_seconds": 1800
        },
        "wind_speed": {
            "sensor_type": "Anemómetro",
            "unit": "km/h",
            "sampling_interval_seconds": 30
        }
    }
}