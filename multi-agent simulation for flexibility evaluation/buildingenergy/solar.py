"""Solar model calculation the radiations on a given unit surface with a specific direction.

Author: stephane.ploix@grenoble-inp.fr
"""
from __future__ import annotations
import configparser
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
from abc import ABC, abstractmethod
from math import asin, atan2, cos, exp, log, sin, sqrt, floor, pi
from buildingenergy.library import SLOPES, DIRECTIONS_CLOCKWISE_SREF
import json
import os.path
import requests
import matplotlib.pyplot as plt
import datetime
import pyexcelerate
from pytz import utc

import random
import logging
import buildingenergy.openweather
from buildingenergy import timemg
import numpy
import enum
import prettytable
import buildingenergy.utils

logging.basicConfig(level=logging.ERROR)


class MOUNT_TYPE(enum.Enum):
    FLAT = 0
    SAW = 1
    ARROW = 2


class RADIATION_TYPE(enum.Enum):
    TOTAL = 0
    DIRECT = 1
    DIFFUSE = 2
    REFLECTED = 3
    NORMAL = 4    


def W2kW(sequence):
    if type(sequence) is list or type(sequence) is tuple:
        return [_ / 1000 for _ in sequence]
    return_sequence = dict()
    if type(sequence) is dict:
        for key in sequence:
            return_sequence[key] = W2kW(sequence[key])
        return return_sequence

# DIRECTION_CLOCKWISE_REF_NORTH: dict[str, int] = {'SOUTH': -180, 'WEST': -90, 'NORTH': 0, 'EAST': 90}
# DIRECTION_CLOCKWISE_SREF: dict[str, int] = {'SOUTH': 0, 'EAST': -90, 'WEST': 90, 'NORTH': 180}
# # 0° means facing the sky, 90° vertical with the collector directed to the defined direction, 180° facing the ground (but not lying on the ground)
# SLOPE: dict[str, int] = {'HORIZONTAL': 0, 'VERTICAL': 90}


config = configparser.ConfigParser()
config.read('setup.ini')
plot_size: tuple[int, int] = (8, 8)


def _encode4file(string: str, number: float = None):  # type: ignore
    if number is not None:
        string += '='+(str(number).replace('.', ','))
    return string.encode("iso-8859-1")


class SkylineRetriever:
    """
    A class to retrieve and store skylines data based on geographic coordinates.

    This class uses a local JSON file to cache skyline data to reduce the need for
    repeated external API calls for the same coordinates.

    Attributes:
        json_database_name (str): The name of the JSON file used for storing data.
        data (dict[str,tuple[tuple[float, float]]): A dictionary to hold the coordinates (azimuth, altitude)"""

    def __init__(self, json_database_name: str = 'skylines.json') -> None:
        """
        Initializes the Skyline Retriever instance.

        Args:
            json_database_name (str): The name of the file where azimuths and altitudes are saved. Defaults to 'skylines.json'.
        """
        self.json_database_name: str = json_database_name
        self.data: dict[str, tuple[tuple[float, float]]] = self._load_data()

    def _load_data(self) -> dict[str, float]:
        """Load skyline data from the JSON file if it exists, else returns an empty dictionary."""
        if os.path.isfile(self.json_database_name):
            with open(self.json_database_name, 'r') as json_file:
                return json.load(json_file)
        else:
            return {}

    def get(self, latitude: float, longitude: float) -> tuple[tuple[float, float]]:
        """
        Retrieves the skyline for the given latitude (degrees north) and longitude (degrees east) coordinates.
        If the skyline data is not cached, it fetches from an external API and stores it.

        Args:
            longitude_deg_east (float): Longitude in degrees east.
            latitude_deg_north (floatS): Latitude in degrees north.

        Returns:
            tuple[tuple[float, float]]: The skyline.
        """
        coordinate = str((latitude, longitude))
        if coordinate not in self.data:
            skyline = self._fetch_skyline_from_api(latitude, longitude)
            self.data[coordinate] = skyline
            self._save_data()
        else:
            skyline: tuple[tuple[float, float]] = self.data[coordinate]
        return skyline

    def _fetch_skyline_from_api(self, latitude_deg_north: float, longitude_deg_east: float) -> float:
        """
        Fetches elevation data from an external API for given coordinates.

        Args:
            longitude_deg_east (float): East degree longitude.
            latitude_deg_north (float): North degree latitude.

        Returns:
            float: The elevation data in meters.
        """
        response = requests.get(
            f'https://re.jrc.ec.europa.eu/api/v5_2/printhorizon?outputformat=json&lat={latitude_deg_north}&lon={longitude_deg_east}',
            headers={'Accept': 'application/json'}
        )
        data = response.json()
        horizon_profile = data['outputs']['horizon_profile']
        skyline = list()
        for hpoint in horizon_profile:
            skyline.append((hpoint['A'], hpoint['H_hor']))
        return tuple(skyline)

    def _save_data(self) -> None:
        """Saves the current elevation data to the JSON file."""
        with open(self.json_database_name, 'w') as json_file:
            json.dump(self.data, json_file)


class Angle:

    def __init__(self, value: float, radian=False) -> None:
        if not radian:
            self._value_rad: float = self._normalize(value / 180 * pi)
        else:
            self._value_rad: float = self._normalize(value)

    @property
    def value_rad(self) -> float:
        return self._normalize(self._value_rad)

    @property
    def value_deg(self) -> float:
        return self._normalize(self._value_rad) / pi * 180

    @staticmethod
    def _normalize(value_rad: float) -> float:
        if -pi <= value_rad <= pi:
            return value_rad
        return (value_rad + pi) % (2 * pi) - pi

    def _diff(self, other_angle: Angle) -> float:
        return (other_angle._value_rad - self._value_rad) * 180/pi

    def __lt__(self, other_angle: Angle) -> bool:
        return self._diff(other_angle) > 0

    def __gt__(self, other_angle: Angle) -> bool:
        return self._diff(other_angle) < 0

    def __eq__(self, other_angle: Angle) -> bool:
        return self._diff(other_angle) == 0

    def __add__(self, other_angle: Angle) -> Angle:
        return Angle(self._value_rad + other_angle._value_rad)

    def __str__(self) -> str:
        return 'Angle:%fdegrees' % self.value_deg


class SolarPosition:
    """Contain self.azimuths_in_deg[i] and altitude angles of the sun for mask processing."""

    def __init__(self, azimuth_in_deg: float, altitude_in_deg: float):
        """Create a solar position with self.azimuths_in_deg[i] (south to west directed) and altitude (south to zenith directed) angles of the sun for mask processing.

        :param azimuth_in_deg: solar angle with the south in degree, counted with trigonometric direction i.e. the self.azimuths_in_deg[i] (0=South, 90=West, 180=North, -90=East)
        :type azimuth_in_deg: float
        :param altitude_in_deg: zenital solar angle with the horizontal in degree i.e. the altitude (0=horizontal, 90=zenith)
        :type altitude_in_deg: float
        """
        if type(azimuth_in_deg) is not Angle:
            self.azimuth_angle: Angle = Angle(azimuth_in_deg)
        else:
            self.azimuth_angle = azimuth_in_deg
        if type(altitude_in_deg) is not Angle:
            self.altitude_angle: Angle = Angle(altitude_in_deg)
        else:
            self.altitude_angle: Angle = altitude_in_deg

    def distance(self, solar_position: SolarPosition) -> float:
        return sqrt((self.azimuth_angle._value_rad-solar_position.azimuth_angle._value_rad)**2 + (self.altitude_angle._value_rad - solar_position.altitude_angle._value_rad)**2)

    def __eq__(self, solar_position: SolarPosition) -> bool:
        return self.altitude_angle == solar_position.altitude_angle and self.azimuth_angle == solar_position.azimuth_angle

    def __lt__(self, solar_position: SolarPosition) -> bool:
        if self.azimuth_angle < solar_position.azimuth_angle:
            return True
        elif self.azimuth_angle == solar_position.azimuth_angle:
            if self.altitude_angle < solar_position.altitude_angle:
                return True
        return False

    def __gt__(self, solar_position: SolarPosition) -> bool:
        if self.azimuth_angle > solar_position.azimuth_angle:
            return True
        elif self.azimuth_angle == solar_position.azimuth_angle:
            if self.altitude_angle > solar_position.altitude_angle:
                return True
        return False

    def __str__(self):
        """Return a description of the solar position.

        :return: string with normalized self.azimuths_in_deg[i] and altitude angles in degree
        :rtype: str
        """
        return '(AZ:%s,AL:%s)' % (self.azimuth_angle.__str__(), self.altitude_angle.__str__())


class SolarModel():
    """Model using the position of the sun and weather data to compute the resulting solar radiations on a directed ground surface."""

    def __init__(self, site_weather_data: buildingenergy.openweather.SiteWeatherData, solar_mask: Mask = None) -> None:
        """Compute all the data that are not specific to a specific collector.

        :param site_weather_data: object generated by an OpenWeatherMapJsonReader, that contains site data like location, altitude,... (see openweather.SiteWeatherData)
        :type site_weather_data: openweather.SiteWeatherData
        """
        self.site_weather_data: buildingenergy.openweather.SiteWeatherData = site_weather_data
        self.time_zone: str = site_weather_data.timezone
        self.latitude_rad: float = site_weather_data.latitude_deg_north / 180 * pi
        self.longitude_rad: float = site_weather_data.longitude_deg_east / 180 * pi
        self.elevation_m: int = site_weather_data.elevation
        self.albedo: float = site_weather_data.albedo
        self.pollution: float = self.site_weather_data.pollution
        if solar_mask is None:
            self.solar_mask = SkyLineMask(*SkylineRetriever().get(site_weather_data.latitude_deg_north, site_weather_data.longitude_deg_east))
        else:
            self.solar_mask = StackedMask(solar_mask, SkyLineMask(*SkylineRetriever().get(site_weather_data.latitude_deg_north, site_weather_data.longitude_deg_east)))
        self.datetimes: list[datetime.datetime] = self.site_weather_data.get('datetime')
        self.temperatures: list[float] = self.site_weather_data.get('temperature')
        self.humidities: list[float] = self.site_weather_data.get('humidity')
        self.nebulosities_percent: list[int] = self.site_weather_data.get('cloudiness')
        self.pressures_Pa: list[int] = [p * 100 for p in self.site_weather_data.get('pressure')]

        self.altitudes_rad = list()
        self.altitudes_deg = list()
        self.azimuths_rad = list()
        self.azimuths_deg = list()
        self.declinations_rad = list()
        self.solar_hour_angles_rad = list()
        self.days_in_year = list()
        self.actual_solartimes_in_secondes = list()
        self.total_solar_irradiances = list()
        self.direct_normal_irradiances = list()
        self.diffuse_horizontal_irradiances = list()
        self.reflected_horizontal_irradiances = list()

        for k, administrative_datetime in enumerate(self.datetimes):
            day_in_year, actual_solartime_in_secondes = self.__solar_time(administrative_datetime)
            self.days_in_year.append(day_in_year)
            self.actual_solartimes_in_secondes.append(actual_solartime_in_secondes)

            altitude_rad, azimuth_rad, declination_rad, solar_hour_angle_rad = self.__solar_angles(administrative_datetime)
            self.altitudes_rad.append(altitude_rad)
            self.azimuths_rad.append(azimuth_rad)
            self.declinations_rad.append(declination_rad)
            self.solar_hour_angles_rad.append(solar_hour_angle_rad)
            self.altitudes_deg.append(altitude_rad/pi*180)
            self.azimuths_deg.append(azimuth_rad/pi*180)
            direct_normal_irradiance, diffuse_horizontal_irradiance, reflected_horizontal_irradiance, total_solar_irradiance = self.__compute_canonic_solar_irradiances(day_in_year, self.elevation_m, altitude_rad, azimuth_rad, self.nebulosities_percent[k], self.temperatures[k], self.humidities[k], self.pressures_Pa[k], self.pollution, self.albedo)
            self.direct_normal_irradiances.append(direct_normal_irradiance)
            self.diffuse_horizontal_irradiances.append(diffuse_horizontal_irradiance)
            self.reflected_horizontal_irradiances.append(reflected_horizontal_irradiance)
            self.total_solar_irradiances.append(total_solar_irradiance)
        self.global_horizontal_irradiances = self.irradiance_W(0, 0)[RADIATION_TYPE.TOTAL]

    def plot_mask(self, name: str = None, **kwargs):
        if name is None:
            self.solar_mask.plot(**kwargs)
        else:
            self.solar_mask.plot(name=name, **kwargs)

    def __solar_time(self, administrative_datetime: datetime.datetime) -> tuple[int, int]:
        """Used by the initializer to calculate the solar time from administrative.

        :param administrative_datetime: the date time
        :type administrative_datetime: datetime.datetime
        :return: elevation_in_rad, azimuth_in_rad, hour_angle_in_rad, latitude_in_rad, declination_in_rad
        :rtype: tuple[float]
        """
        utc_datetime: datetime.datetime = administrative_datetime.astimezone(utc)
        utc_timetuple: time.struct_t.ime = utc_datetime.timetuple()
        day_in_year: int = utc_timetuple.tm_yday % 366
        hour_in_day: int = utc_timetuple.tm_hour
        minute_in_hour: int = utc_timetuple.tm_min
        seconde_in_minute: int = utc_timetuple.tm_sec
        greenwich_time_in_seconds = hour_in_day * 3600 + minute_in_hour * 60 + seconde_in_minute
        local_solar_time_seconds: float = greenwich_time_in_seconds + self.longitude_rad / (2*pi) * 24 * 3600
        time_correction_equation_seconds = 229.2 * (0.000075 + 0.001868*cos(2*pi*day_in_year/365) - 0.032077*sin(2*pi*day_in_year/365) - 0.014615*cos(4*pi*day_in_year/365) - 0.04089*sin(4*pi*day_in_year/365))
        actual_solartime_in_secondes: float = local_solar_time_seconds - time_correction_equation_seconds
        return day_in_year, actual_solartime_in_secondes

    def __solar_angles(self, administrative_datetime: datetime.datetime) -> tuple[float, float]:
        """Used by the initializer to calculate of solar angles.
        :param administrative_datetime: administrative time
        :type administrative_datetime: datetime.datetime
        :return: the altitude, in rad, of the sun in the sky from the horizontal plan to the sun altitude,
        the azimuth, in rad, is the angle with south direction in the horizontal plan,
        the declination, in rad, is the angle between the direction of the sun and equatorial plan of Earth
        the solar hour angle, in rad, is the angle between the Greenwich plan at hour 0 and its current angle
        :rtype: tuple[float]
        """
        day_in_year, solartime_secondes = self.__solar_time(administrative_datetime)
        declination_rad: float = 23.45 / 180 * pi * sin(2*pi * (day_in_year+284)/365.25)
        solar_hour_angle_rad: float = 2*pi * (solartime_secondes / 3600 - 12) / 24
        altitude_rad: float = max(0, asin(sin(declination_rad) * sin(self.latitude_rad) + cos(declination_rad) * cos(self.latitude_rad) * cos(solar_hour_angle_rad)))
        cos_azimuth: float = (cos(declination_rad) * cos(solar_hour_angle_rad) * sin(self.latitude_rad) - sin(declination_rad) * cos(self.latitude_rad)) / cos(altitude_rad)
        sin_azimuth: float = cos(declination_rad) * sin(solar_hour_angle_rad) / cos(altitude_rad)
        azimuth_rad: float = atan2(sin_azimuth, cos_azimuth)  # min(pi/2, max(-pi/2, ))
        return altitude_rad, azimuth_rad, declination_rad, solar_hour_angle_rad

    def __compute_canonic_solar_irradiances(self, day_in_year: int, elevation_m: float, altitude_rad: float, azimuth_rad: float, nebulosity_percent: float, temperature: float, humidity: float, ground_atmospheric_pressure: float, pollution: float, albedo: float) -> dict[str, list[float]]:
        """Used by the initializer to compute the solar power on a 1m2 flat surface.

        :param exposure_in_deg: clockwise angle in degrees between the south and the normal of collecting surface. O means south oriented, 90 means West, -90 East and 180 north oriented
        :type exposure_in_deg: float
        :param slope_in_deg: clockwise angle in degrees between the ground and the collecting surface. 0 means horizontal, directed to the sky, and 90 means vertical directed to the specified direction
        :type slope_in_deg: float
        :return: phi_total, phi_direct_collected, phi_diffuse, phi_reflected
        :rtype: list[float]
        """
        total_solar_irradiance: float = 1367 * (1 + 0.033 * cos(2*pi * (1+day_in_year) / 365.25))
        corrected_total_solar_irradiance: float = (1 - 0.75 * (nebulosity_percent/100) ** 3.4) * total_solar_irradiance
        M0: float = sqrt(1229 + (614*sin(altitude_rad))**2) - 614*sin(altitude_rad)
        Mh: float = (1-0.0065/288*elevation_m)**5.256 * M0
        transmission_coefficient_direct: float = 0.9 ** Mh
        reflected_horizontal_irradiance = max(0, self.albedo * corrected_total_solar_irradiance * (0.271 + 0.706 * transmission_coefficient_direct) * sin(altitude_rad))

        if self.solar_mask.passthrough(SolarPosition(azimuth_rad/pi*180, altitude_rad/pi*180)):
            rayleigh_length: float = 1 / (0.9 * Mh + 9.4)
            partial_steam_pressure: float = 2.165 * (1.098 + temperature/100)**8.02 * humidity/100
            l_Linke: float = 2.4 + 14.6 * pollution + 0.4 * (1 + 2 * pollution) * log(partial_steam_pressure)
            direct_normal_irradiance: float = max(0, transmission_coefficient_direct * corrected_total_solar_irradiance * exp(-Mh * rayleigh_length * l_Linke))
        else:
            direct_normal_irradiance = 0
        # diffuse_horizontal_irradiance: float = corrected_total_solar_irradiance * (0.271 - 0.294 * transmission_coefficient) * sin(altitude_rad)
        transmission_coefficient_diffuse: float = 0.66 ** Mh
        diffuse_horizontal_irradiance = max(0, (1 - transmission_coefficient_diffuse) * sin(altitude_rad) * corrected_total_solar_irradiance)
        return direct_normal_irradiance, diffuse_horizontal_irradiance, reflected_horizontal_irradiance, total_solar_irradiance
    
    @property
    def dni(self):
        return self.direct_normal_irradiances
    
    @property
    def dhi(self):
        return self.diffuse_horizontal_irradiances
    
    @property
    def rhi(self):
        return self.reflected_horizontal_irradiances
    
    @property
    def ghi(self):
        return self.global_horizontal_irradiances
    
    @property
    def tsi(self):
        return self.total_solar_irradiances

    def compute_tilt_solar_irradiances(self, k: int, slope_deg: float, exposure_deg: float, mask: Mask = None) -> tuple[float]:
        """Compute the solar power on a 1m2 flat surface.

        :param exposure_in_deg: clockwise angle in degrees between the south and the normal of collecting surface. O means south oriented, 90 means West, -90 East and 180 north oriented
        :type exposure_in_deg: float
        :param slope_in_deg: clockwise angle in degrees between the ground and the collecting surface. 0 means horizontal, directed to the sky, and 90 means vertical directed to the specified direction
        :type slope_in_deg: float
        :return: phi_total, phi_direct_collected, phi_diffuse, phi_reflected
        :rtype: list[float]
        """
        if not (0 <= slope_deg <= 180):
            raise ValueError("Invalid slope value: %f degrees" % slope_deg)
        slope_rad, exposure_rad = slope_deg / 180 * pi, exposure_deg / 180 * pi

        cos_incidence: float = sin(self.altitudes_rad[k]) * cos(slope_rad) + cos(self.altitudes_rad[k]) * sin(slope_rad) * cos(self.azimuths_rad[k] - exposure_rad)

        if mask is None or mask.passthrough(SolarPosition(self.azimuths_deg[k], self.altitudes_deg[k])):
            direct_tilt_irradiance: float = max(0, cos_incidence * self.direct_normal_irradiances[k])
        else:
            direct_tilt_irradiance = 0
        diffuse_tilt_solar_irradiance = (1 + cos(slope_rad))/2 * self.diffuse_horizontal_irradiances[k]
        reflected_tilt_solar_irradiance = (1 - cos(slope_rad))/2 * self.reflected_horizontal_irradiances[k]
        global_tilt_irradiance: float = direct_tilt_irradiance + diffuse_tilt_solar_irradiance + reflected_tilt_solar_irradiance
        return global_tilt_irradiance, direct_tilt_irradiance, diffuse_tilt_solar_irradiance, reflected_tilt_solar_irradiance

    def irradiance_W(self, exposure_deg: float, slope_deg: float, scale_factor: float = 1, mask: Mask = None):  # -> dict[str, list]:
        tilt_global_irradiances = list()
        tilt_direct_irradiances = list()
        tilt_diffuse_irradiances = list()
        tilt_reflected_irradiances = list()
        
        for k in range(len(self.datetimes)):
            global_tilt_irradiance, direct_tilt_irradiance, diffuse_tilt_irradiance, reflected_tilt_irradiance = self.compute_tilt_solar_irradiances(k, slope_deg, exposure_deg, mask)
            tilt_global_irradiances.append(global_tilt_irradiance * scale_factor)
            tilt_direct_irradiances.append(direct_tilt_irradiance * scale_factor)
            tilt_diffuse_irradiances.append(diffuse_tilt_irradiance * scale_factor)
            tilt_reflected_irradiances.append(reflected_tilt_irradiance * scale_factor)
        return {RADIATION_TYPE.TOTAL: tilt_global_irradiances, RADIATION_TYPE.DIRECT: tilt_direct_irradiances, RADIATION_TYPE.DIFFUSE: tilt_diffuse_irradiances, RADIATION_TYPE.REFLECTED: tilt_reflected_irradiances, RADIATION_TYPE.NORMAL: self.direct_normal_irradiances}

    def try_export(self):
        """Export a TRY weather files for IZUBA Pleiades software. It generates 2 files per full year:
        - site_location + '_' + 'year' + '.INI'
        - site_location + '_' + 'year' + '.TRY'
        The station ID will correspond to the 3 first characters of the site_location in upper case
        """
        site_location: str = self.site_weather_data.location
        location_id: str = site_location[0:3].upper()
        temperatures: list[float] = self.site_weather_data.get('temperature')
        TempSol: float = round(sum(temperatures) / len(temperatures))
        temperature_tenth: list[float] = [10 * t for t in temperatures]
        humidities: list[float] = self.site_weather_data.get('humidity')
        wind_speeds: list[float] = self.site_weather_data.get('wind_speed')
        wind_directions_in_deg: list[float] = self.site_weather_data.get('wind_direction_in_deg')
        irradiations: dict[str, list[float]] = self.irradiance_W(slope_deg=SLOPES.HORIZONTAL.value, exposure_deg=DIRECTIONS_CLOCKWISE_SREF.SOUTH.value)
        irradiations_global_horizontal: list[float] = irradiations[RADIATION_TYPE.TOTAL]
        irradiations_sun_direction: list[float] = irradiations[RADIATION_TYPE.DIRECT]
        ini_file = None
        try_file = None
        for i, dt in enumerate(self.datetimes):
            year, month, day, hour = dt.year, dt.month, dt.day, dt.hour+1
            if month == 1 and day == 1 and hour == 1:
                if ini_file is not None:
                    ini_file.close()
                    try_file.close()
                file_name: str = config['folders']['results'] + site_location + '_' + str(year)
                new_line = '\r\n'
                ini_file = open(file_name + '.ini', "w")
                ini_file.write('[Station]' + new_line)
                ini_file.write('Nom=' + site_location + new_line)
                ini_file.write('Altitude=%i%s' % (int(self.elevation_m), new_line))
                ini_file.write('Lattitude=%s%s' % (en2fr(self.latitude_rad/pi*180), new_line))
                ini_file.write('Longitude=%s%s' % (en2fr(self.longitude_rad/pi*180), new_line))
                ini_file.write('NomFichier=' + site_location + '_' + str(year) + '.try' + new_line)
                ini_file.write('TempSol=%i%s' % (round(TempSol), new_line))
                ini_file.write('TypeFichier=xx' + new_line)
                ini_file.write('Heure solaire=0' + new_line)
                ini_file.write('Meridien=%i%s' % (int(floor(self.latitude_rad/pi*12)), new_line))
                ini_file.write('LectureSeule=1' + new_line)
                ini_file.write('Categorie=OpenMeteo' + new_line)
                ini_file.close()
                try_file = open(file_name + '.try', "bw")
            irrad_coef = 3600 / 10000
            if try_file is not None:
                row: str = f"{location_id}{round(temperature_tenth[i]):4d}{round(irradiations_global_horizontal[i]*irrad_coef):4d}{round(irradiations[RADIATION_TYPE.DIFFUSE][i]*irrad_coef):4d}{round(irradiations_sun_direction[i]*irrad_coef):4d}   E{round(humidities[i]):3d}{round(wind_speeds[i]):3d}{month:2d}{day:2d}{hour:2d}{round(wind_directions_in_deg[i]):4d} 130     E{self.altitudes_deg[i]:6.2f}{self.azimuths_deg[i]+180:7.2f}\r\n"
                row = row.replace('.', ',')
                try_file.write(_encode4file(row))
        try:
            try_file.close()
        except:  # noqa
            pass

    def plot_heliodor(self, year: int, name: str = '', new_figure: bool = True):
        """Plot heliodor at current location.

        :param year: year to be displayed in figure
        :type year: int
        :param name: file_name to be displayed in figure, default to ''
        :type name: str
        """
        stringdates: list[str] = ['21/%i/%i' % (i, year) for i in range(1, 13, 1)]
        legends: list[str] = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.rcParams['font.size'] = 12
        if new_figure:
            _, axis = plt.subplots(figsize=plot_size)
        else:
            plt.gcf()
            axis = plt.gca()
        for day_index, stringdate in enumerate(stringdates):
            altitudes_in_deg, azimuths_in_deg = [], []
            for hour_in_day in range(0, 24, 1):
                for minutes in range(0, 60, 1):
                    altitude_rad, azimuth_rad, declination_rad, solar_hour_angle_rad = self.__solar_angles(timemg.stringdate_to_datetime(stringdate + ' %i:%i:0' % (hour_in_day, minutes)))
                    altitude_deg = altitude_rad/pi*180
                    azimuth_deg = azimuth_rad/pi*180
                    if altitude_deg > 0:
                        altitudes_in_deg.append(altitude_deg)
                        azimuths_in_deg.append(azimuth_deg)
                axis.plot(azimuths_in_deg, altitudes_in_deg)
            i_position = (day_index % 6)*len(altitudes_in_deg)//6 + random.randint(0, len(altitudes_in_deg)//6)
            axis.legend(legends)
            axis.annotate(legends[day_index], (azimuths_in_deg[i_position], altitudes_in_deg[i_position]), )
        axis.set_title('heliodor %s (21th of each month)' % name)
        for hour_in_day in range(0, 24):
            altitudes_deg, azimuths_deg = list(), list()
            for day_in_year in range(0, 365):
                a_datetime = datetime.datetime(int(year), 1, 1, hour=hour_in_day)
                a_datetime += datetime.timedelta(days=int(day_in_year) - 1)
                altitude_rad, azimuth_rad, _, _ = self.__solar_angles(a_datetime)
                if altitude_rad > 0:
                    altitudes_deg.append(altitude_rad/pi*180)
                    azimuths_deg.append(azimuth_rad/pi*180)
            axis.plot(azimuths_deg, altitudes_deg, '.c')
            if len(altitudes_deg) > 0 and max(altitudes_deg) > 0:
                i: int = len(azimuths_deg)//2
                axis.annotate(hour_in_day, (azimuths_deg[i], altitudes_deg[i]))
        axis.axis('tight')
        axis.grid()
        return axis

    def plot_angles(self, with_matplotlib: bool = True):
        """Plot solar angles for the dates corresponding to dates in site_weather_data."""
        if with_matplotlib:
            plt.figure()
            plt.plot(self.datetimes, self.altitudes_deg, self.datetimes, self.azimuths_deg)
            plt.legend(('altitude in deg', 'azimuth in deg'))
            plt.axis('tight')
            plt.grid()
        else:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
            fig.add_trace(go.Scatter(x=self.datetimes, y=self.altitudes_deg, name='sun altitude in °', line_shape='hv'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.datetimes, y=self.azimuths_deg, name='sun azimuth in °', line_shape='hv'), row=2, col=1)
            fig.show()

    def plot_solar_cardinal_irradiations(self, with_matplotlib: bool = True) -> None:
        """Plot total solar irradiation on all cardinal direction and an horizontal one, for the dates corresponding to dates in site_weather_data."""
        solar_gains_wh = dict()
        solar_gains_wh['HORIZONTAL'] = self.irradiance_W(exposure_deg=DIRECTIONS_CLOCKWISE_SREF.SOUTH.value, slope_deg=SLOPES.HORIZONTAL.value)[RADIATION_TYPE.TOTAL]
        for direction in list(DIRECTIONS_CLOCKWISE_SREF):
            solar_gains_wh[direction.name] = self.irradiance_W(exposure_deg=direction.value, slope_deg=SLOPES.VERTICAL.value)[RADIATION_TYPE.TOTAL]
        solar_gains_wh['best'] = self.irradiance_W(exposure_deg=0, slope_deg=35)[RADIATION_TYPE.TOTAL]
        solar_gains_wh['TSI (atmospheric)'] = self.total_solar_irradiances
        solar_gains_wh['DNI (normal)'] = self.direct_normal_irradiances
        solar_gains_wh['DHI (diffuse)'] = self.diffuse_horizontal_irradiances
        solar_gains_wh['RHI (reflected)'] = self.reflected_horizontal_irradiances

        for radiation_name in solar_gains_wh:
            print('energy', radiation_name, ':', sum(solar_gains_wh[radiation_name])/1000, 'kWh/m2')

        if with_matplotlib:
            plt.figure()
            for radiation_name in solar_gains_wh:
                plt.plot(self.datetimes, solar_gains_wh[radiation_name], label=radiation_name)
            plt.legend()
            plt.ylabel('Watt')
            plt.axis('tight')
            plt.grid()
        else:
            fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
            for radiation_name in solar_gains_wh:
                fig.add_trace(go.Scatter(x=self.datetimes, y=solar_gains_wh[radiation_name], name=radiation_name, line_shape='hv'), row=1, col=1)
            fig.show()


def regular_angle_to_decimal_angle_converter(decimals, minutes, seconds):
    """Convert decimals, minutes, seconds to float value.

    :param decimals: number of degrees as an integer
    :type decimals: int
    :param minutes: number of minutes
    :type minutes: int
    :param seconds: number of seconds
    :type seconds: int
    :return: angle in decimal format
    :rtype: float
    """
    return decimals + minutes/60 + seconds/3600


def en2fr(val: float) -> str:
    val = str(val)
    if '.' in val:
        return val.replace('.', ',')
    return str(val)


class SolarSystem:
    """A class used to obtain the solar gains through the windows of a building."""

    def __init__(self, solar_model: SolarModel):
        """Create a set of solar collectors with masks to estimate the global solar gain.

        :param site_weather_data: weather data
        :type site_weather_data: openweather.SiteWeatherData
        :param solar_mask: distant solar mask used for the whole building. None means no global solar masks
        :type solar_mask: ComplexZone
        """
        self.datetimes: list[datetime.datetime] = solar_model.site_weather_data.get('datetime')
        self.stringdates: list[str] = solar_model.site_weather_data.get('stringdate')
        self.temperatures: list[float] = solar_model.site_weather_data.get('temperature')
        self.nebulosities_in_percent: list[float] = solar_model.site_weather_data.get('cloudiness')
        self.humidities: list[float] = solar_model.site_weather_data.get('humidity')
        self.pollution: float = solar_model.site_weather_data.pollution
        self.albedo: float = solar_model.site_weather_data.albedo
        self.solar_model: SolarModel = solar_model
        self.collectors: dict[str, Collector] = dict()

    @property
    def collector_names(self) -> tuple[str]:
        return tuple(self.collectors.keys())
    
    def collector(self, name: str) -> Collector:
        return self.collectors[name]
    
    def mask(self, collector_name: str = None):
        if collector_name is None:
            return self.solar_model.solar_mask
        if collector_name in self.collectors:
            if collector_name in self.collectors:
                return StackedMask(self.solar_model.solar_mask, self.collectors[collector_name].mask)
            else:
                return self.solar_model.solar_mask
        else:
            raise ValueError('unknown collector name: %s' % collector_name)

    def plot_mask(self, collector_name: str = None, **kwargs):
        self.mask(collector_name).plot(**kwargs)

    def clear_collectors(self, collector_name: str = None):
        if collector_name is None:
            self.collectors.clear()
        elif collector_name in self.collector_names:
            del self.collectors[collector_name]

    def solar_gains_W(self, *radiation_types: tuple[RADIATION_TYPE], gather_collectors: bool = False) -> dict[str, dict[RADIATION_TYPE, list[float]]] | dict[RADIATION_TYPE, list[float]]:
        """Return hourly solar gains coming through the collectors and with the type of radiation (RADIATION_TYPE.TOTAL, RADIATION_TYPE.DIRECT, RADIATION_TYPE.DIFFUSE, RADIATION_TYPE.REFLECTED).

        :return: a dictionary with collectors as keys and a dictionary with types of radiations as values
        :rtype: dict[str, dict[str, list[float]]]
        """
        if len(radiation_types) == 0:
            radiation_types = (RADIATION_TYPE.TOTAL,)
        detailed_collector_gains_W = dict()
        for collector_name in self.collectors:
            detailed_collector_gains_W[collector_name] = self.collectors[collector_name].solar_gains_W()

        if len(radiation_types) == 1 and len(self.collectors) == 1:
            return detailed_collector_gains_W[self.collector_names[0]][radiation_types[0]]
        elif len(radiation_types) > 1 and len(self.collectors) == 1:
            return detailed_collector_gains_W[self.collector_names[0]]
        elif len(radiation_types) == 1 and len(self.collectors) > 1:
            if not gather_collectors:
                for collector_name in self.collector_names:
                    detailed_collector_gains_W[collector_name] = detailed_collector_gains_W[collector_name][radiation_types[0]]
                return detailed_collector_gains_W
            else:
                returned_detailed_collector_gains_W = None
                for collector_name in self.collector_names:
                    if returned_detailed_collector_gains_W is None:
                        returned_detailed_collector_gains_W = detailed_collector_gains_W[collector_name][radiation_types[0]]
                    else:
                        for k in range(len(self.datetimes)):
                            returned_detailed_collector_gains_W[k] += detailed_collector_gains_W[collector_name][radiation_types[0]][k]
                return returned_detailed_collector_gains_W
        elif len(radiation_types) > 1 and len(self.collectors) > 1:
            if not gather_collectors:
                return detailed_collector_gains_W
            else:
                returned_detailed_collector_gains_W = dict()
                for collector_name in self.collector_names:
                    for radiation_type in radiation_types:
                        if radiation_type not in returned_detailed_collector_gains_W:
                            returned_detailed_collector_gains_W[radiation_type] = detailed_collector_gains_W[collector_name][radiation_type]
                        else:
                            for k in range(len(self.datetimes)):
                                returned_detailed_collector_gains_W[radiation_type][k] += detailed_collector_gains_W[collector_name][radiation_type][k]
                return returned_detailed_collector_gains_W

    def day_averager(self, vector, average=True):
        """Compute the average or integration hour-time data to get day data.

        :param average: True to compute an average, False for integration
        :type average: bool
        :param vector: vector of values to be down-sampled
        :return: list of floating numbers
        """
        current_day: int = self.datetimes[0].day
        day_integrated_vector = list()
        values = list()
        for k in range(len(self.datetimes)):
            if current_day == self.datetimes[k].day:
                values.append(vector[k])
            else:
                average_value = sum(values)
                if average:
                    average_value = average_value/len(values)
                day_integrated_vector.append(average_value)
                values = list()
            current_day = self.datetimes[k].day
        return day_integrated_vector

    def __len__(self):
        """Return the number of hours in the weather data.

        :return: number of hours in the weather data
        :rtype: int
        """
        return len(self.stringdates)

    def generate_dd_solar_gain_xls(self, file_name='calculations', heat_temperature_reference=18,  cool_temperature_reference=26):
        """Save day degrees and solar gains per window for each day in an xls file.

        :param file_name: file name without extension, default to 'calculation'
        :type file_name: str
        :param temperature_reference: reference temperature for heating day degrees
        :type heat_temperature_reference: float
        :param cool_temperature_reference: reference temperature for cooling day degrees
        :type cool_temperature_reference: float
        """
        print('Heating day degrees')
        stringdate_days, average_temperature_days, min_temperature_days, max_temperature_days, dju_heat_days = self.solar_model.site_weather_data.day_degrees(temperature_reference=heat_temperature_reference, heat=True)
        print('Cooling day degrees')
        _, _, _, _, dju_cool_days = self.solar_model.site_weather_data.day_degrees(temperature_reference=cool_temperature_reference, heat=False)

        data: list[list[str]] = [['date'], ['Tout'], ['Tout_min'], ['Tout_max'], ['dju_heat'], ['dju_cool']]
        data[0].extend(stringdate_days)
        data[1].extend(average_temperature_days)
        data[2].extend(min_temperature_days)
        data[3].extend(max_temperature_days)
        data[4].extend(dju_heat_days)
        data[5].extend(dju_cool_days)

        collectors_solar_gains_in_kWh = self.solar_gains_W()
        i = 6
        for window_name in self.collector_names:
            data.append([window_name+'(Wh)'])
            if len(self.collector_names) > 1:
                data[i].extend(self.day_averager(collectors_solar_gains_in_kWh[window_name], average=False))
            else:
                data[i].extend(self.day_averager(collectors_solar_gains_in_kWh, average=False))
            i += 1

        excel_workbook = pyexcelerate.Workbook()
        excel_workbook.new_sheet(file_name, data=list(map(list, zip(*data))))
        config = configparser.ConfigParser()
        config.read('setup.ini')
        result_folder = config['folders']['results']
        excel_workbook.save(buildingenergy.utils.mkdir_if_not_exist(result_folder + file_name + '.xlsx'))


class Mask(ABC):
    """Abstract class standing for a single zone building. It can be used once specialized."""

    _azimuth_min_max_in_rad: tuple = (-pi, pi)
    _altitude_min_max_in_rad: tuple = (0, pi / 2)

    @abstractmethod
    def passthrough(self, solar_position: SolarPosition) -> bool:
        """Determine whether a solar position of the sun in the sky defined by azimuth and altitude angles is passing through the mask (True) or not (False).

        :param solar_position: solar solar_position in the sky
        :type solar_position: SolarPosition
        :return: True if the solar position is blocked by the mask, False otherwise
        :rtype: bool
        """
        raise NotImplementedError

    def plot(self, name: str = '', axis=None, resolution: int = 40):
        """Plot the mask according to the specified max_plot_resolution and print a description of the zone.

        :param name: file_name of the plot, default to ''
        :return: the zone
        """
        azimuths_in_rad: list[float] = [Mask._azimuth_min_max_in_rad[0] + i * (Mask._azimuth_min_max_in_rad[1] - Mask._azimuth_min_max_in_rad[0]) / (resolution - 1) for i in range(resolution)]
        altitudes_in_rad = [-Mask._altitude_min_max_in_rad[0] + i *
                            (Mask._altitude_min_max_in_rad[1] - Mask._altitude_min_max_in_rad[0]) / (resolution - 1) for i in range(resolution)]
        if axis is None:
            figure, axis = plt.subplots(figsize=plot_size)
            axis.set_xlim((180 / pi * Mask._azimuth_min_max_in_rad[0], 180 / pi * Mask._azimuth_min_max_in_rad[1]))
            axis.set_ylim((-180 / pi * Mask._altitude_min_max_in_rad[0], 180 / pi * Mask._altitude_min_max_in_rad[1]))
        else:
            plt.gcf()
            axis = plt.gca()
        for azimuth_in_rad in azimuths_in_rad:
            for altitude_in_rad in altitudes_in_rad:
                if self.passthrough(SolarPosition(azimuth_in_rad * 180 / pi, altitude_in_rad * 180 / pi)):
                    axis.scatter(180 / pi * azimuth_in_rad, 180 / pi * altitude_in_rad, c='grey', marker='.')
        axis.set_xlabel('Azimuth in degrees (0° = South)')
        axis.set_ylabel('Altitude in degrees')
        axis.set_title(name)


class RectangularMask(Mask):

    def __init__(self, minmax_azimuths_deg: tuple[float, float] = None, minmax_altitudes_deg: tuple[float, float] = None) -> None:
        super().__init__()

        if minmax_azimuths_deg is None:
            self.minmax_azimuth_angles = None
        else:
            self.minmax_azimuth_angles: float = (Angle(minmax_azimuths_deg[0]), Angle(minmax_azimuths_deg[1]))

        if minmax_altitudes_deg is None:
            self.minmax_altitude_angles = None
        else:
            self.minmax_altitude_angles: float | Angle = (Angle(minmax_altitudes_deg[0]), Angle(minmax_altitudes_deg[1]))

    def passthrough(self, solar_position: SolarPosition) -> bool:
        if self.minmax_azimuth_angles is not None:
            if not (self.minmax_azimuth_angles[0] < solar_position.azimuth_angle < self.minmax_azimuth_angles[1]):
                return True
        if self.minmax_altitude_angles is not None:
            if not (self.minmax_altitude_angles[0] < solar_position.altitude_angle < self.minmax_altitude_angles[1]):
                return True
        return False


class EllipsoidalMask(Mask):

    def __init__(self, center_azimuth_altitude_in_deg1: tuple[float | Angle, float | Angle], center_azimuth_altitude_in_deg2: tuple[float | Angle, float | Angle], perimeter_azimuth_altitude_in_deg: tuple[float | Angle, float | Angle]) -> None:
        super().__init__()
        self.center_solar_position1: SolarPosition = SolarPosition(center_azimuth_altitude_in_deg1[0], center_azimuth_altitude_in_deg1[1])
        self.center_solar_position2: SolarPosition = SolarPosition(center_azimuth_altitude_in_deg2[0], center_azimuth_altitude_in_deg2[1])
        self.perimeter_solar_position: SolarPosition = SolarPosition(perimeter_azimuth_altitude_in_deg[0], perimeter_azimuth_altitude_in_deg[1])
        self.length: float | Angle = self._three_positions_length(self.perimeter_solar_position)

    def _three_positions_length(self, solar_position: SolarPosition) -> float | Angle:
        return self.center_solar_position1.distance(self.center_solar_position2) + self.center_solar_position2.distance(solar_position) + solar_position.distance(self.center_solar_position1)

    def passthrough(self, solar_position: SolarPosition) -> bool:
        return self.length > self._three_positions_length(solar_position)


class SkyLineMask(Mask):

    def __init__(self, *azimuths_altitudes_in_deg: tuple[tuple[float | Angle, float | Angle]]) -> None:
        super().__init__()
        azimuths_altitudes_in_deg = list(azimuths_altitudes_in_deg)
        if azimuths_altitudes_in_deg[0][0] != -180:
            azimuths_altitudes_in_deg.insert(0, (-180, 0))
        if azimuths_altitudes_in_deg[-1][0] != 180:
            azimuths_altitudes_in_deg.append((180, azimuths_altitudes_in_deg[0][1]))

        for i in range(1, len(azimuths_altitudes_in_deg)):
            if azimuths_altitudes_in_deg[i-1][0] > azimuths_altitudes_in_deg[i][0]:
                raise ValueError('Skyline is not increasing in azimuth at index %i' % i)
        self.solar_positions: list[SolarPosition] = [SolarPosition(Angle(azimuth_in_deg), Angle(altitude_in_deg)) for azimuth_in_deg, altitude_in_deg in azimuths_altitudes_in_deg]

    def passthrough(self, solar_position: SolarPosition) -> bool:
        index: int = None
        for i in range(1, len(self.solar_positions)):
            index = i - 1
            if self.solar_positions[i-1].azimuth_angle < solar_position.azimuth_angle < self.solar_positions[i].azimuth_angle:
                break
        azimuth_angle0, azimuth_angle1 = self.solar_positions[index].azimuth_angle, self.solar_positions[index+1].azimuth_angle
        altitude_angle0, altitude_angle1 = self.solar_positions[index].altitude_angle, self.solar_positions[index+1].altitude_angle
        if azimuth_angle0 == azimuth_angle1:
            return solar_position.altitude_angle > Angle(max(altitude_angle0.value_deg, altitude_angle1.value_deg))
        altitude_segment: float | Angle = Angle((altitude_angle1.value_deg-altitude_angle0.value_deg)/(azimuth_angle1.value_deg-azimuth_angle0.value_deg) * solar_position.azimuth_angle.value_deg + (altitude_angle0.value_deg*azimuth_angle1.value_deg-altitude_angle1.value_deg*azimuth_angle0.value_deg)/(azimuth_angle1.value_deg-azimuth_angle0.value_deg))
        return solar_position.altitude_angle > altitude_segment


class StackedMask(Mask):

    def __init__(self, *masks: list[Mask]) -> None:
        super().__init__()
        self.masks: list[Mask] = list(masks)

    def add(self, mask: Mask):
        if mask is not None:
            self.masks.append(mask)

    def passthrough(self, solar_position: SolarPosition) -> bool:
        for mask in self.masks:
            if mask is not None and not mask.passthrough(solar_position):
                return False
        return True


class InvertedMask:
    def __init__(self, mask: Mask) -> None:
        super().__init__()
        self.mask: Mask = mask

    def passthrough(self, solar_position: SolarPosition) -> bool:
        if self.mask is None:
            return True
        return not self.mask.passthrough(solar_position)


class Collector:

    def __init__(self, solar_system: SolarSystem | PVsystem, name: str, exposure_deg: float, slope_deg: float, surface_m2: float, solar_factor: float = 1, multiplicity: int = 1, mask: Mask = None, temperature_coefficient: float = 0):
        if type(solar_system) is PVsystem:
            solar_system = solar_system.solar_system
        self.solar_system: SolarSystem = solar_system
        self.solar_model: SolarModel = solar_system.solar_model
        self.datetimes: list[datetime.datetime] = self.solar_model.datetimes
        self.outdoor_temperatures: list[float] = self.solar_model.temperatures
        self.name: str = name
        if name in self.solar_system.collector_names:
            raise ValueError('Solar collector "%s" still exists' % name)
        self.exposure_deg: float = exposure_deg
        self.slope_deg: float = slope_deg
        self.surface_m2: float = surface_m2
        self.solar_factor: float = solar_factor
        self.multiplicity: float = multiplicity
        self.mask: Mask = mask
        self.temperature_coefficient: float = temperature_coefficient
        self.solar_system.collectors[name] = self

    def solar_gains_W(self) -> dict[RADIATION_TYPE, list[float]] | list[float]:
        TaNOCT: float = 46  # in °Celsius
        irradiance_W_per_m2: dict[RADIATION_TYPE, list[float]] = self.solar_model.irradiance_W(self.exposure_deg, self.slope_deg, 1, self.mask)
        _solar_gains_W: dict[RADIATION_TYPE, list[float]] | list[float] = dict()
        for radiation_type in irradiance_W_per_m2:
            _solar_gains_W[radiation_type] = [irradiance_W_per_m2[radiation_type][k] * self.surface_m2 * self.multiplicity * self.solar_factor for k in range(len(self.datetimes))]
            if self.temperature_coefficient != 0:
                for k in range(len(self.datetimes)):
                    if irradiance_W_per_m2[radiation_type][k] != 0:
                        cell_temperature: float = self.outdoor_temperatures[k] + irradiance_W_per_m2[radiation_type][k] / 800 * (TaNOCT - 20)
                        if cell_temperature > 25:
                            _solar_gains_W[radiation_type][k] += - self.temperature_coefficient * max(cell_temperature - 25, 0) * _solar_gains_W[radiation_type][k]
        return _solar_gains_W

    def __str__(self):
        string: str = 'Collector "%s" (EXP:%g°, SLO:%g°) with a surface = %ix%gm2 and a solar factor = %g' % (self.name, self.exposure_deg, self.slope_deg, self.multiplicity, self.surface_m2, self.solar_factor)
        if self.mask is not None:
            string += ', has a specific mask'
        if self.temperature_coefficient != 0:
            string += ' (PV collector with a temperature coefficient = %g%%)' % (100*self.temperature_coefficient)
        return string


class PVsystem:

    def __init__(self, solar_model: SolarModel, peak_power_kW: float = None, array_width_m: float = 1, panel_width_m: float = 1, panel_height_m: float = 1.7, pv_efficiency: float = 0.2, number_of_cell_rows: float = 10, temperature_coefficient: float = 0.0035, min_distance_between_arrays_m: float = 0.4, surface_m2: float = None) -> None:

        self.solar_model: SolarModel = solar_model
        self.solar_system: SolarSystem = None
        self.array_width_m: float = array_width_m if array_width_m >= panel_width_m else panel_width_m
        if peak_power_kW is not None:
            self.peak_power_W: float = peak_power_kW * 1000
        elif surface_m2 is not None:
            self.peak_power_W = compute_peak_power_W(surface_m2, pv_efficiency, panel_width_m, panel_height_m)
        self.panel_surface_m2: float = panel_width_m * panel_height_m
        self.panel_width_m: float = panel_width_m
        self.panel_height_m: float = panel_height_m
        self.pv_efficiency: float = pv_efficiency
        self.number_of_panels: float = round(self.peak_power_W / pv_efficiency / self.panel_surface_m2 / 1000)
        self.number_of_panels_per_array: int = self.array_width_m // panel_width_m
        self.array_surface_in_m2: float = self.number_of_panels_per_array * self.panel_surface_m2
        self.number_of_arrays: int = self.number_of_panels // self.number_of_panels_per_array
        self.number_of_isolated_panels: int = round(self.number_of_panels - self.number_of_panels_per_array * self.number_of_arrays)
        self.surfacePV_m2 = self.panel_surface_m2 * (self.number_of_panels_per_array * self.number_of_arrays + self.number_of_isolated_panels)
        self.mount_type = MOUNT_TYPE.FLAT
        self.min_distance_between_arrays_m = min_distance_between_arrays_m
        self.ground_surface_m2 = self.surfacePV_m2
        self.distance_between_arrays_m = self.panel_height_m
        self.slope_deg = None
        self.exposure_deg = None
        self.n_front_clear_panels = self.number_of_panels
        self.n_front_shadow_panels = 0
        self.n_rear_shadow_panels = 0
        self.n_rear_clear_panels = 0
        self.front_clear_panels_production_Wh: list[float] = None
        self.front_shadow_panels_production: list[list[float]] = None
        self.rear_shadow_panels_production: list[list[float]] = None
        self.rear_clear_panels_production: list[float] = None
        self.outdoor_temperatures: list[float] = self.solar_model.temperatures
        self.infra_panel_array_ratio = min(1, self.distance_between_arrays_m / self.panel_height_m)
        self.temperature_coefficient: float = temperature_coefficient
        self.number_of_cell_rows: float | Angle = number_of_cell_rows
        self.cell_row_surface_in_m2: float = self.panel_surface_m2 / self.number_of_cell_rows
        self.datetimes: list[datetime.datetime] = self.solar_model.site_weather_data.get('datetime')
        self.init = True

    def setup(self, exposure_deg: float = 0, slope_deg: float = 0, distance_between_arrays_m: float = None, mount_type: MOUNT_TYPE = MOUNT_TYPE.FLAT) -> None:
        if distance_between_arrays_m is None:
            distance_between_arrays_m = self.panel_height_m
        elif distance_between_arrays_m < self.min_distance_between_arrays_m:
            distance_between_arrays_m = self.min_distance_between_arrays_m
        self.distance_between_arrays_m = distance_between_arrays_m
        change_setup: bool = (exposure_deg != self.exposure_deg) or (slope_deg != self.slope_deg) or (distance_between_arrays_m != self.distance_between_arrays_m) or (mount_type != self.mount_type)
        if not change_setup and not self.init:
            print('setup stage not needed')
            return
        self.init = False
        self.exposure_deg = exposure_deg
        self.slope_deg = slope_deg

        self.mount_type = mount_type

        self.infra_panel_array_ratio = min(1, self.distance_between_arrays_m / self.panel_height_m)
        self.n_front_clear_panels, self.n_front_shadow_panels, self.n_rear_shadow_panels, self.n_rear_clear_panels = 0, 0, 0, 0
        if mount_type == MOUNT_TYPE.FLAT:
            if (self.number_of_panels <= self.number_of_panels_per_array) or (self.distance_between_arrays_m >= self.panel_height_m):
                self.n_front_clear_panels = self.number_of_panels
            else:
                raise ValueError('Overlapping panels in a flat mount')
        elif mount_type == MOUNT_TYPE.SAW:
            if self.number_of_panels <= self.number_of_panels_per_array:
                self.n_front_clear_panels = self.number_of_panels
            else:
                if self.distance_between_arrays_m < self.min_distance_between_arrays_m:
                    raise ValueError('Distance between arrays must be greater than %icm in a saw mount' % self.min_distance_between_arrays_m*100)
                self.n_front_clear_panels = self.number_of_panels_per_array
                self.n_front_shadow_panels = (self.number_of_arrays - 1) * self.number_of_panels_per_array + self.number_of_isolated_panels
        elif mount_type == MOUNT_TYPE.ARROW:
            if self.number_of_panels <= self.number_of_panels_per_array:
                self.n_front_clear_panels = self.number_of_panels
            else:
                if self.distance_between_arrays_m < self.panel_height_m * cos(slope_deg/180*pi):
                    raise ValueError('Distance between arrays must be greater than %fcm in an arrow mount for a slope = %f° and a panel height = %fm' % (self.distance_between_arrays_m, slope_deg, self.panel_height_m))
                elif self.distance_between_arrays_m < self.min_distance_between_arrays_m:
                    raise ValueError('Distance between arrays must be greater than %i cm in an arrow mount' % self.min_distance_between_arrays_m*100)
                self.n_front_clear_panels = self.number_of_panels_per_array
                n = self.number_of_panels - self.number_of_panels_per_array
                next_is_front = False
                while n > 0:
                    n_candidates = min(n, self.number_of_panels_per_array)
                    n = n - n_candidates
                    if n_candidates > self.number_of_panels_per_array:
                        if next_is_front:
                            self.n_front_shadow_panels += n_candidates
                        else:
                            self.n_rear_shadow_panels += n_candidates
                    else:
                        if next_is_front:
                            self.n_front_shadow_panels += n_candidates
                        else:
                            self.n_rear_clear_panels += n_candidates
                    next_is_front = not next_is_front

        self.ground_surface_m2 = self.n_front_clear_panels * self.panel_width_m * max(self.min_distance_between_arrays_m, self.panel_height_m * cos(self.slope_deg/180*pi) / 2) + (self.number_of_panels - self.n_front_clear_panels) * distance_between_arrays_m * self.panel_width_m

        self.solar_system = SolarSystem(self.solar_model)
        if self.n_front_clear_panels > 0:
            Collector(self, 'clear_front', exposure_deg=self.exposure_deg, slope_deg=self.slope_deg, surface_m2=self.cell_row_surface_in_m2, solar_factor=self.pv_efficiency, multiplicity=self.number_of_cell_rows * self.n_front_clear_panels, temperature_coefficient=self.temperature_coefficient)
        if self.n_rear_clear_panels > 0:
            Collector(self.solar_system, 'clear_rear', exposure_deg=self.exposure_deg, slope_deg=self.slope_deg, surface_m2=self.cell_row_surface_in_m2, solar_factor=self.pv_efficiency, multiplicity=self.number_of_cell_rows * self.n_rear_clear_panels, temperature_coefficient=self.temperature_coefficient)
        if self.n_front_shadow_panels > 0:
            for k in range(self.number_of_cell_rows):
                minimum_sun_visible_altitude_in_deg: float = 180 * atan2(sin(slope_deg / 180 * pi), self.number_of_cell_rows * self.distance_between_arrays_m / (self.number_of_cell_rows - k) / self.panel_height_m - cos(slope_deg / 180 * pi)) / pi
                row_mask = InvertedMask(RectangularMask(minmax_altitudes_deg=(minimum_sun_visible_altitude_in_deg, 180)))
                Collector(self.solar_system, 'shadow_front%i' % k, exposure_deg=self.exposure_deg, slope_deg=self.slope_deg, surface_m2=self.cell_row_surface_in_m2, solar_factor=self.pv_efficiency, multiplicity=self.n_front_shadow_panels, mask=row_mask, temperature_coefficient=self.temperature_coefficient)
        if self.n_rear_shadow_panels > 0:
            for k in range(self.number_of_cell_rows):
                minimum_sun_visible_altitude_in_deg: float = 180 * atan2(sin(slope_deg / 180 * pi), self.number_of_cell_rows * self.distance_between_arrays_m / (self.number_of_cell_rows - k) / self.panel_height_m - cos(slope_deg / 180 * pi)) / pi
                row_mask = InvertedMask(RectangularMask(minmax_altitudes_deg=(minimum_sun_visible_altitude_in_deg, 180)))
                Collector(self.solar_system, 'shadow_rear%i' % k, exposure_deg=self.exposure_deg, slope_deg=self.slope_deg, surface_m2=self.cell_row_surface_in_m2, solar_factor=self.pv_efficiency, multiplicity=self.n_rear_shadow_panels, mask=row_mask, temperature_coefficient=self.temperature_coefficient)

    def electric_power_W(self, exposure_deg: float = 0, slope_deg: float = 0, distance_between_arrays_m: float = None, mount_type: MOUNT_TYPE = MOUNT_TYPE.FLAT, error_message: bool = True) -> list[float]:
        try:
            self.setup(exposure_deg, slope_deg, distance_between_arrays_m, mount_type)
            return self.solar_system.solar_gains_W(RADIATION_TYPE.TOTAL, gather_collectors=True)
        except ValueError as error:
            if error_message:
                print(str(error))
            else:
                print('x', end='')
            return [0 for _ in range(len(self.solar_model.datetimes))]
        # return _prods
    
    def solar_gains_kW(self, exposure_deg: float = 0, slope_deg: float = 0, distance_between_arrays_m: float = None, mount_type: MOUNT_TYPE = MOUNT_TYPE.FLAT, error_message: bool = True) -> list[float]:
        return [gains_W / 1000 for gains_W in self.electric_power_W(exposure_deg, slope_deg, distance_between_arrays_m, mount_type, error_message)]

    def __str__(self) -> str:
        string = 'The PV system is composed of %i arrays (%.2fm width) of %i panels + %i isolated panels for a total PV surface = %.2fm2\n' % (self.number_of_arrays, self.array_width_m, self.number_of_panels_per_array, self.number_of_isolated_panels, self.surfacePV_m2)
        if self.slope_deg is None:
            string += 'A PV panel (EXP: 0, SLO: 0)'
        else:
            string += 'A PV panel (EXP: %.0f°, SLO: %.0f°)' % (self.exposure_deg, self.slope_deg)
        string += ' is W:%.2fm x H:%.2fm (%.2fm2) with an efficiency of %f%% and cells distributed in %i rows\n' % (self.panel_width_m, self.panel_height_m, self.panel_surface_m2, 100 * self.pv_efficiency, self.number_of_cell_rows)
        string += 'The mount type is %s with a peak power of %fkW and the ground surface is  %.2fm2 with a distance between arrays of %.2fm\n' % (self.mount_type.name, self.peak_power_W, self.ground_surface_m2, self.distance_between_arrays_m)
        string += 'There are:\n - %i front facing panels not shadowed\n' % self.n_front_clear_panels
        if self.n_front_shadow_panels > 0:
            string += ' - %i front facing panels shadowed\n' % self.n_front_shadow_panels
        if self.n_rear_shadow_panels > 0:
            string += ' - %i rear facing panels shadowed\n' % self.n_rear_shadow_panels
        if self.n_rear_clear_panels > 0:
            string += ' - %i rear facing panels not shadowed\n' % self.n_rear_clear_panels
        return string
    
    def compute_best_angles(self, distance_between_arrays_m: float = None, mount_type: MOUNT_TYPE = MOUNT_TYPE.FLAT, error_message: bool = False, initial_exposure_deg: float = 0, initial_slope_deg: float = 40) -> dict[str, float]:
        neighborhood: list[tuple[float, float]] = [(-1, 0), (-1, 1), (-1, -1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        taboo = list()
        exposure_slope_in_deg_candidate: tuple[float, float] = (initial_exposure_deg, initial_slope_deg)
        best_exposure_slope_in_deg = tuple(exposure_slope_in_deg_candidate)
        best_total_production_in_Wh = sum(self.electric_power_W(exposure_slope_in_deg_candidate[0], exposure_slope_in_deg_candidate[1], distance_between_arrays_m, mount_type, error_message))
        initial_production_Wh = best_total_production_in_Wh
        taboo.append(exposure_slope_in_deg_candidate)

        improvement = True
        while improvement:
            improvement = False
            for neighbor in neighborhood:
                exposure_slope_in_deg_candidate = (best_exposure_slope_in_deg[0] + neighbor[0], best_exposure_slope_in_deg[1] + neighbor[1])
                exposure_in_deg: float = exposure_slope_in_deg_candidate[0]
                slope_in_deg: float = exposure_slope_in_deg_candidate[1]
                if -90 <= exposure_in_deg <= 90 and 0 <= slope_in_deg <= 90 and exposure_slope_in_deg_candidate not in taboo:
                    taboo.append(exposure_slope_in_deg_candidate)
                    productions_in_Wh = sum(self.electric_power_W(exposure_slope_in_deg_candidate[0], exposure_slope_in_deg_candidate[1], distance_between_arrays_m, mount_type, error_message))
                    if productions_in_Wh > best_total_production_in_Wh:
                        improvement = True
                        best_exposure_slope_in_deg: tuple[float, float] = exposure_slope_in_deg_candidate
                        best_total_production_in_Wh: float = productions_in_Wh
        return {'exposure_deg': best_exposure_slope_in_deg[0], 'slope_deg': best_exposure_slope_in_deg[1], 'best_production_kWh': best_total_production_in_Wh / 1000, 'initial_slope_deg': initial_slope_deg, 'initial_slope_deg': initial_slope_deg, 'initial_production_kWh': initial_production_Wh / 1000, 'mount_type': mount_type.name, 'distance_between_arrays_m': distance_between_arrays_m}

    def print_month_hour_productions(self, exposure_deg: float = 0, slope_deg: float = 0, distance_between_arrays_m: float = None, mount_type: MOUNT_TYPE = MOUNT_TYPE.FLAT):
        productions_kWh = self.solar_gains_kW(exposure_deg, slope_deg, distance_between_arrays_m, mount_type)
        print('total electricity production: %.0fkWh' % sum(productions_kWh))

        month_hour_occurrences: dict[int, dict[int, int]] = [[0 for j in range(24)] for i in range(12)]
        month_hour_productions_in_kWh: dict[int, dict[int, float]] = [[0 for j in range(24)] for i in range(12)]
        table = prettytable.PrettyTable()
        table.set_style(prettytable.MSWORD_FRIENDLY)
        labels: list[str] = ["month#", "cumul"]
        labels.extend(['%i:00' % i for i in range(24)])
        table.field_names = labels
        for i, dt in enumerate(self.datetimes):
            month_hour_occurrences[dt.month-1][dt.hour] = month_hour_occurrences[dt.month-1][dt.hour] + 1
            month_hour_productions_in_kWh[dt.month-1][dt.hour] = month_hour_productions_in_kWh[dt.month-1][dt.hour] + productions_kWh[i]
        for month in range(12):
            number_of_month_occurrences: int = sum(month_hour_occurrences[month-1])
            if number_of_month_occurrences != 0:
                total: str = '%gkWh' % round(sum(month_hour_productions_in_kWh[month-1]))
            else:
                total: str = '0'
            month_row = [month, total]
            for hour in range(24):
                if month_hour_occurrences[month][hour] != 0:
                    month_row.append('%g' % round(1000*month_hour_productions_in_kWh[month][hour] / month_hour_occurrences[month][hour]))
                else:
                    month_row.append('0.')
            table.add_row(month_row)
        print('Following PV productions are in Wh:')
        print(table)


def pv_productions_angles(pv_system: PVsystem, exposures_deg: list[list[float]], slopes_deg: list[float], distance_between_arrays_m: float, mount_type: buildingenergy.solar.MOUNT_TYPE) -> list[float]:
    pv_productions_kWh: list[float] = numpy.zeros((len(slopes_deg), len(exposures_deg)))
    print()
    productions_in_kWh_per_pv_surf: numpy.array = numpy.zeros((len(slopes_deg), len(exposures_deg)))
    counter = 1
    for i, slope_deg in enumerate(slopes_deg):
        for j, exposure_deg in enumerate(exposures_deg):
            print('.', end='')
            counter += 1
            production = sum(pv_system.solar_gains_kW(exposure_deg=exposure_deg, slope_deg=slope_deg, distance_between_arrays_m=distance_between_arrays_m, mount_type=mount_type, error_message=False))
            pv_productions_kWh[i, j] = production
            productions_in_kWh_per_pv_surf[i, j] = production / pv_system.ground_surface_m2
    print()
    return pv_productions_kWh, productions_in_kWh_per_pv_surf


def pv_productions_distances_slopes(pv_system: PVsystem, exposure_in_deg: float, panel_slopes_in_deg: list[float], distances_between_arrays_in_m: list[float], mount_type: buildingenergy.solar.MOUNT_TYPE) -> tuple[list[list[float]], list[list[float]]]:
    productions_in_kWh: numpy.array = numpy.zeros((len(distances_between_arrays_in_m), len(panel_slopes_in_deg)))
    productions_in_kWh_per_pv_surf: numpy.array = numpy.zeros((len(distances_between_arrays_in_m), len(panel_slopes_in_deg)))
    for i, distance_between_arrays_in_m in enumerate(distances_between_arrays_in_m):
        for j, slope_in_deg in enumerate(panel_slopes_in_deg):
            if (mount_type == MOUNT_TYPE.FLAT and pv_system.distance_between_arrays_m >= pv_system.panel_height_m) or (pv_system.distance_between_arrays_m >= pv_system.panel_height_m * cos(slope_in_deg/180*pi)):
                production: float = sum(pv_system.solar_gains_kW(exposure_in_deg, slope_in_deg, distance_between_arrays_in_m, mount_type, error_message=False))
            else:
                production: float = 0
            productions_in_kWh[i, j] = production
            productions_in_kWh_per_pv_surf[i, j] = production / pv_system.ground_surface_m2
    return productions_in_kWh, productions_in_kWh_per_pv_surf


def compute_pv_surface_m2(peak_power_kw: float, pv_efficiency: float = .2, panel_width_m: float = 1, panel_height_m: float = 1.7):
    return round(peak_power_kw / pv_efficiency / (panel_width_m * panel_height_m)) * panel_width_m * panel_height_m


def compute_peak_power_W(pv_surface_m2: float, pv_efficiency: float = .2, panel_width_m: float = 1, panel_height_m: float = 1.7):
    return round(pv_surface_m2 / (panel_width_m * panel_height_m)) * panel_width_m * panel_height_m * pv_efficiency * 1000
