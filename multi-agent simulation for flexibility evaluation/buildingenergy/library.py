"""
This code has been written by stephane.ploix@grenoble-inp.fr
It is protected under GNU General Public License v3.0

Common physical properties used in buildings collected from https://help.iesve.com/ve2021/ and https://hvac-eng.com/materials-thermal-properties-database/

Properties are extracted from an Microsoft Excel file named: propertiesDB.xslx located in the data folder specified in the setup.ini file.
"""
from __future__ import annotations
import openpyxl
from typing import Callable, Dict
from enum import Enum
from numpy import interp
from math import sqrt
from scipy.constants import Stefan_Boltzmann
import configparser


# class Direction(Enum):
#     """
#     Flag dealing with the direction of a side. It can be vertical (walls > 60°): Direction.VERTICAL, horizontal (standard floor < 60°): Direction.HORIZONTAL and horizontal with ascending flow in case of heating floor: Direction.HORIZONTAL_ASCENDING
#     """
#     VERTICAL = 0  # walls > 60°
#     HORIZONTAL = 1  # implicitly descending most situation with floors and ceilings
#     HORIZONTAL_ASC = 2  # in case of heating floors


class SIDE_TYPES(Enum):
    """
    The types of interface between 2 zones where negative value means horizontal, positive means vertical.
    """
    HEATING_FLOOR = -6
    CANOPY = -5
    VELUX = -4
    GROUND = -3
    FLOOR = -3
    ROOF = -2
    CEILING = -2
    SLAB = -1
    BRIDGE = 0
    WALL = 1
    DOOR = 2
    GLAZING = 3
    CUPBOARD = 4
    JOINERIES = 5


class ZONE_TYPES(Enum):
    """
    Location for a zone (wall or floor)
    """
    SIMULATED = 0
    OUTDOOR = 1
    INFINITE = 2  # temperature is known but indoor surface coefficients apply


class SLOPES(Enum):
    """
    Slope for a side (wall or floor). It can be vertical (sides > 60°): SLOPES.VERTICAL, horizontal (standard floor < 60°): SLOPES.HORIZONTAL and horizontal with ascending flow in case of heating floor: SLOPES.DESCENDING
    """
    HORIZONTAL = 0
    VERTICAL = 90
    HORIZONTAL_HFLOOR = 180
    BEST = 35


class SIDES(Enum):
    OUTDOOR = 0
    INDOOR = 1


class DIRECTIONS_CLOCKWISE_NREF(Enum):
    SOUTH = -180
    WEST = -90
    NORTH = 0
    EAST = 90


class DIRECTIONS_CLOCKWISE_SREF(Enum):
    SOUTH = 0
    WEST = 90
    NORTH = 180
    EAST = -90


class EmissivityModel(Enum):
    SWINBANK = 0
    BERDAHL = 1
    TANG = 2


class SkyTemperatureModel(Enum):
    SWINBANK = 0
    GARG = 1


class Properties:
    """
    Library of material properties loaded from the file propertiesDB.xlsx located in the data folder whose location is given by the config file setup.ini at the project root.
    It also contains common conduction, convection and radiation models for sides (floor, roof or wall) composed of material and air layers.
    Selected material data from the file propertiesDB.xlsx are loaded locally: there made available to use.

    :raises ValueError: Error when 2 materials with the same are loaded into the local database.
    """

    @staticmethod
    def U_indoor_surface_convection(slope: SLOPES, surface: float = 1) -> float:
        """
        Return the RT2012 indoor surface transmission coefficient in W/m2.h for the specified side direction.

        :param direction: direction of the side (see Direction)
        :type direction: SLOPE.'direction'
        :return: thermal power losses through the side in W/
        :rtype: float
        """
        hi: dict[SLOPES, float] = {SLOPES.VERTICAL: 7.69, SLOPES.HORIZONTAL_HFLOOR: 10, SLOPES.HORIZONTAL: 5.88}
        return hi[slope] * surface
    
    @staticmethod
    def U_outdoor_surface_convection(wind_speed_is_m_per_sec: float = 2.4, surface: float = 1) -> float:
        """
        Return the RT2012 outdoor surface transmission coefficient in W/m2.h for the specified side direction.

        :param direction: direction of the side (see Direction)
        :type direction: SLOPE.'direction'
        :return: thermal power losses through the side in W/
        :rtype: float
        """
        return (11.4 + 5.7 * wind_speed_is_m_per_sec) * surface


    @staticmethod
    def clear_sky_emitance(weather_temperature_K, dewpoint_temperature_K, altitude_deg, emissivity_model=EmissivityModel.SWINBANK, sky_temperature_model=SkyTemperatureModel.SWINBANK):
        if emissivity_model == EmissivityModel.SWINBANK:
            if altitude_deg > 0:  # day
                clear_sky_emissivity = 0.727 + 0.0060 * (dewpoint_temperature_K-273.15)
            else:
                clear_sky_emissivity = 0.741 + 0.0062 * (dewpoint_temperature_K-273.15)
        elif emissivity_model == EmissivityModel.BERDAHL:
            clear_sky_emissivity = 0.711 + 0.56*(dewpoint_temperature_K/100) + 0.73*(dewpoint_temperature_K/100)**2
        elif emissivity_model == EmissivityModel.TANG:
            clear_sky_emissivity = 0.754 + 0.0044*dewpoint_temperature_K
        else:
            raise ValueError("Unknown emissivity model")
        
        if sky_temperature_model == SkyTemperatureModel.SWINBANK:
            T_sky = 0.0552 * weather_temperature_K**1.5
        elif sky_temperature_model == SkyTemperatureModel.GARG:
            T_sky = weather_temperature_K - 20
        else:
            raise ValueError("Unknown sky temperature model")
        return clear_sky_emissivity * Stefan_Boltzmann * T_sky**4

    @staticmethod
    def P_sky_surface_exact(weather_temperature_celsius: float, cloudiness_percent: float, altitude_deg: float, dewpoint_temperature_celsius: float, emissivity: float, surface_temperature_celsius: float, surface: float = 1):
        cloud_emissivity = 0.96
        cloudiness: float = cloudiness_percent / 100
        dewpoint_temperature: float = dewpoint_temperature_celsius + 273.15
        weather_temperature: float = weather_temperature_celsius + 273.15
        surface_temperature: float = surface_temperature_celsius + 273.15
        clear_sky_emitance = Properties.clear_sky_emitance(weather_temperature_K=weather_temperature, dewpoint_temperature_K=dewpoint_temperature, altitude_deg=altitude_deg) 
        val = emissivity * Stefan_Boltzmann * surface_temperature**4 + (cloudiness-1) * clear_sky_emitance - cloudiness*cloud_emissivity*Stefan_Boltzmann*(weather_temperature-5)**4
        return -val*surface

    @staticmethod
    def P_sky_surface_linear(weather_temperature_celsius: float, cloudiness_percent: float, altitude_deg: float, dewpoint_temperature_celsius: float, emissivity: float, average_temperature_celsius: float, surface: float = 1):
        cloud_emissivity = 0.96
        cloudiness: float = cloudiness_percent / 100
        dewpoint_temperature: float = dewpoint_temperature_celsius + 273.15
        weather_temperature: float = weather_temperature_celsius + 273.15
        average_temperature: float = average_temperature_celsius + 273.15
        clear_sky_emitance = Properties.clear_sky_emitance(weather_temperature_K=weather_temperature, dewpoint_temperature_K=dewpoint_temperature, altitude_deg=altitude_deg)
        
        val: float = 4*emissivity*Stefan_Boltzmann*average_temperature**3*weather_temperature - 3*emissivity*Stefan_Boltzmann*average_temperature**4 - cloudiness*cloud_emissivity*Stefan_Boltzmann*(weather_temperature-5)**4+(cloudiness-1)*clear_sky_emitance
        
        return -val*surface

    @staticmethod
    def U_surface_radiation(emissivity: float, average_temperature_celsius: float, surface_m2: float = 1) -> float:
        """
        Return the RT2012 indoor surface transmission coefficient in W/m2.K for the specified side direction.

        :param direction: direction of the side (see Direction)
        :type direction: Direction.'direction'
        :return: heat loss through the side in W/
        :rtype: float
        """
        return 4 * Stefan_Boltzmann * emissivity * surface_m2 * (average_temperature_celsius + 273.15) ** 3

    @staticmethod
    def R_surface_radiation(slope: SLOPES, side: SIDES, emissivity: float, wind_speed_is_m_per_sec: float = 2.4, average_temperature_celsius: float = 13, surface: float = 1) -> float:
        if side == SIDES.INDOOR:
            return 1/(Properties.U_surface_radiation(emissivity, average_temperature_celsius, surface) + Properties.U_indoor_surface_convection(slope, surface))
        elif side == SIDES.OUTDOOR:
            return 1/(Properties.U_surface_radiation(emissivity, average_temperature_celsius, surface) + Properties.U_outdoor_surface_convection(wind_speed_is_m_per_sec, surface))

    def __init__(self):
        """
        initialize the BuildingEnergy object
        """
        config = configparser.ConfigParser()
        config.read('./setup.ini')
        self.library = dict()
        self.excel_workbook: openpyxl.Workbook = openpyxl.load_workbook(config['folders']['properties'] + 'propertiesDB.xlsx')
        self.sheet_mapping: dict[str, Callable] = {'thermal': self._get_thermal, 'Uw_glazing': self._get_Uw_glazing, 'glass_transparency': self._get_glass_transparency, 'shading': self._get_shading, 'solar_absorptivity': self._get_solar_absorptivity, 'gap_resistance': self._get_gap_resistance, 'ground_reflectance': self._get_ground_reflectance}

        self.store('plaster', 'thermal', 14)
        self.store('polystyrene', 'thermal', 145)
        self.store('steel', 'thermal', 177)
        self.store('gravels', 'thermal', 203)
        self.store('stone', 'thermal', 204)
        self.store('tile', 'thermal', 236)
        self.store('plywood', 'thermal', 240)
        self.store('air', 'thermal', 259)
        self.store('foam', 'thermal', 260)
        self.store('glass_foam', 'thermal', 261)
        self.store('straw', 'thermal', 261)
        self.store('wood_floor', 'thermal', 264)
        self.store('gypsum', 'thermal', 265)
        self.store('glass', 'thermal', 267)
        self.store('brick', 'thermal', 268)
        self.store('concrete', 'thermal', 269)
        self.store('wood', 'thermal', 277)
        self.store('insulation', 'thermal', 278)
        self.store('usual', 'thermal', 278)
        self.store('PVC', 'thermal', 279)

    def store(self, short_name: str, sheet_name: str, row_number: int):
        """
        Load for usage a physical property related to a sheet name from the from the 'propertiesDB.xlsx' file, and a row number.
        :param short_name: short name used to refer to a material or a component
        :type short_name: str
        :param sheet_name: sheet name in the xlsx file where the property is
        :type sheet_name: str
        :param row_number: row in the sheet of the file containing the property loaded for local usage
        :type row_number: int
        """
        if short_name in self.library:
            if self.library[short_name] != self.sheet_mapping[sheet_name](row_number):
                print(f'Beware: definition of "{short_name}" has changed from {self.get(short_name)} to ', end='')
                del self.library[short_name]
                self.library[short_name] = self.sheet_mapping[sheet_name](row_number)
                print(f'{self.get(short_name)}')
        else:
            self.library[short_name] = self.sheet_mapping[sheet_name](row_number)

    def __str__(self) -> str:
        _str = ''
        for short_name in self.library:
            _str += 'loaded data: %s' % short_name
            _str += str(self.get(short_name)) + '\n'
        return _str

    def get(self, short_name: str) -> Dict[str, float]:
        """
        return the properties loaded locally with the 'store' method, corresponding to the specified sheet of the xlsx sheet, at the specified row

        :param short_name: short name used to refer to a material or a component
        :type short_name: str
        :return: dictionary of values. If the short name is not present in the local database (locally loaded with 'store' method)
        :rtype: Dict[str, float]
        """
        _property_dict = dict(self.library[short_name])
        for property in _property_dict:
            if property == 'emissivity' and _property_dict[property] is None:
                _property_dict[property] = 0.93
        _property_dict['diffusivity'] = _property_dict['conductivity'] / _property_dict['density'] / _property_dict['Cp']
        _property_dict['effusivity'] = sqrt(_property_dict['conductivity'] * _property_dict['density'] * _property_dict['Cp'])
        return _property_dict

    def __contains__(self, short_name: str) -> bool:
        """
        Used for checking whether a short name is in local database

        :param short_name: short name used to refer to a material or a component
        :type short_name: str
        :return: true if the short name is existing
        :rtype: bool
        """
        return short_name in self.library

    def _extract_from_worksheet(self, worksheet_name: str, description_column: str, property_column: str, row_number: str) -> float:
        """
        Get a property value from the xlsx file

        :param worksheet_name: sheet name from the xlsx file
        :type worksheet_name: str
        :param description_column: column where the description of the property is
        :type description_column: str
        :param property_column: column where the value of the property is
        :type property_column: str
        :param row_number: row where the property is
        :type row_number: str
        :return: the referred property value
        :rtype: float
        """
        worksheet = self.excel_workbook[(worksheet_name)]
        # property_description = worksheet["%s%i" % (description_column, row_number)].value
        # property_name = worksheet['%s1' % property_column].value
        property_value = worksheet["%s%i" % (property_column, row_number)].value
        # print('> get property "%s" for "%s"' % (property_name, property_description))
        return property_value

    def _get_thermal(self, row_number: int) -> dict[str, float]:
        """
        get a thermal property (sheet thermal)

        :param row_number: row number in xlsx file
        :type row_number: int
        :return: conductivity, Cp, density and emissivity (0.93 is used in case the value is not present) properties
        :rtype: Dict[str, float]
        """
        properties: dict[str, float] = {}
        properties['conductivity'] = self._extract_from_worksheet('thermal', 'B', 'C', row_number=row_number)
        properties['Cp'] = self._extract_from_worksheet('thermal', 'B', 'D', row_number=row_number)
        properties['density'] = self._extract_from_worksheet('thermal', 'B', 'E', row_number=row_number)
        emissivity: float = self._extract_from_worksheet('thermal', 'B', 'F', row_number=row_number)
        if emissivity == '':
            emissivity = 0.93
        properties['emissivity'] = emissivity
        return properties

    def _get_Uw_glazing(self, row_number: int) -> Dict[str, float]:
        """
        get a heat transmission coefficient for a type of window (sheet Uw_glazing)

        :param row_number: row number in xlsx file
        :type row_number: int
        :return: Uw, Uw_sheltered and Uw_severe properties
        :rtype: Dict[str, float]
        """
        properties: dict[str, float] = {}
        properties['Uw'] = self._extract_from_worksheet('Uw_glazing', 'A', 'C', row_number=row_number)
        properties['Uw_sheltered'] = self._extract_from_worksheet('Uw_glazing', 'A', 'B', row_number=row_number)
        properties['Uw_severe'] = self._extract_from_worksheet('Uw_glazing', 'A', 'D', row_number=row_number)
        return properties

    def _get_glass_transparency(self, row_number: int) -> Dict[str, float]:
        """
        get distribution coefficients between reflection, absorption and transmission for different types of glasses (sheet glass_transparency), and the refractive_index

        :param row_number: row number in xlsx file
        :type row_number: int
        :return: reflection, absorption, transmission and refractive_index
        :rtype: Dict[str, float]
        """
        properties = {}
        properties['reflection'] = self._extract_from_worksheet('glass_transparency', 'A', 'B', row_number=row_number)
        properties['absorption'] = self._extract_from_worksheet('glass_transparency', 'A', 'C', row_number=row_number)
        properties['transmission'] = self._extract_from_worksheet('glass_transparency', 'A', 'D', row_number=row_number)
        properties['refractive_index'] = self._extract_from_worksheet('glass_transparency', 'A', 'E', row_number=row_number)
        return properties

    def _get_shading(self, row_number: int) -> Dict[str, float]:
        """
        get shading coefficient for different building components (sheet shading)

        :param row_number: row number in xlsx file
        :type row_number: int
        :return: shading coefficient
        :rtype: Dict[str, float]
        """
        properties = {}
        properties['shading_coefficient'] = self._extract_from_worksheet('shading', 'A', 'B', row_number=row_number)
        return properties

    def _get_solar_absorptivity(self, row_number: int) -> Dict[str, float]:
        """
        get solar absorptivity coefficient for different surfaces (sheet solar_absorptivity)

        :param row_number: row number in xlsx file
        :type row_number: int
        :return: absorption coefficient
        :rtype: Dict[str, float]
        """
        properties = {}
        properties['absorption'] = self._extract_from_worksheet('solar_absorptivity', 'A', 'B', row_number=row_number)
        return properties

    def _get_gap_resistance(self, row_number: int):
        """
        get air gap convection resistance for different thickness (sheet gap_resistance)

        :param row_number: row number in xlsx file
        :type row_number: int
        :return: thermal resistance Rth
        :rtype: Dict[str, float]
        """
        properties = {}
        properties['Rth'] = self._extract_from_worksheet('gap_resistance', 'B', 'C', row_number=row_number)
        return properties

    def _get_ground_reflectance(self, row_number: int) -> Dict[str, float]:
        """
        get ground reflectance (albedo) for different surfaces (sheet ground_reflectance)

        :param row_number: row number in xlsx file
        :type row_number: int
        :return: albedo
        :rtype: Dict[str, float]
        """
        properties = {}
        properties['albedo'] = self._extract_from_worksheet('ground_reflectance', 'A', 'B', row_number=row_number)
        return properties

    def thermal_air_gap_resistance(self, material1: str, material2: str, gap_in_m: float, slope: SLOPES, average_temperature_celsius: float = 20) -> float:
        """
        Compute the thermal resistance of a 1m2 air gap including convection and radiation phenomena

        :param material1: first material
        :type material1: str
        :param material2: second material
        :type material2: str
        :param gap_in_m: thickness of the air gap in m
        :type gap_in_m: float
        :param direction: direction of the air gap (VERTICAL, HORIZONTAL or HORIZONTAL_ASCENDING)
        :type direction: Direction
        :param average_temperature_celsius: average temperature used for linearization, defaults to 20°C for indoor (12°C for outdoor)
        :type average_temperature_celsius: float, optional
        :return: _description_
        :rtype: the equivalent thermal resistance
        """
        _thicknesses: tuple[float] = (0, 5e-3, 7e-3, 10e-3, 15e-3, 25e-3, 30e-3)
        _thermal_air_gap_resistances = (0, 0.11, 0.13, 0.15, 0.17, 0.18, 0.18)
        emissivity: float = (self.get(material1)['emissivity'] + self.get(material2)['emissivity']) / 2
        if gap_in_m <= _thicknesses[-1]:
            hi = 1 / interp(gap_in_m, _thicknesses, _thermal_air_gap_resistances, left=0, right=_thermal_air_gap_resistances[-1])
        else:
            hi: float = 2 * Properties.U_indoor_surface_convection(slope)

        return 1 / (Properties.U_surface_radiation(emissivity, average_temperature_celsius) + hi)

    def indoor_surface_resistance(self, material: str, slope: SLOPES, average_temperature_celsius: float = 20, surface: float = 1) -> sigma:
        """Indoor convective and radiative transmission coefficient for a vertical surface.

        :param material: name of the material at surface
        :type material: str
        :param direction: direction of the air gap (VERTICAL, HORIZONTAL or HORIZONTAL_ASCENDING)
        :type direction: Direction
        :param average_temperature_celsius: the temperature for which the coefficient is calculated
        :type average_temperature_celsius: float
        :return: the coefficient in W/m2.K
        :rtype: float
        """
        hi: float = Properties.U_indoor_surface_convection(slope)
        hr: float = Properties.U_surface_radiation(self.get(material)['emissivity'], average_temperature_celsius)
        return 1 / (hi + hr) / surface

    def outdoor_surface_resistance(self, material: str, slope: SLOPES, average_temperature_celsius: float = 12, wind_speed_is_m_per_sec: float = 2.4, surface: float = 1) -> float:
        """Return outdoor convective and radiative transmission coefficient.

        :param material: one string among 'usual', 'glass', 'wood', 'plaster', 'concrete' or 'polystyrene'
        :type material: str
        :param direction: not used, here just for homogeneity
        :type direction: Direction
        :param average_temperature_celsius: the temperature for which the coefficient is calculated
        :type average_temperature_celsius: float
        :param wind_speed_is_m_per_sec: wind speed in m/s
        :type wind_speed_is_m_per_sec: wind speed on site
        :return: the coefficient in W/m2.K
        :rtype: float
        """

        return 1 / (11.4 + 5.7 * wind_speed_is_m_per_sec + Properties.U_surface_radiation(self.get(material)['emissivity'], average_temperature_celsius)) / surface

    def conduction_resistance(self, material: str, thickness: float, surface: float = 1):
        """Compute the conductive resistance of an layer depending of its thickness.

        :param thickness: thickness of the layer
        :type thickness: float
        :param material: one string among 'usual', 'glass', 'wood', 'plaster', 'concrete' or 'polystyrene'
        :type material: str
        :return: thermal resistance in K.m2/W
        :rtype: float
        """
        return thickness / self.get(material)['conductivity'] / surface

    def capacitance(self, material: str, thickness: float, surface: float = 1):
        return self.get(material)['Cp']*self.get(material)['density']*surface*thickness


properties = Properties()

if __name__ == '__main__':
    print(properties)
