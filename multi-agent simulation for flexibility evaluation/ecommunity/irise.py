"""1h-sampled consumption data extraction for 100 Houses from the IRISE database: it includes all the electric appliances

Author: stephane.ploix@grenoble-inp.fr
"""
from __future__ import annotations

import sqlalchemy
import numpy
import datetime
import configparser
import enum
import logging
import matplotlib.pyplot as plt
import pandas
import scipy.stats
from buildingenergy import timemg
import buildingenergy.openweather
import ecommunity.indicators
from buildingenergy.solar import MOUNT_TYPE 

logging.basicConfig(level=logging.INFO)


class FLEXIBILITY(enum.Enum):
    not_flexible = 0
    CANCELLABLE = -1
    INTERRUPTIBLE = -2
    DELAYABLE = -3
    ANTICIPATORY = -4
    STARTABLE = 1


def filter_outliers(values: list, outlier_sensitivity: float = 10):
    delta_values = [values[k + 1] - values[k] for k in range(len(values) - 1)]
    mean_delta = numpy.mean(delta_values)
    sigma_delta = numpy.std(delta_values)
    for k in range(1, len(values) - 1):
        pdiff = values[k] - values[k - 1]
        fdiff = values[k + 1] - values[k]
        if (abs(pdiff) > mean_delta + outlier_sensitivity * sigma_delta) and (abs(fdiff) > mean_delta + outlier_sensitivity * sigma_delta) and pdiff * fdiff < 0:
            values[k] = (values[k - 1] + values[k + 1]) / 2
    return values


class IRISE:
    """House data extractor from the IRISE SQLite3 database: it contains all the electric consumptions measured for each of the 100 houses involved in the REMODECE/IRISE project during the beginning of 1998 till 2000. Because the weather data are going to be adapted to our context i.e. the data are shifted in order for the day of weeks to be preserved and to be as close. The administrative time changes carried you twice a year are taken into account."""

    def __init__(self, target_datetimes: list[datetime.datetime], zipcode_pattern: str = '381%') -> None:
        """Initialize the house data extractor by getting connected to the database, extracting all the data and closing the connection.

        :param simulator: the simulator that is going to use the IRISE database
        :type from_datestring: scheduler.IRISEsimulator
        """
        self.datetimes: list[datetime.datetime] = target_datetimes
        config = configparser.ConfigParser()
        config.read('setup.ini')
        filename: str = config['folders']['data'] + config['databases']['irise']
        database: sqlalchemy.Engine = sqlalchemy.create_engine('sqlite:///'+filename)
        database_connection: sqlalchemy.Connection = database.connect()
        self.id_houses = dict()
        # select houses matching the zip code
        appliance_query: sqlalchemy.CursorResult[sqlalchemy.Any] = database_connection.execute(sqlalchemy.text("SELECT ID, ZIPcode, Location FROM HOUSE WHERE ZIPcode LIKE '%s';" % zipcode_pattern))
        house_data: numpy.Sequence[sqlalchemy.Any] = appliance_query.cursor.fetchall()
        self.number_of_houses = len(house_data)
        self.house_id_datetime_equivalences = dict()
        for house_id, zip_code, location in house_data:
            house_datetimes = list()
            similarity = list()
            house_weekdays = list()
            appliance_query = database_connection.execute(sqlalchemy.text("SELECT DISTINCT EpochTime FROM Consumption WHERE houseIDREF = '%i' ORDER BY EpochTime ASC;" % house_id))
            house_epochtimes_10min: numpy.Sequence[sqlalchemy.Any] = appliance_query.cursor.fetchall()
            if len(house_epochtimes_10min) != 0:
                print('reading house: %i (%s: %s)' % (house_id, zip_code, location))
                house_datetimes_10min: list[int] = [timemg.epochtimems_to_datetime(house_epochtimes_10min[i][0] * 1000).replace(tzinfo=None) for i in range(len(house_epochtimes_10min))]
                while house_datetimes_10min[0].minute != 0:
                    del house_datetimes_10min[0]
                for i in range(0, len(house_datetimes_10min)-6, 6):
                    house_datetimes.append(house_datetimes_10min[i])
                    # compute the type of day in the week (Monday > 0, Sunday > 6)
                    house_weekdays.append(house_datetimes_10min[i].weekday())
                    # similarity evaluate the hour in the year of a date, giving more importance to the hour in the day than to the day in the year
                    similarity.append(house_datetimes_10min[i].timetuple().tm_yday + 366 * house_datetimes_10min[i].hour)
                house_df = pandas.DataFrame(index=house_datetimes, data={'similarity': similarity, 'weekday': house_weekdays})
                irise_equivalent_datetimes: list[datetime.datetime] = [self._equivalent_datetime(the_datetime, house_df) for the_datetime in self.datetimes]
                self.house_id_datetime_equivalences[house_id] = tuple(irise_equivalent_datetimes)
                datetimes_1h = dict()
                for i, datetime_10min in enumerate(house_datetimes_10min):
                    datetime_1h = datetime_10min.replace(second=0, microsecond=0, minute=0)
                    if datetime_1h not in datetimes_1h:
                        datetimes_1h[datetime_1h] = [i]
                    else:
                        datetimes_1h[datetime_1h].append(i)
                mapping_1h_datetimes_10min_indices = dict()
                for a_datetime in irise_equivalent_datetimes:
                    mapping_1h_datetimes_10min_indices[a_datetime] = datetimes_1h[a_datetime]
                appliance_query = database_connection.execute(sqlalchemy.text("SELECT Value FROM Appliance, Consumption WHERE Appliance.ID = Consumption.ApplianceIDREF AND Appliance.HouseIDREF = Consumption.HouseIDREF AND  Consumption.HouseIDREF='%i' AND Appliance.Name='Site consumption ()' ORDER BY EpochTime ASC;" % house_id))
                house_consumption_10min: list[float] = [row[0] for row in appliance_query.cursor.fetchall()]
                if len(house_consumption_10min) != 0:
                    house_consumption_1h = IRISE._merge_1h(irise_equivalent_datetimes, mapping_1h_datetimes_10min_indices, house_consumption_10min)
                    self.id_houses[house_id] = House(house_id, zip_code, location, 1/self.number_of_houses, target_datetimes, house_consumption_1h)
                    appliance_query = database_connection.execute(sqlalchemy.text("SELECT ID, Name FROM Appliance WHERE HouseIDREF=%i;" % house_id))
                    for appliance_id, name in appliance_query.cursor.fetchall():
                        if name != 'Site consumption ()':
                            appliance_consumptions_query = database_connection.execute(sqlalchemy.text("SELECT  EpochTime, Value FROM Appliance,  Consumption WHERE Appliance.ID = Consumption.ApplianceIDREF AND Appliance.HouseIDREF = Consumption.HouseIDREF AND  Consumption.HouseIDREF='%i' AND Appliance.Name!='Site consumption ()' AND Appliance.ID='%i' ORDER BY EpochTime ASC" % (house_id, appliance_id)))
                            appliance_consumptions_10min: list[float] = [row[1] for row in appliance_consumptions_query.cursor.fetchall()]
                            if len(appliance_consumptions_10min) > 0:
                                appliance_consumptions_1h = IRISE._merge_1h(irise_equivalent_datetimes, mapping_1h_datetimes_10min_indices, appliance_consumptions_10min)
                                self.id_houses[house_id]._add_appliance(appliance_id, name, target_datetimes, appliance_consumptions_1h)
                            else:
                                print('ignoring appliance %s in house: %i (%s: %s): lack of data' % (name, house_id, zip_code, location))
                else:
                    print('removing house: %i (%s: %s): lack of global power meter' % (house_id, zip_code, location))
            else:
                print('rejecting house: %i (%s: %s): lack of data' % (house_id, zip_code, location))

        database_connection.invalidate()
        database.dispose()

    def print_time_equivalences(self) -> None:
        print('Time equivalence for houses:')
        for k in range(len(self.datetimes)):
            print('%s > ' % timemg.datetime_to_stringdate(self.datetimes[k]), end='')
            for house_id in self.id_houses:
                print('%i: %s' % (house_id, timemg.datetime_to_stringdate(self.house_id_datetime_equivalences[house_id][k])), end=',')
            print()

    def _equivalent_datetime(self, the_datetime: datetime.datetime, irise_df: pandas.DataFrame) -> datetime.datetime:
        similarity: int = the_datetime.timetuple().tm_yday + the_datetime.hour * 366
        target_weekday: int = the_datetime.weekday()
        irise_target_weekday_df: pandas.DataFrame = irise_df.loc[irise_df.weekday == target_weekday]
        closest_irise_datetime: datetime.datetime = irise_target_weekday_df.similarity.sub(similarity).abs().idxmin()
        return closest_irise_datetime

    def _merge_1h(irise_equivalent_datetimes, corresponding_indices: dict[datetime.datetime, list[float]], house_values_10min):
        merged_values = list()
        for dt in irise_equivalent_datetimes:
            merged_values.append(sum([house_values_10min[i] for i in corresponding_indices[dt]]))
        return filter_outliers(merged_values)

    def get_houses(self) -> list[House]:
        return self.id_houses.values()

    def get_house(self, id: int) -> House:
        return self.id_houses[id]
    
# before it was summing up all values and it was giving really big values which logicly was not correct now I thinkg it is correct.
    def get_av_hourly_consumptions_kWh(self, *flexibilities: FLEXIBILITY):
        hourly_consumption: list[float] = [0. for h in range(24)]
        consumptions_kWh: list[float] = self.get_consumptions_kWh(*flexibilities)
        days_count: list[int] = [0 for h in range(24)]
        
        for k in range(len(self.datetimes)):
            hour: int = self.datetimes[k].hour
            hourly_consumption[hour] += consumptions_kWh[k]
            days_count[hour] += 1
        
        # Calculate average hourly consumption
        average_hourly_consumption = [consumption / count if count > 0 else 0 
                                    for consumption, count in zip(hourly_consumption, days_count)]
        
        return average_hourly_consumption
#
    def get_hourly_consumptions_kWh(self, *flexibilities: FLEXIBILITY):
        hourly_consumption: list[float] = [0. for h in range(24)]
        consumptions_kWh: list[float] = self.get_consumptions_kWh(*flexibilities)
        for k in range(len(self.datetimes)):
            hour: int = self.datetimes[k].hour
            hourly_consumption[hour] += consumptions_kWh[k]
        return hourly_consumption

    def get_consumptions_Wh(self, *flexibility_types: tuple[FLEXIBILITY]) -> list[int]:
        consumptions = [0 for _ in self.datetimes]
        for house in self.get_houses():
            house_consumptions = house.get_consumptions_Wh(*flexibility_types)
            for k in range(len(self.datetimes)):
                consumptions[k] += house_consumptions[k]
        return consumptions

    def get_consumptions_kWh(self, *flexibility_types: tuple[FLEXIBILITY]) -> list[float]:
        return [c/1000 for c in self.get_consumptions_Wh(*flexibility_types)]

    def __str__(self) -> str:
        string: str = ''
        for house in self.get_houses():
            string += '- %s\n' % str(house)
        return string


class House:
    """A house generated from the IRISE SQLite3 database."""

    def __init__(self, house_id: int, zip_code: str, location: str, share: float, datetimes: list[datetime.datetime], consumptions: list[float]):
        """Initialize a house

        :param database_connection: SQLalchemy database connection
        :type database_connection: SQLalchemy
        :param id: id of the house
        :type id: int
        :param zip_code: zip code of the house
        :type zip_code: str
        :param location: location of the house
        :type location: str
        """
        self.id: int = house_id
        self.zip_code: str = zip_code
        self.name: str = 'house:id=%i@%s-%s' % (house_id, zip_code, location)
        self.location: str = location
        self.id_appliances: dict[int, Appliance] = dict()
        self.share: float = share
        self.consumptions = consumptions
        self.datetimes: list[datetime.datetime] = datetimes

    def _add_appliance(self, appliance_id: int, name: str, datetimes: list[datetime.datetime], consumptions: list[float]):
        self.id_appliances[appliance_id] = Appliance(appliance_id, self.id, name, datetimes, consumptions)

    def get_appliance(self, id_appliance: int) -> Appliance:
        return self.id_appliances[id_appliance]

    def get_appliances(self) -> list[Appliance]:
        return self.id_appliances.values()

    def set_flexible_appliance(self, appliance_id: int, *flexibilities: FLEXIBILITY):
        for flexibility in flexibilities:
            self.id_appliances[appliance_id].possible_flexibilities.add(flexibility)

    def clear_flexibilities(self):
        for id_appliance in self.id_appliances:
            self.id_appliances[id_appliance].flexibilities.clear()

    def get_hourly_consumptions_kWh(self, *flexibilities: FLEXIBILITY):
        hourly_consumption: list[float] = [0. for h in range(24)]
        consumptions_kWh: list[float] = self.get_consumptions_kWh(*flexibilities)
        for k in range(len(self.datetimes)): #self.datetimes:
            hour: int = self.datetimes[k].hour
            hourly_consumption[hour] += consumptions_kWh[k]
        return hourly_consumption
    
    def get_av_hourly_consumptions_kWh(self, *flexibilities: FLEXIBILITY):
        hourly_consumption: list[float] = [0. for h in range(24)]
        consumptions_kWh: list[float] = self.get_consumptions_kWh(*flexibilities)
        days_count: list[int] = [0 for h in range(24)]

        for k in range(len(self.datetimes)):
            hour: int = self.datetimes[k].hour
            hourly_consumption[hour] += consumptions_kWh[k]
            days_count[hour] += 1
            
        # Average Hourly Consumption 
        average_hourly_consumption = [consumption / count if count > 0 else 0 
                                    for consumption, count in zip(hourly_consumption, days_count)]
        
        return average_hourly_consumption  
    

    def split_days(self):
        pass

    def get_consumptions_Wh(self, *flexibility_types: tuple[FLEXIBILITY]):
        """Return the site consumption of the house.

        :return: site consumption (detailed consumptions are also available)
        :rtype: list[float]
        """
        if len(flexibility_types) == 0:
            return self.consumptions
        else:
            consumptions = [0 for _ in self.datetimes]
            for id in self.id_appliances:
                appliance_flexible_consumptions: list[float] = self.get_appliance(id).get_consumptions_Wh(*flexibility_types)
                for k in range(len(self.datetimes)):
                    consumptions[k] = consumptions[k] + appliance_flexible_consumptions[k]
            return consumptions

    def get_consumptions_kWh(self, *flexibility_types: tuple[FLEXIBILITY]):
        return [c/1000 for c in self.get_consumptions_Wh(*flexibility_types)]
    
    def get_10min_consumptions_kWh(self):
        """Return the site consumption in kWh for each 10-minute interval."""
        ten_min_consumptions_kWh = [consumption / 1000 for consumption in self.consumptions]  # Convert Wh to kWh
        return ten_min_consumptions_kWh

    def __str__(self) -> str:
        """Return a description of the house

        :return: text description
        :rtype: str
        """
        string: str = 'House %s located at %s %s with appliances:\n' % (self.id, self.zip_code, self.location)
        for appliance_id in self.id_appliances:
            string += '\t%i: %s\n' % (appliance_id, str(self.get_appliance(appliance_id)))
        return string

class Appliance:
    """An appliance related to a house."""

    def __init__(self, appliance_id: int, house_id: int, name: str, datetimes: list[datetime.datetime], consumptions: list[float]):
        """Initialize the appliance."""
        self.name: str = name.split(" (")[0]
        self.name = self.name.replace(" ", "_")
        self.name = self.name.lower()
        self.appliance_id: int = appliance_id
        self.house_id: int = house_id
        self.possible_flexibilities = set()
        self.datetimes = datetimes
        self.consumptions = consumptions

    def is_flexible(self, *flexibility_types: tuple[FLEXIBILITY]) -> bool:
        if FLEXIBILITY.not_flexible in self.possible_flexibilities:
            return False
        if len(flexibility_types) == 0:
            return len(self.possible_flexibilities) > 0 and FLEXIBILITY.not_flexible not in self.possible_flexibilities
        else:
            for f in flexibility_types:
                if f in self.possible_flexibilities:
                    return True
            return False

    @property
    def flexibilities(self) -> tuple[FLEXIBILITY]:
        return tuple(self.possible_flexibilities)

    def get_consumptions_Wh(self, *flexibility_types: FLEXIBILITY) -> list[float]:
        if not flexibility_types:
            if self.is_flexible():
                return self.consumptions
            else:
                return self.consumptions #[0 for _ in range(len(self.consumptions))]
        elif FLEXIBILITY.not_flexible in flexibility_types:
            if not self.is_flexible():
                return self.consumptions
            else:
                return [0 for _ in range(len(self.consumptions))]
        elif self.is_flexible(*flexibility_types):
            return self.consumptions
        else:
            return [0 for _ in range(len(self.consumptions))]

    def get_consumptions_kWh(self, *flexibility_types: tuple[FLEXIBILITY]):
        return [c/1000 for c in self.get_consumptions_Wh(*flexibility_types)]

    def __str__(self) -> str:
        """Return a description of the appliance."""
        string = 'Appliance: ' + self.name + ' '
        if self.is_flexible():
            for f in self.flexibilities:
                string += f.name + ","
        else:
            string += "not_flexible"
        return string

# class Appliance:
#     """An appliance related to a house."""

#     def __init__(self, appliance_id: int, house_id: int, name: str, datetimes: list[datetime.datetime], consumptions: list[float]):
#         """Initialize the appliance.

#         :param database_connection: sqlite3 database connection
#         :type database_connection: SQLalchemy
#         :param id: identifier of the appliance in a house
#         :type id: int
#         :param house_id: identifier of the house containing the appliance
#         :type house_id: int
#         :param name: name of the appliance
#         :type name: str
#         """
#         self.name: str = name.split(" (")[0]
#         self.name = self.name.replace(" ", "_")
#         self.name = self.name.lower()
#         self.appliance_id: int = appliance_id
#         self.house_id: int = house_id
#         self.possible_flexibilities = set()
#         self.datetimes = datetimes
#         self.consumptions = consumptions

#     def is_flexible(self, *flexibility_types: tuple[FLEXIBILITY]) -> bool:
#         if len(flexibility_types) == 0:
#             return len(self.possible_flexibilities) > 0
#         else:
#             for f in flexibility_types:
#                 if f in self.possible_flexibilities:
#                     return True
#             return False

#     @property
#     def flexibilities(self) -> tuple[FLEXIBILITY]:
#         return tuple(self.possible_flexibilities)

#     def get_consumptions_Wh(self, *flexibility_types: FLEXIBILITY) -> list[float]:
#         if self.is_flexible(*flexibility_types):
#             return self.consumptions
#         if self.is_flexible
#         else:
#             return [0 for _ in range(len(self.consumptions))]

#     def get_consumptions_kWh(self, *flexibility_types: tuple[FLEXIBILITY]):
#         return [c/1000 for c in self.get_consumptions_Wh(*flexibility_types)]

#     def __str__(self) -> str:
#         """Return a description of the appliance.

#         :return: a text description
#         :rtype: str
#         """
#         string = 'Appliance: ' + self.name + ' '
#         if self.is_flexible():
#             for f in self.flexibilities:
#                 string += f.name + ","
#         return string 