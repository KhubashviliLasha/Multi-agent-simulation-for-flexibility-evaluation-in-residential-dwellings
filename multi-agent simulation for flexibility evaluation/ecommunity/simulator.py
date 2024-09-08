"""Ecommunity simulator

Author: stephane.ploix@grenoble-inp.fr
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import prettytable
import datetime
import random
import enum
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import buildingenergy.timemg
import ecommunity.indicators
import buildingenergy.openweather


class COLOR(enum.Enum):
    BLINKING_GREEN = 3
    SUPER_GREEN = 2
    GREEN = 1
    WHITE = 0
    RED = -1
    SUPER_RED = -2
    BLINKING_RED = -3
    
    
def randomize(number_of_values: int, max_amplitude: float, sampling_smooth: int) -> list[float]:
    """Generate a random series of values more or less smooth following a uniform distribution. 
    Each value belongs to [-max_amplitude, max_amplitude]
    :param number_of_values: the length of the random series of values
    :type number_of_values: int
    :param max_amplitude: define the domain for each value i.e. [-max_amplitude, max_amplitude]
    :type max_amplitude: float
    :param sampling_smooth: size of the average filter applied to the series of values
    :type sampling_smooth: int
    """
    variations: list[float] = [max_amplitude*random.uniform(-1, 1) for _ in range(number_of_values)]
    max_variation: float = max([abs(v) for v in variations])
    smooth_variations = list()
    for i in range(0, min(number_of_values,  sampling_smooth)):
        smooth_variations.append(sum(variations[i: i+sampling_smooth])/sampling_smooth)
    for i in range(min(number_of_values,  sampling_smooth), min(number_of_values, number_of_values-sampling_smooth)):
        smooth_variations.append(sum(variations[i - sampling_smooth: i + sampling_smooth])/2/sampling_smooth) if sampling_smooth else smooth_variations.append(variations[i])
    for i in range(max(len(variations) - sampling_smooth, len(smooth_variations)), len(variations)):
        smooth_variations.append(sum(variations[i-sampling_smooth: i])/sampling_smooth)
    max_smooth_variation: float = max([abs(v) for v in smooth_variations])
    return [v / max_smooth_variation * max_variation for v in smooth_variations]


class HourIterator:

    def __init__(self, all_starting_days_hours, last_hours_index: int) -> None:
        self._all_starting_days_hours = all_starting_days_hours
        self._all_ending_days_hours = []
        for i in range(1, len(self._all_starting_days_hours)):
            self._all_ending_days_hours.append(self._all_starting_days_hours[i] - 1)
        self._all_ending_days_hours.append(last_hours_index)
        self._number_of_days = len(all_starting_days_hours)
        self._current_day_index, self._current_hour_index = 0, 0
        self._day_changed = True
        self._last_hours_index = last_hours_index
        self._init = True

    def next(self) -> bool:
        if self._current_hour_index >= self._all_ending_days_hours[self._current_day_index]:
            self._day_changed = True
            if self._current_hour_index == self._last_hours_index:
                self._current_hour_index = None
            else:
                self._current_day_index += 1
                self._current_hour_index += 1
        else:
            if self._init:
                self._init = False
            else:
                self._day_changed = False
                self._current_hour_index += 1
        return self._current_hour_index is not None

    @property
    def day_has_changed(self) -> bool:
        return self._day_changed

    @property
    def number_of_days(self) -> int:
        return self._number_of_days

    @property
    def day_hour_indices(self) -> list[int]:
        return [hour_index for hour_index in range(self._all_starting_days_hours[self._current_day_index], self._all_ending_days_hours[self._current_day_index] + 1)]

    @property
    def current_indices(self) -> tuple[int]:
        return self._current_day_index, self._current_hour_index

    def reset(self) -> None:
        self._current_day_index: int = 0 
        self._current_hour_index: int = 0

        
def presence_model(the_datetime: datetime.datetime):
    if the_datetime.hour < 8 or the_datetime.hour > 22:
        return False
    if the_datetime.weekday() <= 5:
        if 8 <= the_datetime.hour <= 18:
            return random.uniform(0, 1) < .35
        else:
            return random.uniform(0, 1) < .95
    else:
        if 8 <= the_datetime.hour <= 18:
            return random.uniform(0, 1) < .70
        else:
            return random.uniform(0, 1) < .80


class Manager(ABC):

    def __init__(self, pv_system: buildingenergy.solar.PVsystem, number_of_day_interactions: int, number_of_hour_interactions: int, no_alert_threshold: float, randomize_ratio: int, averager_depth_in_hours: int) -> None:
        self.pv_plant: buildingenergy.solar.PVsystem = pv_system
        self.number_of_day_interactions: int = number_of_day_interactions
        self.number_of_hour_interactions: int = number_of_hour_interactions
        self.site_weather_data: buildingenergy.openweather.SiteWeatherData = pv_system.solar_model.site_weather_data 
        self.datetimes: list[datetime.datetime] = pv_system.datetimes
        self._hour_colors: list[COLOR] = [COLOR.WHITE for _ in range(len(self.datetimes))]
        self.no_alert_threshold: float = no_alert_threshold

        self.actual_productions_kWh: list[float] = pv_system.electric_power_W()
        disturbances: list[float] = randomize(len(pv_system.datetimes), randomize_ratio, averager_depth_in_hours)
        self.predicted_productions_kWh: list[float] = [(1 + disturbances[i]) * p for i, p in enumerate(self.actual_productions_kWh)]

        starting_days_indices: list[int] = list()
        for k, a_datetime in enumerate(self.datetimes):
            if a_datetime.hour == 0:
                starting_days_indices.append(k)

        self.hour_iterator = HourIterator(starting_days_indices, len(self.datetimes) - 1)
        self.day_index, self.hour_index = self.hour_iterator.current_indices
        self.synchronized_member_groups: dict[str, list[SynchronizedMember]] = dict()
        self.number_of_hours: int = len(self.datetimes)
    
    def get_hour_colors(self, hour_index: int = None) -> list[COLOR] | COLOR:
        if hour_index is None:
            return self._hour_colors
        return self._hour_colors[hour_index]
    
    def set_hour_colors(self, hour_index: int, color: ecommunity.simulator.COLOR) -> None:
        if self.datetimes[hour_index].hour > 7 and self.datetimes[hour_index].hour < 23:
            self._hour_colors[hour_index] = color
            for group_name in self.synchronized_member_groups:
                for synchronized_member in self.synchronized_member_groups[group_name]:
                    synchronized_member.on_color_change(self.datetimes[hour_index], hour_index, color)
        
    def get_actual_consumption_kWh(self, hour_index: int, group_name: str = None) -> float:
        consumption: float = 0
        if group_name is None:
            for group_name in self.synchronized_member_groups:
                consumption = sum([synchronized_member._actual_consumptions_kWh[hour_index] for synchronized_member in self.synchronized_member_groups[group_name]])
        else:
            consumption = sum([synchronized_member._actual_consumptions_kWh[hour_index] for synchronized_member in self.synchronized_member_groups[group_name]])
        return consumption
    
    def get_actual_consumptions_kWh(self, group_name: str = None) -> list[float]:
        return [self.get_actual_consumption_kWh(k, group_name) for k in range(len(self.datetimes))]
    
    def get_predicted_consumption_kWh(self, hour_index: int, group_name: str = None) -> float:
        consumption: float = 0
        if group_name is None:
            for group_name in self.synchronized_member_groups:
                consumption = sum([synchronized_member.get_predicted_consumptions_kWh()[hour_index] for synchronized_member in self.synchronized_member_groups[group_name]])
        else:
            consumption = sum([synchronized_member.get_predicted_consumptions_kWh()[hour_index] for synchronized_member in self.synchronized_member_groups[group_name]])
        return consumption
    
    def get_predicted_consumptions_kWh(self, group_name: str = None) -> list[float]:
        return [self.get_predicted_consumption_kWh(hour_index=k, group_name=group_name) for k in range(len(self.datetimes))]

    @property
    def number_of_days(self) -> int:
        return self.hour_iterator.number_of_days

    def register_synchronized_member(self, synchronized_member: SynchronizedMember) -> None:
        group_name = synchronized_member.group_name
        if group_name in self.synchronized_member_groups:
            self.synchronized_member_groups[group_name].append(synchronized_member)
        else:
            self.synchronized_member_groups[group_name] = [synchronized_member]
        synchronized_member.manager = self
        
    @property
    def members(self) -> list[SynchronizedMember]:
        _members = list()
        for group_name in self.synchronized_member_groups:
            _members.extend(self.synchronized_member_groups[group_name])
        return _members

    def get_synchronized_members_group(self, group_name: str) -> list[SynchronizedMember]:
        return self.synchronized_member_groups[group_name]

    def get_group_names(self) -> tuple[str]:
        return tuple(self.synchronized_member_groups.keys())

    def time_index_to_datetimes(self, day_hours_indices) -> datetime.datetime | list[datetime.datetime]:
        if type(day_hours_indices) is int:
            return self.datetimes[day_hours_indices]
        else:
            return [self.datetimes[k] for k in day_hours_indices]

    def run(self):
        """Start a simulation."""
        first_day: bool = True
        first_hour: bool = True
        self.predicted_consumptions_kWh: list[float] = self.get_predicted_consumptions_kWh()
        while self.hour_iterator.next():
            _, current_hour_index = self.hour_iterator.current_indices
            if self.hour_iterator.day_has_changed:
                day_hours_indices: list[int] = self.hour_iterator.day_hour_indices
                self.day_round(day_hours_indices, first_day)
                first_day = False
            self.hour_round(current_hour_index, first_hour)
            first_hour = False
        self.finalize()
        
        self.actual_consumptions_kWh: list[float] = self.get_actual_consumptions_kWh()
        for member in self.synchronized_member_groups['ecom']:
            total_contribution: float = sum([member.contribution(self.actual_productions_kWh, self.actual_consumptions_kWh, member._predicted_consumptions_kWh, member._actual_consumptions_kWh) for member in self.synchronized_member_groups['ecom']])
            member.actual_share = member.contribution(self.actual_productions_kWh, self.actual_consumptions_kWh, member._predicted_consumptions_kWh, member._actual_consumptions_kWh) / total_contribution
            print(member)
        self.has_result_set = True
        print(self)
        print()
        self.plot_results()
        
    def day_round(self, day_hour_indices: list[int], init: bool) -> None:
        the_date: datetime.date = self.time_index_to_datetimes(day_hour_indices[0]).date()
        interaction: int = self.number_of_day_interactions
        while interaction > 0:
            interaction -= 1
            self.day_interactions(the_date, day_hour_indices, interaction, init)
            for group_name in self.synchronized_member_groups:
                for synchronized_member in self.synchronized_member_groups[group_name]:
                    synchronized_member.day_interactions(the_date, day_hour_indices, interaction, init)
    
    def hour_round(self, hour_index: int, init: bool):
        the_datetime: datetime.datetime = self.time_index_to_datetimes(hour_index)
        interaction: int = self.number_of_hour_interactions
        while interaction > 0:
            interaction -= 1
            self.hour_interactions(the_datetime, hour_index, interaction, init)
            for group_name in self.synchronized_member_groups:
                for synchronized_member in self.synchronized_member_groups[group_name]:
                    synchronized_member.hour_interactions(the_datetime, hour_index, interaction, init)
            self.hour_interactions(the_datetime, hour_index, interaction, init)
      
    @abstractmethod
    def day_interactions(self, the_date: datetime.date, day_hour_indices: list[int], interaction: int, init: bool):
        raise NotImplementedError
        
    @abstractmethod
    def hour_interactions(self, the_datetime: datetime.datetime, hour_index: int, interaction: int, init: bool):
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        string: str
        if not self.has_result_set:
            for group_name in self.synchronized_member_groups:
                string = '* group: %s\n' % group_name
                for synchronized_system in self.synchronized_member_groups[group_name]:
                    string += ' - ' + synchronized_system.name + '\n'
            if self.hour_index is not None:
                string = '\tcurrent hour is %s\n' % (buildingenergy.timemg.datetime_to_stringdate(self.time_index_to_datetimes(self.hour_index)))
        else:
            string = '# community\n'
            string += 'alert threshold: %.2fkW\n' % self.no_alert_threshold
            string += 'self-consumption: %.2f%% > %.2f%%\n' % (100 * self.predicted_self_consumption, 100 * self.actual_self_consumption)
            string += 'dependency: %.2f%% > %.2f%%\n' % (100 * self.predicted_dependency, 100 * self.actual_dependency)
            string += 'autonomy: %.2f%% > %.2f%%\n' % (100 * self.predicted_autonomy, 100 * self.actual_autonomy)
            string += 'cost: %.2f€ > %.2f€\n' % (self.predicted_cost, self.actual_cost)
            string += 'NEEG per day: %.2fkWh/day > %.2fkWh/day\n' % (self.predicted_NEEG_per_day, self.actual_NEEG_per_day)
            string += 'renunciation: %.2f%%\n' % (100 * self.renunciation)
            string += 'effort: %.2f%%\n' % (100 * self.effort)
            string += 'daily savings per member: %.2f€\n' % self.daily_savings_per_member
            string += 'alerts/day: %.2f\n' % (self.alerts_per_day)
        return string

    @property
    def predicted_self_consumption(self) -> float:
        return ecommunity.indicators.self_consumption(self.get_predicted_consumptions_kWh(), self.predicted_productions_kWh)

    @property
    def actual_self_consumption(self) -> float:
        return ecommunity.indicators.self_consumption(self.get_actual_consumptions_kWh(), self.actual_productions_kWh)

    @property
    def predicted_dependency(self) -> float:
        return ecommunity.indicators.dependency(self.get_predicted_consumptions_kWh(), self.predicted_productions_kWh)

    @property
    def actual_dependency(self) -> float:
        return ecommunity.indicators.dependency(self.get_actual_consumptions_kWh(), self.actual_productions_kWh)

    @property
    def predicted_autonomy(self) -> float:
        return ecommunity.indicators.year_autonomy(self.get_predicted_consumptions_kWh(), self.predicted_productions_kWh)

    @property
    def actual_autonomy(self) -> float:
        return ecommunity.indicators.year_autonomy(self.get_actual_consumptions_kWh(), self.actual_productions_kWh)

    @property
    def predicted_cost(self) -> float:
        return ecommunity.indicators.cost(self.get_predicted_consumptions_kWh(), self.predicted_productions_kWh)

    @property
    def actual_cost(self) -> float:
        return ecommunity.indicators.cost(self.get_actual_consumptions_kWh(), self.actual_productions_kWh)

    @property
    def predicted_NEEG_per_day(self) -> float:
        return ecommunity.indicators.NEEG_per_day(self.get_predicted_consumptions_kWh(), self.predicted_productions_kWh)

    @property
    def actual_NEEG_per_day(self) -> float:
        return ecommunity.indicators.NEEG_per_day(self.get_actual_consumptions_kWh(), self.actual_productions_kWh)

    @property
    def renunciation(self) -> float:
        return ecommunity.indicators.renunciation(self.get_predicted_consumptions_kWh(), self.get_actual_consumptions_kWh())

    @property
    def effort(self) -> float:
        return ecommunity.indicators.effort(self.get_predicted_consumptions_kWh(), self.get_actual_consumptions_kWh())

    @property
    def daily_savings_per_member(self) -> float:
        return (ecommunity.indicators.cost(self.get_predicted_consumptions_kWh(), self.predicted_productions_kWh) - ecommunity.indicators.cost(self.get_actual_consumptions_kWh(), self.actual_productions_kWh)) / self.number_of_days / len(self.members)

    @property
    def alerts_per_day(self) -> float:
        members = self.synchronized_member_groups['ecom']
        total_alerts = sum([members[i].number_of_color_changes() for i in range(len(members))])
        return total_alerts / len(members) / len(self.datetimes) * 24

    def plot_results(self):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=self.datetimes, y=self.predicted_productions_kWh, name='predicted productions in kW'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.datetimes, y=self.actual_productions_kWh, name='actual productions in kW'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.datetimes, y=self.get_predicted_consumptions_kWh(), name='predicted consumption in kW'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.datetimes, y=self.get_actual_consumptions_kWh(), name='actual consumptions in kW'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.datetimes, y=[-1 if c == COLOR.RED else None for c in self._hour_colors], mode='markers', marker_color='red', marker_size=20, name='red'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.datetimes, y=[-2 if c == COLOR.SUPER_RED else None for c in self._hour_colors], mode='markers', marker_color='darkred', marker_size=20, name='super-red'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.datetimes, y=[-3 if c == COLOR.BLINKING_RED else None for c in self._hour_colors], mode='markers', marker_color='darkred', marker_size=20, name='blinking-red'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.datetimes, y=[1 if c == COLOR.GREEN else None for c in self._hour_colors], mode='markers', marker_color='green', marker_size=20, name='green'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.datetimes, y=[2 if c == COLOR.SUPER_GREEN else None for c in self._hour_colors], mode='markers', marker_color='darkgreen', marker_size=20, name='super-green'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.datetimes, y=[3 if c == COLOR.BLINKING_GREEN else None for c in self._hour_colors], mode='markers', marker_color='darkgreen', marker_size=20, name='blinking-green'), row=2, col=1)
        fig.show()


class SynchronizedMember(ABC):

    def __init__(self, member: ecommunity.irise.House, datetimes: list[datetime.datetime], group_name: str = 'ecom', randomize_ratio: int = .2, averager_depth_in_hours: int = 3):
        self.name = str(member.name)
        self.group_name: str = group_name
        self.member: ecommunity.irise.House = member
        self.datetimes: list[datetime.datetime] = datetimes
        self.current_day_hours_indices = None
        self.day_ahead_hours_indices = None
        self._actual_consumptions_kWh: list[float] = member.get_consumptions_kWh()
        random_values: list[float] = randomize(len(datetimes), randomize_ratio, averager_depth_in_hours)
        self._predicted_consumptions_kWh: list[float] = [c * (1 + random_values[i]) for i, c in enumerate(self._actual_consumptions_kWh)]
        self.theoretical_share: float = member.share
        self.actual_share = None
        self.manager = None
        self.presences: list[bool] = [presence_model(dt) for dt in self.datetimes]
        self.color_change_counter = 0
        self.colors: list[ecommunity.simulator.COLOR] = [ecommunity.simulator.COLOR.WHITE for _ in range(len(datetimes))]
        
    def number_of_hours(self) -> int:
        return len(self.datetimes)
    
    def get_predicted_consumptions_kWh(self) -> list[float]:
        return self._predicted_consumptions_kWh
    
    def get_actual_consumptions_kWh(self):
        return self._actual_consumptions_kWh()
    
    def number_of_color_changes(self) -> int:
        return self.color_change_counter 
        
    def set_actual_consumption_kWh(self, hour_index: int, consumption_in_kWh: float):
        self._actual_consumptions_kWh[hour_index] = consumption_in_kWh
        
    @abstractmethod
    def day_interactions(self, the_date: datetime.date, day_hour_indices: list[int], interaction: int, init: bool) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def hour_interactions(self, the_datetime: datetime.datetime, hour_index: int,  interactions: int, sinit: bool) -> None:
        raise NotImplementedError
    
    def on_color_change(self, the_datetime: datetime.datetime, hour_index: int, hour_color: ecommunity.simulator.COLOR):
        if hour_color !=  self.colors[hour_index] and self.presences[hour_index]:
            self.color_change_counter += 1
            self.colors[hour_index] = hour_color
            if hour_color == ecommunity.simulator.COLOR.GREEN:
                self.set_actual_consumption_kWh(hour_index, self._predicted_consumptions_kWh[hour_index] * random.uniform(1, 1.5))
            elif hour_color == ecommunity.simulator.COLOR.RED:
                self.set_actual_consumption_kWh(hour_index, self._predicted_consumptions_kWh[hour_index] * random.uniform(.75, 1))
            elif hour_color == ecommunity.simulator.COLOR.SUPER_GREEN:
                self._actual_consumptions_kWh[hour_index] = self._actual_consumptions_kWh[hour_index] * random.uniform(.75, 2)
            elif hour_color == ecommunity.simulator.COLOR.SUPER_RED:
                self._actual_consumptions_kWh[hour_index] = self._actual_consumptions_kWh[hour_index] * random.uniform(.5, 1)
            elif hour_color == ecommunity.simulator.COLOR.BLINKING_RED:
                self._actual_consumptions_kWh[hour_index] = self._actual_consumptions_kWh[hour_index] * random.uniform(0, 0.5)
            elif hour_color == ecommunity.simulator.COLOR.BLINKING_GREEN:
                self._actual_consumptions_kWh[hour_index] = self._actual_consumptions_kWh[hour_index] * random.uniform(1, 2)

    def contribution(self, actual_communities_supplied_powers, actual_community_consumptions, predicted_consumed_powers, actual_consumed_powers):
        _contribution = 0
        for i in range(len(self.datetimes)):
            member_contribution = actual_consumed_powers[i] - predicted_consumed_powers[i]
            theoretical_contribution = self.theoretical_share * (actual_communities_supplied_powers[i] - actual_community_consumptions[i])
            _contribution += abs(member_contribution - theoretical_contribution)
        return _contribution / len(actual_consumed_powers)

    def __str__(self):
        string = '# member %s\n' % self.name
        _, color_level_average_values = ecommunity.indicators.color_statistics(self._actual_consumptions_kWh, self._predicted_consumptions_kWh, self.manager.get_hour_colors())
        ptable = prettytable.PrettyTable(['color','average power variation'])
        for color in color_level_average_values:
            ptable.add_row([ecommunity.simulator.COLOR(color), color_level_average_values[color]])
        string += ptable.__str__() + '\n'
        string += 'Actual share is %.2f%% for a theoretical share of %.2f%%\n' % (self.actual_share * 100, self.theoretical_share * 100)
        string +=  'Alerts/day: %f' % (self.number_of_color_changes() / len(self.datetimes) * 24)
        return string
