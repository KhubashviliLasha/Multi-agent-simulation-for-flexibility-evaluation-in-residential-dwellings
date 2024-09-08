import random
import ecommunity.simulator


def self_consumption(consumptions: list[float], productions: list[float]) -> float:
    """Compute the ratio of production that is consumed locally

    :param consumption: series of consumption values
    :type consumption
    :param production: series of production values
    :type production
    :return: self-consumption coefficient
    :rtype: float
    """
    locally_consumed_production = 0
    for i in range(len(consumptions)):
        locally_consumed_production += min(productions[i], consumptions[i])
    return locally_consumed_production / sum(productions)


def self_sufficiency(consumptions: list[float], production: list[float]) -> float:
    """Compute the ratio of consumption that is produced locally

    :param consumption: series of consumption values
    :type consumption
    :param production: series of production values
    :type production
    :return: self-production coefficient
    :rtype: float
    """
    locally_fed_consumption = 0
    for i in range(len(consumptions)):
        if consumptions[i] == 0 or production[i] > consumptions[i]:
            locally_fed_consumption += consumptions[i]
        elif production[i] <= consumptions[i]:
            locally_fed_consumption += production[i]
    return locally_fed_consumption / sum(consumptions)


def dependency(consumptions: list[float], productions: list[float]) -> float:
    """Compute the ratio of consumption that is coming from the grid

    :param consumption: series of consumption values
    :type consumption
    :param production: series of production values
    :type production
    :return: self-production coefficient
    :rtype: float
    """
    return 1 - self_sufficiency(consumptions, productions)


def individual_contribution(theoretical_contribution_ratio: float, actual_individual_consumptions: list[float], predicted_individual_consumptions: list[float], actual_community_consumptions: list[float], actual_communities_supplied_powers: list[float]) -> float:
    _contribution = 0
    for i in range(len(actual_individual_consumptions)):
        member_contribution = actual_individual_consumptions[i] - predicted_individual_consumptions[i]
        theoretical_contribution = theoretical_contribution_ratio * (actual_communities_supplied_powers[i] - actual_community_consumptions[i])
        _contribution += abs(member_contribution - theoretical_contribution)
    return _contribution / len(actual_individual_consumptions)


def year_autonomy(consumptions: list[float], productions: list[float]) -> float:
    """Compute the ratio of autonomy wrt the grid

    :param consumption: series of consumption values
    :type consumption
    :param production: series of production values
    :type production
    :return: autonomy coefficient
    :rtype: float
    """
    total_consumption: float = sum(consumptions)
    total_production: float = sum(productions)

    if total_production == 0:
        return 1
    if total_consumption == 0:
        return None
    else:
        return total_production / total_consumption


def renunciation(consumptions_before: list[float], consumptions_after: list[float]) -> float:
    """Quantity of renounced power

    :param consumption_before: expected consumption without incitation
    :type consumption_before
    :param consumption_after: expected consumption without incitation
    :type consumption_after
    :return: coefficient between 0 (no renunciation) to infinity
    :rtype: float
    """
    return max(0, sum(consumptions_before) - sum(consumptions_after)) / sum(consumptions_before)


def effort(consumptions_before: list[float], consumptions_after: list[float], threshold: float = .2) -> float:
    """effort is the ratio of time where the consumption is significantly changed, according to the threshold

    :param consumption_before: expected consumption without incitation
    :type consumption_before
    :param consumption_after: expected consumption without incitation
    :type consumption_after
    :param threshold: variation in the consumption below which it's consider that change is not annoying for members
    :type threshold: float
    :return: coefficient between 0 (no renunciation) to infinity
    :rtype: float
    """
    _effort = 0
    for i in range(len(consumptions_before)):
        _effort += (abs(consumptions_before[i] - consumptions_after[i])) > (threshold * consumptions_after[i])
    return _effort / len(consumptions_before)


def cost(consumptions_in_kWh: list[float], productions_in_kWh: list[float], grid_injection_tariff_kWh: float = .1, grid_drawn_tariff_kWh: float = .2) -> float:
    """energy cost of the provided sequences of consumptions and production values.

    :param consumption: power consumption values
    :type consumption
    :param production: power production values
    :type production
    :param grid_injection_tariff_kWh: tariff for 1kWh of electricity sold to the grid
    :type grid_injection_tariff_kWh: float
    :param grid_drawn_tariff_kWh: tariff for 1kWh of electricity bought on the grid
    :type grid_injection_tariff_kWh: float
    :return: a cost in euros
    :rtype: float
    """
    _cost = 0
    for i in range(len(consumptions_in_kWh)):
        if consumptions_in_kWh[i] > productions_in_kWh[i]:
            _cost += (consumptions_in_kWh[i] - productions_in_kWh[i]) * grid_drawn_tariff_kWh
        else:
            _cost += (consumptions_in_kWh[i] - productions_in_kWh[i]) * grid_injection_tariff_kWh
    return _cost


def NEEG_per_day(consumptions: list[float], productions: list[float]) -> float:
    """net energy exchanged with the grid per day in kWh

    :param consumption: power consumption values
    :type consumption
    :param production: power production values
    :type production
    :return: a quantity of energy per day in kWh
    :rtype: float
    """
    return sum([abs(consumptions[i] - productions[i]) for i in range(len(consumptions))]) * 24 / len(consumptions)


def NEEG_percent(consumptions: list[float], productions: list[float]) -> float:
    """net energy exchanged with the grid per day in kWh

    :param consumption: power consumption values
    :type consumption
    :param production: power production values
    :type production
    :return: a quantity of energy per day in kWh
    :rtype: float
    """
    return sum([abs(consumptions[i] - productions[i]) for i in range(len(consumptions))]) / sum([consumptions[i] for i in range(len(consumptions))])
    return sum([abs(consumptions[i] - productions[i]) for i in range(len(consumptions))]) * 24 / len(consumptions)


def member_contribution(self, member, actual_community_supplied_powers, actual_community_consumed_powers, hour_colors):
    _member_contribution = 0
    _community_contribution = 0
    for h in range(self.number_of_hours):
        coef = int(hour_colors[h].value > 0) - int(hour_colors[h].value < 0)
        _member_contribution += coef * (member.theoretical_share * actual_community_supplied_powers[h] - member._actual_consumptions_kWh[h])
        _community_contribution += coef * (actual_community_supplied_powers[h] - actual_community_consumed_powers[h])
    if _community_contribution == 0:
        return 0
    else:
        return _member_contribution / _community_contribution


def color_statistics(actual_consumptions_kWh: list[float], predicted_consumptions_kWh: list[float], hour_colors: list[ecommunity.simulator.COLOR]) -> tuple[dict[ecommunity.simulator.COLOR, float], dict[ecommunity.simulator.COLOR, float]]:
    color_level_values: dict[ecommunity.simulator.COLOR, int] = dict()
    for i in range(len(hour_colors)):
        if hour_colors[i] not in color_level_values:
            color_level_values[hour_colors[i]] = list()
        color_level_values[hour_colors[i]].append(actual_consumptions_kWh[i] - predicted_consumptions_kWh[i])
    color_level_average_values: dict[ecommunity.simulator.COLOR, float] = dict()
    color_level_ratio: dict[ecommunity.simulator.COLOR, float] = dict()
    for color in color_level_values:
        if len(color_level_values[color]) == 0:
            color_level_average_values[color] = None
            color_level_ratio[color] = 0
        else:
            color_level_average_values[color] = sum(color_level_values[color]) / len(color_level_values[color])
            color_level_ratio[color] = len(color_level_values[color]) / len(hour_colors)
    return color_level_ratio, color_level_average_values


if __name__ == '__main__':
    colors = (ecommunity.simulator.COLOR.BLINKING_RED, ecommunity.simulator.COLOR.SUPER_RED, ecommunity.simulator.COLOR.RED, ecommunity.simulator.COLOR.WHITE, ecommunity.simulator.COLOR.GREEN, ecommunity.simulator.COLOR.SUPER_GREEN, ecommunity.simulator.COLOR.BLINKING_GREEN)
    color_ratios = (.01, .04, .2, .5, .2, .04, .01)
    n = 7*24
    consumed_powers = [random.uniform(0, 2000) for _ in range(n)]
    supplied_powers = [random.uniform(0, 10000) for _ in range(n)]
    color_adapter = ecommunity.dynamic.ColorAdapter(colors, color_ratios, supplied_powers, consumed_powers)
    print('white', color_adapter.get_color(5000, 5000))
    print('red', color_adapter.get_color(0, 5000))
    print('green', color_adapter.get_color(5000, 0))
    print('superred', color_adapter.get_color(0, 8700))
    print('supergreen', color_adapter.get_color(8700, 0))
    print('blinkingred', color_adapter.get_color(0, 9600))
    print('blinkinggreen', color_adapter.get_color(9600, 0))
