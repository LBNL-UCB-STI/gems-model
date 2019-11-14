#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import copy

from utils.Network import Network
from utils.supply import DemandCharacteristics, BusDemandCharacteristics, TravelDemand, ModeParams, BusParams
import utils.supply as supply


class ModeCharacteristics:
    def __init__(self, mode_name: str, params: supply.ModeParams, demand: float = 0.0):
        self.mode_name = mode_name
        self.params = params
        self.demand_characteristics = getDefaultDemandCharacteristics(mode_name)
        self.supply_characteristics = getDefaultSupplyCharacteristics()
        self.demand = demand

    def __str__(self):
        return self.mode_name.upper() + ': ' + str(self.demand_characteristics) + ', ' + str(
            self.supply_characteristics)

    def setSupplyCharacteristics(self, supply_characteristics: supply.SupplyCharacteristics):
        self.supply_characteristics = supply_characteristics

    def setDemandCharacteristics(self, demand_characteristics: supply.DemandCharacteristics):
        self.demand_characteristics = demand_characteristics

    def getSpeed(self):
        return self.demand_characteristics.speed

    def getFlow(self):
        return self.demand_characteristics.passenger_flow


class CollectedModeCharacteristics:
    def __init__(self):
        self._data = dict()

    def __setitem__(self, mode_name: str, mode_info: ModeCharacteristics):
        self._data[mode_name] = mode_info

    def __getitem__(self, mode_name) -> ModeCharacteristics:
        return self._data[mode_name]

    def getModes(self):
        return list(self._data.keys())

    def __str__(self):
        return str([str(self._data[key]) for key in self._data])

    #    def setModeDemand(self, mode: str, new_demand: float):
    #        self._data[mode].demand = new_demand

    def addModeDemand(self, mode: str, demand: float):
        self._data[mode].demand += demand


#    def getModeSpeed(self, mode: str) -> float:
#        return self._data[mode].demand_characteristics.passenger_flow

class Microtype:
    def __init__(self, network_params: Network, mode_characteristics: CollectedModeCharacteristics):
        self.modes = mode_characteristics.getModes()
        self.network_params = network_params
        self._baseSpeed = network_params.getBaseSpeed()
        self._mode_characteristics = mode_characteristics
        self._travel_demand = TravelDemand(self.modes)
        self.updateSupplyCharacteristics()
        self.updateDemandCharacteristics()

    def getModeSpeed(self, mode) -> float:
        return self.getModeCharacteristics(mode).demand_characteristics.getSpeed()

    def getBaseSpeed(self) -> float:
        return self._baseSpeed

    def getModeFlow(self, mode) -> float:
        return self._travel_demand.getRateOfPMT(mode)

    def getModeDemandForPMT(self, mode):
        return self._travel_demand.getRateOfPMT(mode)

    def addModeDemand(self, mode, demand):
        self._mode_characteristics.addModeDemand(mode, demand)

    def setModeDemand(self, mode, demand, trip_distance):
        self._travel_demand.setSingleDemand(mode, demand, trip_distance)

    def getModeCharacteristics(self, mode: str) -> ModeCharacteristics:
        return self._mode_characteristics[mode]

    def getStartAndEndRate(self, mode: str) -> (float, float):
        return self._travel_demand.getStartRate(mode), self._travel_demand.getStartRate(mode)

    def getModeMeanDistance(self, mode: str):
        return self._travel_demand.getAverageDistance(mode)

    def setModeSupplyCharacteristics(self, mode: str, supply_characteristics: supply.SupplyCharacteristics):
        self.getModeCharacteristics(mode).setSupplyCharacteristics(supply_characteristics)

    def setModeDemandCharacteristics(self, mode: str, demand_characteristics: supply.DemandCharacteristics):
        self.getModeCharacteristics(mode).setDemandCharacteristics(demand_characteristics)

    def getModeDensity(self, mode):
        mc = self.getModeCharacteristics(mode)
        fixed_density = mc.params.getFixedDensity()
        mode_speed = mc.demand_characteristics.getSpeed()
        if mode_speed > 0:
            littles_law_density = self._travel_demand.getRateOfPMT(mode) / mode_speed
        else:
            littles_law_density = np.nan
        return fixed_density or littles_law_density

    def updateDemandCharacteristics(self):
        for mode in self.modes:
            self.setModeDemandCharacteristics(mode,
                                              copy.deepcopy(getModeDemandCharacteristics(self._baseSpeed,
                                                                                         self.getModeCharacteristics(
                                                                                             mode),
                                                                                         self._travel_demand)))

    def updateSupplyCharacteristics(self):
        for mode in self.modes:
            density = self.getModeDensity(mode)
            L_eq = getModeBlockedDistance(self, mode)
            N_eq = (self.getModeCharacteristics(mode).params.size or 1.0) * density
            supplyCharacteristics = supply.SupplyCharacteristics(density, N_eq, L_eq)
            self.setModeSupplyCharacteristics(mode, supplyCharacteristics)

    def getNewSpeedFromDensities(self):
        N_eq = np.sum([self.getModeCharacteristics(mode).supply_characteristics.getN() for mode in self.modes])
        L_eq = self.network_params.L - np.sum(
            [self.getModeCharacteristics(mode).supply_characteristics.getL() for mode in self.modes])
        return self.network_params.MFD(N_eq, L_eq)

    def setSpeed(self, speed):
        self._baseSpeed = speed
        self.updateDemandCharacteristics()
        self.updateSupplyCharacteristics()

    def findEquilibriumDensityAndSpeed(self):
        newData = copy.deepcopy(self)
        oldData = copy.deepcopy(self)
        keepGoing = True
        ii = 0
        while keepGoing:
            newSpeed = newData.getNewSpeedFromDensities()
            print('New Speed: ', newSpeed)
            newData.setSpeed(newSpeed)
            print('Diff: ', np.abs(newData._baseSpeed - oldData._baseSpeed))
            keepGoing = (np.abs(newData._baseSpeed - oldData._baseSpeed) > 0.001) & (ii < 20)
            oldData = copy.deepcopy(newData)
            if ii == 20:
                newSpeed = 0.0
        self.setSpeed(newSpeed)

    def getFlows(self):
        return [np.nan_to_num(np.max([self.getModeFlow(mode), 0.0])) for mode in
                self.modes]

    def getSpeeds(self):
        return [self.getModeSpeed(mode) for mode in self.modes]

    def getDemandsForPMT(self):
        return [self.getModeDemandForPMT(mode) for mode in self.modes]

    def getTravelTimes(self):
        speeds = np.array(self.getSpeeds())
        speeds[~(speeds > 0)] = np.nan
        distances = np.array([self.getModeMeanDistance(mode) for mode in self.modes])
        return distances / speeds

    def getTotalTimes(self):
        speeds = self.getSpeeds()
        demands = self.getDemandsForPMT()
        return np.array(speeds) * np.array(demands)

    def print(self):
        print('------------')
        print('Modes:')
        print(self.modes)
        print('Supply Characteristics:')
        print(self._mode_characteristics)
        print('Demand Characteristics:')
        print(self._travel_demand)
        print('------------')


def main():
    network_params_default = Network(0.068, 15.42, 1.88, 0.145, 0.177, 1000, 50)
    bus_params_default = BusParams(road_network_fraction=1000, relative_length=3.0,
                                   fixed_density=150. / 100., min_stop_time=15., stop_spacing=1. / 500.,
                                   passenger_wait=5.)

    car_params_default = ModeParams(relative_length=1.0)

    modeCharacteristics = CollectedModeCharacteristics()
    modeCharacteristics['car'] = ModeCharacteristics('car', car_params_default)
    modeCharacteristics['bus'] = ModeCharacteristics('bus', bus_params_default)

    m = Microtype(network_params_default, modeCharacteristics)
    m.setModeDemand('car', 70 / (10 * 60), 1000.0)
    m.setModeDemand('bus', 10 / (10 * 60), 1000.0)
    m.print()


def getDefaultDemandCharacteristics(mode):
    """

    :param mode: str
    :return: DemandCharacteristics
    """
    if mode == 'car':
        return supply.DemandCharacteristics(15., 0.0)
    elif mode == 'bus':
        return supply.BusDemandCharacteristics(15., 0.0, 0.0, 0.0, 0.0)
    else:
        return supply.DemandCharacteristics(15., 0.0)


def getDefaultSupplyCharacteristics():
    return supply.SupplyCharacteristics(0.0, 0.0, 0.0)


def getBusdwellTime(v, params_bus, trip_start_rate, trip_end_rate):
    if v > 0:
        out = 1. / (params_bus.s_b * v) * (
                v * params_bus.k * params_bus.t_0 * params_bus.s_b +
                params_bus.gamma_s * 2 * (trip_start_rate + trip_end_rate)) / (
                      params_bus.k - params_bus.gamma_s * (trip_start_rate + trip_end_rate))
    else:
        out = np.nan
    return out


def getModeDemandCharacteristics(base_speed: float, mode_characteristics: ModeCharacteristics, td: TravelDemand):
    mode = mode_characteristics.mode_name
    mode_params = mode_characteristics.params
    if mode == 'car':
        return DemandCharacteristics(base_speed, td.getRateOfPMT(mode))
    elif mode == 'bus':
        assert (isinstance(mode_params, BusParams))
        dwellTime = getBusdwellTime(base_speed, mode_params, td.getStartRate(mode), td.getEndRate(mode))
        if dwellTime > 0:
            speed = base_speed / (1 + dwellTime * base_speed * mode_params.s_b)
            headway = mode_params.road_network_fraction / speed
        else:
            speed = 0.0
            headway = np.nan

        if (dwellTime > 0) & (base_speed > 0):
            passengerFlow: float = td.getRateOfPMT(mode)
            occupancy: float = passengerFlow / mode_params.k / speed
        else:
            passengerFlow: float = 0.0
            occupancy: float = np.nan

        return BusDemandCharacteristics(speed, passengerFlow, dwellTime, headway, occupancy)

    else:
        return DemandCharacteristics(base_speed, td.getRateOfPMT(mode))


def getModeBlockedDistance(microtype, mode):
    """

    :rtype: float
    :param microtype: Microtype
    :param mode: str
    :return: float
    """
    if mode == 'car':
        return 0.0
    elif mode == 'bus':
        modeParams = microtype.getModeCharacteristics(mode).params
        modeSpeed = microtype.getModeSpeed(mode)
        trip_start_rate, trip_end_rate = microtype.getStartAndEndRate(mode)
        dwellTime = getBusdwellTime(microtype.getBaseSpeed(), modeParams, trip_start_rate, trip_end_rate)
        return microtype.network_params.l * modeParams.road_network_fraction * modeParams.s_b * modeParams.k * dwellTime * modeSpeed / microtype.network_params.L
    else:
        return 0.0


if __name__ == "__main__":
    main()