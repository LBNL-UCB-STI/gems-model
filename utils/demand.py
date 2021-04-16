from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd

from .OD import TripCollection, OriginDestination, TripGeneration, DemandIndex, ODindex, ModeSplit, TransitionMatrices
from .choiceCharacteristics import CollectedChoiceCharacteristics
from .microtype import MicrotypeCollection
from .misc import DistanceBins, TimePeriods
from .population import Population


class TotalUserCosts:
    def __init__(self, total=0., totalEqualVOT=0., totalIVT=0., totalOVT=0., demandForTripsPerHour=0.,
                 demandForPMTPerHour=0.):
        self.total = total
        self.totalEqualVOT = totalEqualVOT
        self.totalIVT = totalIVT
        self.totalOVT = totalOVT
        self.demandForTripsPerHour = demandForTripsPerHour
        self.demandForPMTPerHour = demandForPMTPerHour

    def __str__(self):
        return str(self.total) + ' ' + str(self.totalEqualVOT)

    def __mul__(self, other):
        result = TotalUserCosts()
        result.total = self.total * other
        result.totalEqualVOT = self.totalEqualVOT * other
        result.totalIVT = self.totalIVT * other
        result.totalOVT = self.totalOVT * other
        result.demandForTripsPerHour = self.demandForTripsPerHour * other
        result.demandForPMTPerHour = self.demandForPMTPerHour * other
        return result

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        self.total *= other
        self.totalEqualVOT *= other
        self.totalIVT *= other
        self.totalOVT *= other
        self.demandForTripsPerHour *= other
        self.demandForPMTPerHour *= other
        return self

    def copy(self):
        return TotalUserCosts(self.total, self.totalEqualVOT, self.totalIVT, self.totalOVT, self.demandForTripsPerHour,
                              self.demandForPMTPerHour)

    def __add__(self, other):
        out = self.copy()
        out.total += other.total
        out.totalEqualVOT += other.totalEqualVOT
        out.totalIVT += other.totalIVT
        out.totalOVT += other.totalOVT
        out.demandForTripsPerHour += other.demandForTripsPerHour
        out.demandForPMTPerHour += other.demandForPMTPerHour
        return out

    def toDataFrame(self, index=None):
        return pd.DataFrame({"totalCost": self.total, "demandForTripsPerHour": self.demandForTripsPerHour,
                             "inVehicleTime": self.totalIVT, "outOfVehicleTime": self.totalOVT,
                             "demandForPMTPerHour": self.demandForPMTPerHour}, index=index)


class CollectedTotalUserCosts:
    def __init__(self):
        self.__costsByPopulationAndMode = dict()
        self.total = 0.
        self.totalEqualVOT = 0.
        self.demandForTripsPerHour = 0.
        self.demandForPMTPerHour = 0.

    def __setitem__(self, key: (DemandIndex, str), value: TotalUserCosts):
        self.__costsByPopulationAndMode[key] = value
        # self.updateTotals(value)

    def __getitem__(self, item) -> TotalUserCosts:
        if isinstance(item, DemandIndex):
            print('BAD')
            return TotalUserCosts()
        elif isinstance(item, str):
            print('BAD')
            return TotalUserCosts()
        elif isinstance(item, tuple):
            return self.__costsByPopulationAndMode[item]
        else:
            print("BADDDDD")
            return TotalUserCosts()

    def __iter__(self):
        return iter(self.__costsByPopulationAndMode.items())

    def updateTotals(self):
        self.total = sum([c.total for c in self.__costsByPopulationAndMode.values()])
        self.totalEqualVOT = sum([c.totalEqualVOT for c in self.__costsByPopulationAndMode.values()])
        self.demandForTripsPerHour = sum([c.demandForTripsPerHour for c in self.__costsByPopulationAndMode.values()])
        self.demandForPMTPerHour = sum([c.demandForPMTPerHour for c in self.__costsByPopulationAndMode.values()])
        return self

    def copy(self):
        out = CollectedTotalUserCosts()
        out.total = self.total
        out.totalEqualVOT = self.totalEqualVOT
        out.demandForTripsPerHour = self.demandForTripsPerHour
        out.demandForPMTPerHour = self.demandForPMTPerHour
        out.__costsByPopulationAndMode = self.__costsByPopulationAndMode.copy()
        return out

    def __imul__(self, other):
        for item in self.__costsByPopulationAndMode.keys():
            self.__costsByPopulationAndMode[item] = self.__costsByPopulationAndMode[item] * other
        return self

    def __mul__(self, other):
        out = self.copy()
        for item in out.__costsByPopulationAndMode.keys():
            out.__costsByPopulationAndMode[item] = out.__costsByPopulationAndMode[item] * other
        return out

    def __rmul__(self, other):
        return self * other

    def __iadd__(self, other):
        for item, cost in other:
            self.__costsByPopulationAndMode[item] = self.__costsByPopulationAndMode.setdefault(item,
                                                                                               TotalUserCosts()) + cost
        return self

    def toDataFrame(self, index=None) -> pd.DataFrame:
        muc = pd.concat([val.toDataFrame(pd.MultiIndex.from_tuples([key[0].toTupleWith(key[1])])) for key, val in
                         self.__costsByPopulationAndMode.items()])
        muc.index.set_names(['homeMicrotype', 'populationGroupType', 'tripPurpose', 'mode'], inplace=True)
        return muc.swaplevel(0, -1)
        # return pd.concat([val.toDataFrame([key.toIndex()]) for key, val in self.__costs.items()])

    def groupBy(self, vals) -> pd.DataFrame:
        df = self.toDataFrame()
        return df.groupby(level=vals).agg(sum)


class Demand:
    def __init__(self, modes: set):
        self.__modes = modes
        self.__modeToIdx = {val: ind for ind, val in enumerate(modes)}
        self.__modeSplit = dict()
        self.__odiToIdx = defaultdict(lambda: -1)
        self.__diToIdx = defaultdict(lambda: -1)
        self.__modeSplitData = np.ndarray(0)
        self.__tripRate = np.ndarray(0)
        self.__toStarts = np.ndarray(0)
        self.__toEnds = np.ndarray(0)
        self.__toThroughDistance = np.ndarray(0)
        self.tripRate = 0.0
        self.demandForPMT = 0.0
        self.pop = 0.0
        self.timePeriodDuration = 0.0
        self.__population = Population(modes)
        self.__trips = TripCollection()
        self.__distanceBins = DistanceBins()
        self.__transitionMatrices = TransitionMatrices()

    @property
    def odiToIdx(self):
        return self.__odiToIdx

    @property
    def diToIdx(self):
        return self.__diToIdx

    @property
    def modeToIdx(self):
        return self.__modeToIdx

    def nModes(self):
        return len(self.__modes)

    def keys(self):
        return product(self.__diToIdx.keys(), self.__odiToIdx.keys())

    # def __iter__(self) -> ((DemandIndex, ODindex), np.ndarray):
    #     return np.ndenumerate(self.__numpy)

    def __setitem__(self, key: (DemandIndex, ODindex), value: ModeSplit):
        self.__modeSplit[key] = value

    def __getitem__(self, item: (DemandIndex, ODindex)) -> ModeSplit:
        if item in self:
            return self.__modeSplit[item]
        else:  # else return empty mode split
            (demandIndex, odi) = item
            print("WTF")

    def __contains__(self, item):
        """ Return true if the correct value"""
        if item in self.__modeSplit:
            return True
        else:
            return False

    def initializeDemand(self, population: Population, originDestination: OriginDestination,
                         tripGeneration: TripGeneration, trips: TripCollection, microtypes: MicrotypeCollection,
                         distanceBins: DistanceBins, transitionMatrices: TransitionMatrices, timePeriods: TimePeriods,
                         currentTimePeriod: int, multiplier=1.0):
        self.__population = population
        self.__trips = trips
        self.__distanceBins = distanceBins
        self.__transitionMatrices = transitionMatrices
        self.timePeriodDuration = timePeriods[currentTimePeriod]

        # newTransitionMatrix = microtypes.emptyTransitionMatrix()
        # weights = transitionMatrices.emptyWeights()
        weights = np.zeros(len(trips))
        # counter = 0
        # for demandIndex, _ in population:
        #     for _, _ in originDestination[demandIndex].items():
        #         counter += 1
        for demandIndex, _ in population:
            od = originDestination[demandIndex]
            for odi, _ in od.items():
                # TODO: Make smoother
                trip = trips[odi]
        self.__modeSplitData = np.zeros((len(population), len(trips), len(self.__modes)))
        self.__modeSplitData[:, :, self.__modeToIdx['auto']] = 0.9
        self.__modeSplitData[:, :, self.__modeToIdx['walk']] = 0.1

        self.__tripRate = np.zeros((len(population), len(trips)))
        self.__toStarts = np.zeros((len(population), len(trips), len(microtypes)))
        self.__toEnds = np.zeros((len(population), len(trips), len(microtypes)))
        self.__toThroughDistance = np.zeros((len(population), len(trips), len(microtypes)))
        self.__toThroughCounts = np.zeros((len(population), len(trips), len(microtypes)))

        popCounter = 0
        maxODindex = 0
        for demandIndex, utilityParams in population:
            od = originDestination[demandIndex]
            ratePerHourPerCapita = tripGeneration[demandIndex.populationGroupType, demandIndex.tripPurpose] * multiplier
            pop = population.getPopulation(demandIndex.homeMicrotype, demandIndex.populationGroupType)

            for odi, portion in od.items():
                trip = trips[odi]
                common_modes = [microtypes[trip.odIndex.o].mode_names, microtypes[trip.odIndex.d].mode_names]
                # # Now we're switching over to only looking at origin and destination modes
                # common_modes = []
                # for microtypeID, allocation in trip.allocation:
                #     if allocation > 0:
                #         common_modes.append(microtypes[microtypeID].mode_names)
                modes = set.intersection(*common_modes)
                tripRatePerHour = ratePerHourPerCapita * pop * portion
                self.tripRate += tripRatePerHour
                demandForPMT = ratePerHourPerCapita * pop * portion * distanceBins[odi.distBin]
                # newTransitionMatrix.addAndMultiply(transitionMatrices[odi],
                #                                    tripRatePerHour)  # += transitionMatrices[odi] * tripRatePerHour

                self.demandForPMT += demandForPMT
                self.pop += pop
                # modeSplit = dict()
                # for mode in modes:
                #     if mode == "auto":
                #         modeSplit[mode] = 1.0
                #     else:
                #         modeSplit[mode] = 0.0
                if odi not in self.__odiToIdx:
                    self.__odiToIdx[odi] = maxODindex
                    maxODindex += 1
                currentODindex = self.__odiToIdx[odi]
                weights[currentODindex] += tripRatePerHour
                self.__tripRate[popCounter, currentODindex] = tripRatePerHour
                self.__toStarts[popCounter, currentODindex, microtypes.microtypeToIdx[odi.o]] = 1.0
                self.__toEnds[popCounter, currentODindex, microtypes.microtypeToIdx[odi.d]] = 1.0
                for mID, pct in trip.allocation:
                    self.__toThroughDistance[
                        popCounter, currentODindex, microtypes.microtypeToIdx[mID]] = pct * distanceBins[odi.distBin]
                    self.__toThroughCounts[popCounter, currentODindex, microtypes.microtypeToIdx[mID]] = 1.0
                self[demandIndex, odi] = ModeSplit(demandForTrips=tripRatePerHour, demandForPMT=demandForPMT,
                                                   data=self.__modeSplitData[popCounter, currentODindex, :],
                                                   modes=self.__modes)

                # dist, alloc = transitionMatrices[odi].getSteadyState()
                # distReal = self.__distanceBins[odi.distBin] * 1609.34
                # allocReal = trip.allocation.sortedValueArray()
                # diff = alloc - allocReal
                # print("WHAT")
            self.__diToIdx[demandIndex] = popCounter
            popCounter += 1

        self.__transitionMatrices.reIndex(self.__odiToIdx, currentTimePeriod)
        otherMatrix = transitionMatrices.averageMatrix(weights)
        microtypes.transitionMatrix.updateMatrix(otherMatrix)

    # @profile
    def updateMFD(self, microtypes: MicrotypeCollection, nIters=3):
        for microtypeID, microtype in microtypes:
            microtype.resetDemand()
        totalDemandForTrips = 0.0
        startsByMode = np.einsum('...,...i->...i', self.__tripRate, self.__modeSplitData)
        startsByOrigin = np.einsum('ijk,ijl->lk', startsByMode, self.__toStarts)
        startsByDestination = np.einsum('ijk,ijl->lk', startsByMode, self.__toEnds)
        distanceByMicrotype = np.einsum('ijk,ijl->lk', startsByMode, self.__toThroughDistance)
        throughCountsByMicrotype = np.einsum('ijk,ijl->lk', startsByMode, self.__toThroughCounts)
        newData = np.stack([startsByOrigin, startsByDestination, throughCountsByMicrotype, distanceByMicrotype],
                           axis=-1)
        microtypes.updateNumpy(newData)
        weights = np.sum(startsByMode[:, :, self.__modeToIdx["auto"]], axis=0)
        # for di, odi in self.keys():
        #     if (di, odi) not in self:
        #         continue
        #     else:
        #         ms = self[(di, odi)]
        #     for mode, split in ms:
        #         if split > 0:
        #             microtypes[odi.o].addModeStarts(mode, ms.demandForTripsPerHour * split)
        #             microtypes[odi.d].addModeEnds(mode, ms.demandForTripsPerHour * split)
        #             if mode == "auto":
        #                 weights[self.__transitionMatrices.idx(odi)] += ms.demandForTripsPerHour * split
        #                 totalDemandForTrips += ms.demandForTripsPerHour * split
        #             else:
        #                 newAllocation = microtypes.filterAllocation(mode, self.__trips[odi].allocation)
        #                 for k, portion in newAllocation.items():
        #                     microtypes[k].addModeDemandForPMT(mode, ms.demandForTripsPerHour * split,
        #                                                       self.__distanceBins[odi.distBin])
        otherMatrix = self.__transitionMatrices.averageMatrix(weights)
        microtypes.transitionMatrix.updateMatrix(otherMatrix)

        for it in range(nIters):
            microtypes.transitionMatrixMFD(self.timePeriodDuration)
            for microtypeID, microtype in microtypes:
                microtype.updateNetworkSpeeds(1)

    def updateModeSplit(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
                        originDestination: OriginDestination, oldModeSplit: ModeSplit):
        newModeSplit = modeSplitMatrixCalc(self.__population.numpy, collectedChoiceCharacteristics.numpy)
        np.copyto(self.__modeSplitData, newModeSplit)
        # for demandIndex, utilityParams in self.__population:
        #     od = originDestination[demandIndex]
        #     for odi, portion in od.items():
        #         ms = self.__population[demandIndex].updateModeSplit(
        #             collectedChoiceCharacteristics[self.__odiToIdx[odi]])
        #         self[demandIndex, odi].updateMapping(ms)
        #         self[demandIndex, odi] *= oldModeSplit
        newModeSplit = self.getTotalModeSplit()
        diff = oldModeSplit - newModeSplit
        return diff

    def getTotalModeSplit(self, userClass=None, microtypeID=None, distanceBin=None, otherModeSplit=None) -> ModeSplit:
        demandForTrips = 0
        demandForDistance = 0
        trips = dict()
        for (di, odi), ms in self.__modeSplit.items():
            relevant = ((userClass is None) or (di.populationGroupType == userClass)) & (
                    (microtypeID is None) or (di.homeMicrotype == microtypeID)) & (
                               (distanceBin is None) or (odi.distBin == distanceBin))
            if relevant:
                ar = self.__modeSplitData[self.__diToIdx[di], self.__odiToIdx[odi], :]
                for mode, split in ms:
                    new_demand = trips.setdefault(mode, 0) + split * ms.demandForTripsPerHour
                    trips[mode] = new_demand
                demandForTrips += ms.demandForTripsPerHour
                demandForDistance += ms.demandForPmtPerHour
        for mode in trips.keys():
            if otherModeSplit is not None:
                trips[mode] /= (demandForTrips * 2.)
                trips[mode] += otherModeSplit[mode] / 2.
            else:
                trips[mode] /= demandForTrips
        return ModeSplit(trips, demandForTrips, demandForDistance)

    def getMatrixUserCosts(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics) -> np.ndarray:
        startsByMode = np.einsum('...,...i->...i', self.__tripRate, self.__modeSplitData)
        costByMode = utils(self.__population.numpyCost, collectedChoiceCharacteristics.numpy)
        return startsByMode * costByMode

    def getUserCosts(self, collectedChoiceCharacteristics: CollectedChoiceCharacteristics,
                     originDestination: OriginDestination, modes=None) -> CollectedTotalUserCosts:
        out = CollectedTotalUserCosts()
        for demandIndex, utilityParams in self.__population:
            totalCost = 0.
            totalCostDefault = 0.
            totalDemandForTripsPerHour = 0.
            totalDemandForPMTPerHour = 0.
            totalInVehicle = 0.
            totalOutVehicle = 0.
            od = originDestination[demandIndex]
            demandClass = self.__population[demandIndex]
            for odi, portion in od.items():
                ms = self[(demandIndex, odi)]
                mcc = collectedChoiceCharacteristics[self.__odiToIdx[odi]]
                for mode in ms.keys():
                    cost, inVehicle, outVehicle, demandForTripsPerHour, distance = demandClass.getCostPerCapita(mcc, ms,
                                                                                                                [mode])
                    if demandForTripsPerHour > 0:
                        out[demandIndex, mode] = TotalUserCosts(cost * demandForTripsPerHour, 0.0,
                                                                inVehicle * demandForTripsPerHour,
                                                                outVehicle * demandForTripsPerHour,
                                                                demandForTripsPerHour,
                                                                demandForTripsPerHour * distance)
                # cost, inVehicle, outVehicle, demandForTripsPerHour, distance = demandClass.getCostPerCapita(mcc, ms,
                #                                                                                             modes)
                # # costDefault = demandClass.getCostPerCapita(mcc, ms, modes) * ms.demandForTripsPerHour  # TODO: Add default
                # totalCost -= cost * demandForTripsPerHour
                # totalInVehicle += inVehicle * demandForTripsPerHour
                # totalOutVehicle += outVehicle * demandForTripsPerHour
                # totalCostDefault -= 0.0
                # totalDemandForTripsPerHour += demandForTripsPerHour
                # totalDemandForPMTPerHour += distance * demandForTripsPerHour
            # out[demandIndex] = TotalUserCosts(totalCost, totalCostDefault, totalInVehicle, totalOutVehicle,
            # totalDemandForTripsPerHour, totalDemandForPMTPerHour)
        return out.updateTotals()

    def __str__(self):
        return "Trips: " + str(self.tripRate) + ", PMT: " + str(self.demandForPMT)


def utils(popVars: np.ndarray, choiceChars: np.ndarray) -> np.ndarray:
    return np.einsum('ikl,jkl->ijk', popVars, choiceChars)


def modeSplitMatrixCalc(popVars: np.ndarray, choiceChars: np.ndarray) -> np.ndarray:
    expUtils = np.exp(utils(popVars, choiceChars))
    probabilities = expUtils / np.expand_dims(np.nansum(expUtils, axis=2), 2)
    probabilities[np.isnan(expUtils)] = 0
    return probabilities
