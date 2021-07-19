from collections import OrderedDict

import pandas as pd


class TimePeriods:
    """
    Collection of time period of day and length of said periods.
    """

    def __init__(self):
        self.__timePeriods = OrderedDict()
        self.__ids = OrderedDict()

    def __setitem__(self, key: str, value: float):
        self.__timePeriods[key] = value

    def __getitem__(self, item) -> float:
        return self.__timePeriods[item]

    def __iter__(self):
        return iter(self.__timePeriods.items())

    def importTimePeriods(self, df: pd.DataFrame, nSubBins=1):
        idx = 0
        for row in df.itertuples():
            for sub in range(nSubBins):
                self[str(idx)] = row.DurationInHours / nSubBins
                self.__ids[str(idx)] = row.TimePeriodID
                idx += 1
        print("|  Loaded ", len(df) * nSubBins, " time periods")

    def __contains__(self, item):
        if item in self.__timePeriods:
            return True
        else:
            return False

    def getTimePeriodName(self, item):
        return self.__ids[item]


class DistanceBins:
    """
    Collection of distances of trips.
    """

    def __init__(self):
        self.__distanceBins = dict()

    def __setitem__(self, key: str, value: float):
        self.__distanceBins[key] = value

    def __getitem__(self, item) -> float:
        return float(self.__distanceBins[item])

    def importDistanceBins(self, df: pd.DataFrame):
        for row in df.itertuples():
            self[row.DistanceBinID] = row.MeanDistanceInMiles
        print("|  Loaded ", len(df), " distance bins")
