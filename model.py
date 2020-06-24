import pandas as pd
import os
from utils.microtype import Microtype
from utils.network import Network, NetworkCollection, NetworkFlowParams, BusModeParams, \
    AutoModeParams, Costs
from typing import Dict, List


class ModeParamFactory:
    def __init__(self, path: str):
        self.path = path
        self.modeParams = dict()
        self.readFiles()

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        self.__path = path

    def readFiles(self):
        (_, _, filenames) = next(os.walk(os.path.join(self.path, "modes")))
        for file in filenames:
            self.modeParams[file.split(".")[0]] = pd.read_csv(os.path.join(self.path, "modes", file))

    def get(self, modeName: str, microtypeID: str):
        print("AAH")
        if modeName.lower() == "bus":
            data = self.modeParams["bus"]
            data = data.loc[data["MicrotypeID"] == microtypeID].iloc[0]
            return BusModeParams(1000. / data.Headway, data.VehicleSize, 15., data.StopSpacing, 5.)
        else:
            return AutoModeParams()


class Model:
    def __init__(self, path: str):
        self.path = path
        self.microtypes = dict()
        self.__population = dict()
        self.modes = dict()
        self.readFiles()

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, path):
        self.__path = path

    def readFiles(self):
        microtypeData = pd.read_csv(os.path.join(self.path, "microtypes.csv"))
        subNetworkData = pd.read_csv(os.path.join(self.path, "subnetworks.csv"))
        modeToSubnetworkData = pd.read_csv(os.path.join(self.path, "mode-to-subnetwork.csv"))
        modeParamFactory = ModeParamFactory(self.path)
        for microtypeID, grouped in subNetworkData.groupby('MicrotypeID'):
            subNetworkToModes = dict()
            modeToModeParams = dict()
            allModes = set()
            for row in grouped.itertuples():
                joined = modeToSubnetworkData.loc[modeToSubnetworkData['SubnetworkID'] == row.SubnetworkID]
                subNetwork = Network(row.Length, NetworkFlowParams(0.068, 15.42, 1.88, 0.145, 0.177, 50))
                for n in joined.itertuples():
                    subNetworkToModes.setdefault(subNetwork, []).append(n.ModeType)
                    allModes.add(n.ModeType)
            for mode in allModes:
                modeToModeParams[mode] = modeParamFactory.get(mode, microtypeID)
            networkCollection = NetworkCollection(subNetworkToModes, modeToModeParams)
            costs1 = {'auto': Costs(0.0003778, 0., 3.0, 1.0), 'bus': Costs(0., 2.5, 0., 1.0)}
            self.microtypes[microtypeID] = Microtype(networkCollection, costs1)
            self.modes[microtypeID] = networkCollection.modes
            print("AAH")


if __name__ == "__main__":
    a = Model("input-data")
    print("aah")