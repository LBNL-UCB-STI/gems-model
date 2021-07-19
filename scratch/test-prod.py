import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



import sys
sys.path.append("/Users/zaneedell/Desktop/git/task-3-modeling")

from model import Model

for factor in range(0,105,5):
    spds = dict()
    modesplits = dict()
    userCosts = dict()
    operatorCosts = dict()

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    a = Model(ROOT_DIR + "/../input-data-production")
    original = a.scenarioData['subNetworkData'].at[1, "Length"]
    dist = factor * original / 100.
    string = str(factor) + "pctbuslane.csv"

    #TODO - get this value from timePeriod df
    #index of timePeriod
    for timePeriod in [0, 1, 2, 3, 4]:
        a.initializeTimePeriod(timePeriod) # why we are re-initialising?
        a.scenarioData['subNetworkData'].at[3, "Length"] = dist              #
        a.scenarioData['subNetworkData'].at[1, "Length"] = original - dist
        a.findEquilibrium()
        spds[timePeriod] = a.getModeSpeeds()
        modesplits[timePeriod] = pd.DataFrame([a.getModeSplit()], index=["Aggregate"])
        userCosts[timePeriod] = a.getModeUserCosts()
        operatorCosts[timePeriod] = a.getOperatorCosts().toDataFrame()

    all = pd.concat(spds)
    all.columns = pd.MultiIndex.from_tuples(all.columns.to_series().apply(lambda x: (x[0], x[2])))
    all.to_csv("out/A_speeds-" + string)

    pd.concat(userCosts).to_csv("out/A_userCosts-"+string)
    pd.concat(operatorCosts).to_csv("out/A_operatorCosts-"+string)
    # for g in all.columns.get_level_values(0).unique():
    #     plt.plot(all[g].loc[("morning_rush","bus"),:])

    popGroups = a.scenarioData["populations"]["PopulationGroupTypeID"].unique()
    microtypes = a.scenarioData["populations"]["MicrotypeID"].unique()
    dbins = a.scenarioData["distanceBins"]["DistanceBinID"].unique()

    allModeSplits = dict()
    for timePeriod in a.scenarioData["timePeriods"].TimePeriodID.values:
        for popGroup in popGroups:
            allModeSplits[popGroup + '_' + timePeriod] = pd.DataFrame([a.getModeSplit(timePeriod=timePeriod, userClass=popGroup)],index=["popGroup"])

    for timePeriod in a.scenarioData["timePeriods"].TimePeriodID.values:
        for microtype in microtypes:
            allModeSplits[microtype + '_' + timePeriod] = pd.DataFrame([a.getModeSplit(timePeriod=timePeriod, microtypeID=microtype)], index=["microtype"])
            for popGroup in popGroups:
                allModeSplits[popGroup + '_' + microtype + '_' + timePeriod] = pd.DataFrame([a.getModeSplit(timePeriod=timePeriod, userClass=popGroup,microtypeID=microtype)], index=["popGroupMicrotype"])

    for timePeriod in a.scenarioData["timePeriods"].TimePeriodID.values:
        for dbin in dbins:
            allModeSplits[dbin + '_' + timePeriod] = pd.DataFrame([a.getModeSplit(timePeriod=timePeriod, distanceBin=dbin)], index=["dbin"])

    joined = pd.concat(allModeSplits)
    joined.to_csv("out/A_groupModeSplits-"+string)
    pd.concat(modesplits).to_csv("out/A_modeSplits-"+string)