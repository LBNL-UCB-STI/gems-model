import os

import matplotlib.pyplot as plt
import numpy as np

from model import Model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data-simpler", nSubBins=2)
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -0.02
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "auto", "BetaTravelTime"] = -0.02
model.readFiles()
# Redo this
fig, axs = plt.subplots(5, 4)
demands = [180, 240, 280, 340]
# demands = [240, 260, 280, 300]
# demands = [390, 400, 405, 410]
totalTimesPlot = []
totalTimesSpeed = []
for ind, d in enumerate(demands):
    model.updateTimePeriodDemand('2', d)
    model.updateTimePeriodDemand('3', d)

    # model.updateTimePeriodDemand('3', d)
    # model.updateTimePeriodDemand('4', d)
    # model.updateTimePeriodDemand('5', d)

    # model.updateTimePeriodDemand('4', d)
    # model.updateTimePeriodDemand('5', d)
    # model.updateTimePeriodDemand('6', d)
    # model.updateTimePeriodDemand('7', d)

    vectorUserCosts, utils = model.collectAllCharacteristics()
    x, y = model.plotAllDynamicStats("delay")
    axs[0, ind].clear()
    axs[0, ind].set_title("Demand: " + str(d))
    axs[0, ind].plot(x, y[0,:,:])
    axs[0, ind].plot(x, y[1, :, :])

    axs[1, ind].clear()
    axs[1, ind].plot(np.transpose(y[:,1:,0]), np.transpose(np.diff(y[:,:,0], axis=1)))
    axs[1, ind].set_ylim((0, 6))

    axs[2, ind].clear()
    axs[2, ind].plot(x[:-1], np.interp(np.transpose(y[0, :,0]), np.transpose(y[1, :, 0]), x)[:-1] / 60. - x[:-1] / 60., 'C3')
    axs[2, ind].set_ylim((0, 40))
    axs[2, ind].step(np.linspace(0, 3 * 3600, len(model.timePeriods()) + 1),
                     np.vstack([6436 / model.getModeSpeeds('0')['auto'] / 60] +
                               [6436 / model.getModeSpeeds(p)['auto'] / 60 for p in model.timePeriods().keys()]),
                     color='C4')
    # axs[1, ind].plot(x[1:], np.diff(y, axis=0))
    # axs[1, ind].step(np.array([0, 1, 2, 3]) * 3600, np.vstack(
    #     [6436 / model.getModeSpeeds(0)['bus'], 6436 / model.getModeSpeeds(0)['bus'],
    #      6436 / model.getModeSpeeds(1)['bus'], 6436 / model.getModeSpeeds(2)['bus']]))

    axs[3, ind].clear()
    # axs[2, ind].plot(x, y[:, 0] - y[:, 1])
    axs[3, ind].step(np.arange(len(model.timePeriods()) + 1),
                     np.vstack([model.getModeSplit('0')] + [model.getModeSplit(p) for p in model.timePeriods().keys()]))
    axs[3, ind].set_ylim([0, 1])

    axs[3, ind].lines[model.modeToIdx['auto']].set_color('C5')
    axs[3, ind].lines[model.modeToIdx['bus']].set_color('C6')
    axs[3, ind].lines[model.modeToIdx['walk']].set_color('C7')

    axs[4, ind].clear()
    # axs[3, ind].set_ylim((200,1000))
    # axs[3, ind].step(np.array([0, 1, 2, 3])*3600, np.vstack(
    #     [model.getModeSpeeds(0)['auto'], model.getModeSpeeds(0)['auto'], model.getModeSpeeds(1)['auto'], model.getModeSpeeds(2)['auto']]))
    # axs[1, ind].plot(x[1:], np.diff(y, axis=0))
    # axs[3, ind].step(np.array([0, 1, 2, 3]) * 3600, np.vstack(
    #     [model.getModeSpeeds(0)['bus'], model.getModeSpeeds(0)['bus'],
    #      model.getModeSpeeds(1)['bus'], model.getModeSpeeds(2)['bus']]))
    # axs[3, ind].step(np.array([0, 1, 2, 3]) * 3600, np.vstack(
    #     [model.getModeSpeeds(0)['walk'], model.getModeSpeeds(0)['walk'],
    #      model.getModeSpeeds(1)['walk'], model.getModeSpeeds(2)['walk']]))

    axs[4, ind].step(np.arange(len(model.timePeriods()) + 1) * 1800,
                     np.vstack([utils[:, :, 0, :].mean(axis=1)[0, :], utils[:, :, 0, :].mean(axis=1)]))
    axs[4, ind].lines[model.modeToIdx['auto']].set_color('C7')
    axs[4, ind].lines[model.modeToIdx['bus']].set_color('C5')
    axs[4, ind].lines[model.modeToIdx['walk']].set_color('C6')
    totalTimesPlot.append(np.sum(y[:, 0] - y[:, 1]))
    totalTimesSpeed.append(vectorUserCosts[2, 1] * 60)
axs[0, 0].set_ylabel("Cumulative vehicles")
axs[0, 0].set_xlabel("Time")
axs[0, 0].legend(['Arrivals', 'Departures'])
axs[1, 0].set_ylabel("Arrival/Departure Rate")
axs[1, 0].set_xlabel("Cumulative vehicles")
axs[2, 0].set_ylabel("Travel Time (min)")
axs[2, 0].set_xlabel("Time")
axs[2, 0].legend(["Inst. Travel Time", "Avg. Travel Time"])
axs[3, 0].set_ylabel("Mode Split")
axs[3, 0].legend(list(model.modeToIdx.keys()))
axs[4, 0].set_ylabel("Utility")
axs.flat[0].set_ylabel("Cumulative vehicles")

fig.suptitle('No Bus Lanes', fontsize=16)

print('done')

# model.scenarioData['populationGroups'].loc[
#         model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -0.01
# model.scenarioData['populationGroups'].loc[
#         model.scenarioData['populationGroups']['Mode'] == "bus", "BetaWaitTime"] = 0.0
model.scenarioData['subNetworkData'].loc[2, :] = model.scenarioData['subNetworkData'].loc[1, :].copy()
model.scenarioData['subNetworkData'].loc[2, "Length"] = 0
model.scenarioData['subNetworkDataFull'].loc[2, :] = model.scenarioData['subNetworkDataFull'].loc[1, :].copy()
model.scenarioData['subNetworkDataFull'].loc[2, "Length"] = 0
model.scenarioData['subNetworkDataFull'].loc[2, "ModesAllowed"] = 'Bus-Walk'
model.scenarioData['subNetworkDataFull'].loc[2, "Dedicated"] = True
newmap = model.scenarioData['modeToSubNetworkData'].iloc[1:, ].copy()
newmap['SubnetworkID'] += 1
model.scenarioData['modeToSubNetworkData'] = model.scenarioData['modeToSubNetworkData'].append(newmap)
# model.scenarioData['modeToSubNetworkData'].set_index('SubnetworkID', inplace=True)
model.scenarioData['modeToSubNetworkData'].reset_index(drop=True, inplace=True)

# model.scenarioData['modeData']['bus']['CoveragePortion'] = 0.95

model.readFiles()
model.initializeAllTimePeriods(True)
initialDistance = model.scenarioData['subNetworkData'].loc[1, "Length"]
busLaneDistance = 500

model.scenarioData['subNetworkData'].at[2, "Length"] = busLaneDistance
model.scenarioData['subNetworkData'].at[1, "Length"] = initialDistance - busLaneDistance
model.microtypes.updateNetworkData()

fig100m, axs = plt.subplots(5, 4)
demands = [180, 220, 260, 300]
totalTimesPlot = []
totalTimesSpeed = []
for ind, d in enumerate(demands):
    model.updateTimePeriodDemand('2', d)
    model.updateTimePeriodDemand('3', d)

    # model.updateTimePeriodDemand('3', d)
    # model.updateTimePeriodDemand('4', d)
    # model.updateTimePeriodDemand('5', d)

    # model.updateTimePeriodDemand('4', d)
    # model.updateTimePeriodDemand('5', d)
    # model.updateTimePeriodDemand('6', d)
    # model.updateTimePeriodDemand('7', d)

    vectorUserCosts, utils = model.collectAllCharacteristics()

    x, y = model.plotAllDynamicStats("delay")
    axs[0, ind].clear()
    axs[0, ind].set_title("Demand: " + str(d))
    axs[0, ind].plot(x, y)

    axs[1, ind].clear()
    axs[1, ind].plot(y[1:], np.diff(y, axis=0))
    axs[1, ind].set_ylim((0, 6))

    axs[2, ind].clear()
    axs[2, ind].plot(x[:-1], np.interp(y[:, 0], y[:, 1], x)[:-1] / 60. - x[:-1] / 60., 'C3')
    axs[2, ind].set_ylim((0, 40))
    axs[2, ind].step(np.linspace(0, 3 * 3600, len(model.timePeriods()) + 1),
                     np.vstack([6436 / model.getModeSpeeds('0')['auto'] / 60] +
                               [6436 / model.getModeSpeeds(p)['auto'] / 60 for p in model.timePeriods().keys()]),
                     color='C4')
    # axs[1, ind].plot(x[1:], np.diff(y, axis=0))
    # axs[1, ind].step(np.array([0, 1, 2, 3]) * 3600, np.vstack(
    #     [6436 / model.getModeSpeeds(0)['bus'], 6436 / model.getModeSpeeds(0)['bus'],
    #      6436 / model.getModeSpeeds(1)['bus'], 6436 / model.getModeSpeeds(2)['bus']]))

    axs[3, ind].clear()
    # axs[2, ind].plot(x, y[:, 0] - y[:, 1])
    axs[3, ind].step(np.arange(len(model.timePeriods()) + 1),
                     np.vstack([model.getModeSplit('0')] + [model.getModeSplit(p) for p in model.timePeriods().keys()]))

    axs[3, ind].set_ylim([0, 1])

    axs[3, ind].lines[model.modeToIdx['auto']].set_color('C5')
    axs[3, ind].lines[model.modeToIdx['bus']].set_color('C6')
    axs[3, ind].lines[model.modeToIdx['walk']].set_color('C7')

    axs[4, ind].clear()
    # axs[3, ind].set_ylim((200,1000))
    # axs[3, ind].step(np.array([0, 1, 2, 3])*3600, np.vstack(
    #     [model.getModeSpeeds(0)['auto'], model.getModeSpeeds(0)['auto'], model.getModeSpeeds(1)['auto'], model.getModeSpeeds(2)['auto']]))
    # axs[1, ind].plot(x[1:], np.diff(y, axis=0))
    # axs[3, ind].step(np.array([0, 1, 2, 3]) * 3600, np.vstack(
    #     [model.getModeSpeeds(0)['bus'], model.getModeSpeeds(0)['bus'],
    #      model.getModeSpeeds(1)['bus'], model.getModeSpeeds(2)['bus']]))
    # axs[3, ind].step(np.array([0, 1, 2, 3]) * 3600, np.vstack(
    #     [model.getModeSpeeds(0)['walk'], model.getModeSpeeds(0)['walk'],
    #      model.getModeSpeeds(1)['walk'], model.getModeSpeeds(2)['walk']]))

    axs[4, ind].step(np.arange(len(model.timePeriods()) + 1) * 1800,
                     np.vstack([utils[:, :, 0, :].mean(axis=1)[0, :], utils[:, :, 0, :].mean(axis=1)]))
    axs[4, ind].lines[model.modeToIdx['auto']].set_color('C5')
    axs[4, ind].lines[model.modeToIdx['bus']].set_color('C6')
    axs[4, ind].lines[model.modeToIdx['walk']].set_color('C7')
    totalTimesPlot.append(np.sum(y[:, 0] - y[:, 1]))
    totalTimesSpeed.append(vectorUserCosts[2, 1] * 60)
axs[0, 0].set_ylabel("Cumulative vehicles")
axs[0, 0].set_xlabel("Time")
axs[0, 0].legend(['Arrivals', 'Departures'])
axs[1, 0].set_ylabel("Arrival/Departure Rate")
axs[1, 0].set_xlabel("Cumulative vehicles")
axs[2, 0].set_ylabel("Travel Time (min)")
axs[2, 0].set_xlabel("Time")
axs[2, 0].legend(["Inst. Travel Time", "Avg. Travel Time"])
axs[3, 0].set_ylabel("Mode Split")
axs[3, 0].legend(list(model.modeToIdx.keys()))
axs[4, 0].set_ylabel("Utility")
axs.flat[0].set_ylabel("Cumulative vehicles")

fig100m.suptitle('0m Bus Lanes', fontsize=16)

print('done')
