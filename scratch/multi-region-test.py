import os

import matplotlib.pyplot as plt
import numpy as np

from model import Model

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data")

model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -0.02
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "walk", "BetaTravelTime"] = -0.05
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bike", "BetaTravelTime"] = -0.08
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "rail", "BetaTravelTime"] = -0.03
model.readFiles()


initialDistance = model.scenarioData['subNetworkData'].loc[1, "Length"]
busLaneDistance = 0

model.scenarioData['subNetworkData'].at[9, "Length"] = busLaneDistance
model.scenarioData['subNetworkData'].at[1, "Length"] = initialDistance - busLaneDistance

initialDistance = model.scenarioData['subNetworkData'].loc[4, "Length"]
busLaneDistance = 0

model.scenarioData['subNetworkData'].at[12, "Length"] = busLaneDistance
model.scenarioData['subNetworkData'].at[4, "Length"] = initialDistance - busLaneDistance


vectorUserCosts, utils = model.collectAllCharacteristics()
# a.plotAllDynamicStats("N")
x, y = model.plotAllDynamicStats("delay")

colors = ['C0', 'C1', 'C2', 'C3', 'C0', 'C1', 'C2', 'C3']

fig, axs = plt.subplots(4, 4)
for ind, m in enumerate(model.microtypes):
    y1 = y[0,:,ind]
    y2 = y[1,:,ind]
    axs[0, ind].plot(x, y1, color = "#800080")
    axs[0, ind].plot(x, y2, color = "#00DBFF")
    axs[1, ind].plot(x, y1 - y2, color="#E56717")
    axs[2, ind].plot(x[:-1], np.interp(y1, y2, x)[:-1] * 60. - x[:-1] * 60., '#ED4337')
    axs[0, ind].set_title("Microtype " + m[0])

    axs[3, ind].clear()
    axs[3, ind].step(np.arange(len(model.timePeriods()) + 1),
                     np.vstack([model.getModeSplit('0', microtypeID=m[0])] + [model.getModeSplit(p, microtypeID=m[0]) for p in model.timePeriods().keys()]))
    axs[3, ind].set_ylim([0, 1])

    axs[3, ind].lines[model.modeToIdx['auto']].set_color('#C21807')
    axs[3, ind].lines[model.modeToIdx['bus']].set_color('#1338BE')
    axs[3, ind].lines[model.modeToIdx['walk']].set_color('#3CB043')
    axs[3, ind].lines[model.modeToIdx['rail']].set_color('orange')
    axs[3, ind].lines[model.modeToIdx['bike']].set_color('blue')

axs[3, 0].legend(['bus', 'rail','walk','bike','auto'])
axs[0, 0].set_ylabel('cumulative vehicles')
axs[1, 0].set_ylabel('accumulation')
axs[2, 0].set_ylabel('travel time')
axs[3, 0].set_ylabel('mode split')
print('DONE')