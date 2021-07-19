import os
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from model import Model

colors = {'arrivals': '#800080',
          'departures': '#00DBFF',
          'accumulation': '#E56717',
          'autoTime': '#ED4337',
          'transitTime': '#3895D3',
          'walkTime': '#3DED97',
          'autoSplit': '#C21807',
          'transitSplit': '#1338BE',
          'walkSplit': '#3CB043',
          }
nSubBins = 6


def gatherOutputs(model: Model):
    t_timePeriod = np.cumsum(np.array(list(chain(
        *[[0]] + [[dur / model.nSubBins] * model.nSubBins for dur in
                  model.scenarioData['timePeriods'].DurationInHours]))))

    output = dict()
    vectorUserCosts, utils = model.collectAllCharacteristics()
    x, y = model.plotAllDynamicStats("delay")
    inflow = y[0, :, :]
    outflow = y[1, :, :]
    autoTravelTime = np.zeros((inflow.shape[0] - 1, inflow.shape[1]))
    for i in range(autoTravelTime.shape[1]):
        autoTravelTime[:, i] = np.interp(inflow[:, i], outflow[:, i], x)[:-1] * 60. - x[:-1] * 60.
    # autoTravelTime = np.interp(np.squeeze(inflow), np.squeeze(outflow), x)[:-1] * 60. - x[:-1] * 60.

    output['inflow'] = inflow
    output['outflow'] = outflow
    output['time'] = x
    output['autoTravelTime'] = autoTravelTime

    x, y = model.plotAllDynamicStats("n")

    output['accumulation'] = y

    autoTravelTime_timePeriod = np.vstack([6436 / model.getModeSpeeds('0')['auto'] / 60] +
                                          [6436 / model.getModeSpeeds(p)['auto'] / 60 for p in
                                           model.timePeriods().keys()])

    busTravelTime_timePeriod = np.vstack([6436 / model.getModeSpeeds('0')['bus'] / 60] +
                                         [6436 / model.getModeSpeeds(p)['bus'] / 60 for p in
                                          model.timePeriods().keys()])

    walkTravelTime_timePeriod = np.vstack([6436 / model.getModeSpeeds('0')['walk'] / 60] +
                                          [6436 / model.getModeSpeeds(p)['walk'] / 60 for p in
                                           model.timePeriods().keys()])

    autoSpeed_timePeriod = np.vstack([model.getModeSpeeds('0')['auto']] +
                                     [model.getModeSpeeds(p)['auto'] for p in model.timePeriods().keys()])

    busSpeed_timePeriod = np.vstack([model.getModeSpeeds('0')['bus']] +
                                    [model.getModeSpeeds(p)['bus'] for p in model.timePeriods().keys()])

    walkSpeed_timePeriod = np.vstack([model.getModeSpeeds('0')['walk']] +
                                     [model.getModeSpeeds(p)['walk'] for p in model.timePeriods().keys()])

    output['t_timePeriod'] = t_timePeriod
    output['autoTravelTime_timePeriod'] = autoTravelTime_timePeriod
    output['autoSpeed_timePeriod'] = autoSpeed_timePeriod
    output['busTravelTime_timePeriod'] = busTravelTime_timePeriod
    output['busSpeed_timePeriod'] = busSpeed_timePeriod
    output['walkTravelTime_timePeriod'] = walkTravelTime_timePeriod
    output['walkSpeed_timePeriod'] = walkSpeed_timePeriod

    x, y = model.plotAllDynamicStats("v")
    output['autoSpeed'] = y

    output['utils'] = np.vstack([utils[:, :, 0, :].mean(axis=1)[0, :], utils[:, :, 0, :].mean(axis=1)])

    modeSplit = np.zeros((len(t_timePeriod), len(model.modeToIdx), len(model.microtypes)))
    for mID, idx in model.microtypeIdToIdx.items():
        modeSplit[:, :, idx] = np.vstack(
            [model.getModeSplit('0', microtypeID=mID)] + [model.getModeSplit(p, microtypeID=mID) for p in
                                                          model.timePeriods().keys()])
    output['modeSplit'] = np.squeeze(modeSplit)

    return output


scenarios = dict()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data", nSubBins=nSubBins)

model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "auto", "BetaTravelTime"] = -0.04
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -0.02
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "walk", "BetaTravelTime"] = -0.05
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bike", "BetaTravelTime"] = -0.07
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "rail", "BetaTravelTime"] = -0.03
model.readFiles()

basePopulation = model.scenarioData['populations'].Population.values.copy()

popMultipliers = [1.0, 1.3, 1.45, 1.5]

for ind, pop in enumerate(popMultipliers):
    scenario = '4-microtype-pop-' + str(pop)

    model.scenarioData['populations'].Population = basePopulation * pop
    model.updatePopulation()

    scenarios[scenario] = gatherOutputs(model)

scenarioNames = ['4-microtype-pop-1.0', '4-microtype-pop-1.3', '4-microtype-pop-1.45', '4-microtype-pop-1.5']
titles = ['Base population', 'Population +30%', 'Population +45%', 'Population +50%']

mColors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

figs = []
axs = []
for i in range(4):
    fig, ax = plt.subplots(1, len(scenarioNames), figsize=(12, 4), sharey=True)
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]
    fig = figs[0]
    ax = axs[0]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['inflow'][:, idx], color=mColors[idx],
                     label='Microtype ' + mID + ' arrivals')
        ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['outflow'][:, idx], color=mColors[idx],
                     label='Microtype ' + mID + ' departures', linestyle='dashed')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles')
    ax[ind].set_title(title)

    fig = figs[1]
    ax = axs[1]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['inflow'][:, idx], axis=0),
                     color=mColors[idx],
                     label='Microtype ' + mID)
        ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['outflow'][:, idx], axis=0),
                     color=mColors[idx],
                     linestyle='dashed')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[2]
    ax = axs[2]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[ind].plot(scenarios[scenario]['inflow'][1:, idx], np.diff(scenarios[scenario]['inflow'][:, idx], axis=0),
                     color=mColors[idx],
                     label='Microtype ' + mID)
        ax[ind].plot(scenarios[scenario]['outflow'][1:, idx], np.diff(scenarios[scenario]['outflow'][:, idx], axis=0),
                     color=mColors[idx],
                     linestyle='dashed')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Number of vehicles')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[3]
    ax = axs[3]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['accumulation'][:, idx], color=mColors[idx], label="Microtype "+mID)
    ax[ind].set_xlabel('Time (h)')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if ind == 0:
        ax[ind].set_ylabel('Vehicles')
    ax[ind].set_title(title)

groupName = 'validationForQR/4-microtype-modes-'

figs[0].savefig(groupName + 'queuing.pdf', bbox_inches='tight')
figs[1].savefig(groupName + 'inout-t.pdf', bbox_inches='tight')
figs[2].savefig(groupName + 'inout-n.pdf', bbox_inches='tight')
figs[3].savefig(groupName + 'accumulation.pdf', bbox_inches='tight')

figs = []
axs = []
for i in range(2):
    fig, ax = plt.subplots(len(model.microtypes), len(scenarioNames), figsize=(12, 11), sharex=True, sharey=True)
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]
    fig = figs[0]
    ax = axs[0]

    for mID, idx in model.microtypeIdToIdx.items():
        ax[idx, ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoSpeed_timePeriod'][:, idx], color='C4',
                     label='Averaged auto')
        ax[idx, ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['walkSpeed_timePeriod'][:, idx], color='C2',
                     label='Averaged walk')
        ax[idx, ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busSpeed_timePeriod'][:, idx], color='C3',
                     label='Averaged bus')
        ax[idx, ind].plot(scenarios[scenario]['time'], scenarios[scenario]['autoSpeed'][:, idx], color='C4',
                     label='Instantaneous auto',
                     linestyle='dashed')
        if idx == 0:
            ax[idx, ind].set_title(title)
            if ind == len(scenarioNames) - 1:
                ax[idx, ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        if ind == 0:
            ax[idx, ind].set_ylabel('Speed (m/s)')

    fig = figs[1]
    ax = axs[1]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[idx, ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['modeSplit'][:, :, idx])
        # ax[idx, ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
        # ax[idx, ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
        # ax[idx, ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])
        if idx == 0:
            ax[idx, ind].set_title(title)
            if ind == len(scenarioNames) - 1:
                ax[idx, ind].legend(list(model.modeToIdx.keys()), bbox_to_anchor=(1.05, 1), loc='upper left')
        if ind == 0:
            ax[idx, ind].set_ylabel('Mode split')

figs[0].savefig(groupName + 'speed.pdf', bbox_inches='tight')
figs[1].savefig(groupName + 'modeSplit.pdf', bbox_inches='tight')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data", nSubBins=nSubBins)

model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "auto", "BetaTravelTime"] = -0.04
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -0.02
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "walk", "BetaTravelTime"] = -0.05
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bike", "BetaTravelTime"] = -0.07
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "rail", "BetaTravelTime"] = -0.03
model.readFiles()

model.scenarioData['populations'].Population = basePopulation * 1.4
model.updatePopulation()

busLanes = [0.0, 0.1, 0.15, 0.2]

for ind, portionDedicated in enumerate(busLanes):
    scenario = '4-microtype-dedicated-' + str(portionDedicated)

    initialDistance = model.scenarioData['subNetworkData'].loc[1, "Length"]
    routeDistance = model.scenarioData['modeData']['bus'].CoveragePortion[0] * initialDistance

    busLaneDistance = portionDedicated * routeDistance

    model.scenarioData['subNetworkData'].at[9, "Length"] = busLaneDistance
    model.scenarioData['subNetworkData'].at[1, "Length"] = initialDistance - busLaneDistance

    initialDistance = model.scenarioData['subNetworkData'].loc[2, "Length"]
    routeDistance = model.scenarioData['modeData']['bus'].CoveragePortion[1] * initialDistance

    busLaneDistance = portionDedicated * routeDistance

    model.scenarioData['subNetworkData'].at[10, "Length"] = busLaneDistance
    model.scenarioData['subNetworkData'].at[2, "Length"] = initialDistance - busLaneDistance

    scenarios[scenario] = gatherOutputs(model)

scenarioNames = ['4-microtype-dedicated-0.0', '4-microtype-dedicated-0.1', '4-microtype-dedicated-0.15', '4-microtype-dedicated-0.2']
titles = ['No bus lanes', '10% dedicated', '15% dedicated', '20% dedicated']

figs = []
axs = []
for i in range(4):
    fig, ax = plt.subplots(1, len(scenarioNames), figsize=(12, 4), sharey=True)
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]
    fig = figs[0]
    ax = axs[0]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['inflow'][:, idx], color=mColors[idx],
                     label='Microtype ' + mID + ' arrivals')
        ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['outflow'][:, idx], color=mColors[idx],
                     label='Microtype ' + mID + ' departures', linestyle='dashed')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles')
    ax[ind].set_title(title)

    fig = figs[1]
    ax = axs[1]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['inflow'][:, idx], axis=0),
                     color=mColors[idx],
                     label='Microtype ' + mID)
        ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['outflow'][:, idx], axis=0),
                     color=mColors[idx],
                     linestyle='dashed')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[2]
    ax = axs[2]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[ind].plot(scenarios[scenario]['inflow'][1:, idx], np.diff(scenarios[scenario]['inflow'][:, idx], axis=0),
                     color=mColors[idx],
                     label='Microtype ' + mID)
        ax[ind].plot(scenarios[scenario]['outflow'][1:, idx], np.diff(scenarios[scenario]['outflow'][:, idx], axis=0),
                     color=mColors[idx],
                     linestyle='dashed')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Number of vehicles')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[3]
    ax = axs[3]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['accumulation'][:, idx], color=mColors[idx], label="Microtype "+ mID)
    ax[ind].set_xlabel('Time (h)')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if ind == 0:
        ax[ind].set_ylabel('Vehicles')
    ax[ind].set_title(title)

groupName = 'validationForQR/4-microtype-busLanes-'

figs[0].savefig(groupName + 'queuing.pdf', bbox_inches='tight')
figs[1].savefig(groupName + 'inout-t.pdf', bbox_inches='tight')
figs[2].savefig(groupName + 'inout-n.pdf', bbox_inches='tight')
figs[3].savefig(groupName + 'accumulation.pdf', bbox_inches='tight')

figs = []
axs = []
for i in range(2):
    fig, ax = plt.subplots(len(model.microtypes), len(scenarioNames), figsize=(12, 11), sharex=True, sharey=True)
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]
    fig = figs[0]
    ax = axs[0]

    for mID, idx in model.microtypeIdToIdx.items():
        ax[idx, ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoSpeed_timePeriod'][:, idx], color='C4',
                     label='Averaged auto')
        ax[idx, ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['walkSpeed_timePeriod'][:, idx], color='C2',
                     label='Averaged walk')
        ax[idx, ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busSpeed_timePeriod'][:, idx], color='C3',
                     label='Averaged bus')
        ax[idx, ind].plot(scenarios[scenario]['time'], scenarios[scenario]['autoSpeed'][:, idx], color='C4',
                     label='Instantaneous auto',
                     linestyle='dashed')
        if idx == 0:
            ax[idx, ind].set_title(title)
            if ind == len(scenarioNames) - 1:
                ax[idx, ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        if ind == 0:
            ax[idx, ind].set_ylabel('Speed (m/s)')

    fig = figs[1]
    ax = axs[1]
    for mID, idx in model.microtypeIdToIdx.items():
        ax[idx, ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['modeSplit'][:, :, idx])
        # ax[idx, ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
        # ax[idx, ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
        # ax[idx, ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])
        if idx == 0:
            ax[idx, ind].set_title(title)
            if ind == len(scenarioNames) - 1:
                ax[idx, ind].legend(list(model.modeToIdx.keys()), bbox_to_anchor=(1.05, 1), loc='upper left')
        if ind == 0:
            ax[idx, ind].set_ylabel('Mode split')

figs[0].savefig(groupName + 'speed.pdf', bbox_inches='tight')
figs[1].savefig(groupName + 'modeSplit.pdf', bbox_inches='tight')






# %%
nSubBins = 4

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
model = Model(ROOT_DIR + "/../input-data-simpler", nSubBins=nSubBins)
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "bus", "BetaTravelTime"] = -1000.0
model.scenarioData['populationGroups'].loc[
    model.scenarioData['populationGroups']['Mode'] == "walk", "BetaTravelTime"] = -1000.0
model.scenarioData['modeData']['bus'].Headway = 1e6
model.readFiles()

pops = [800, 1500, 1700]

for ind, pop in enumerate(pops):
    scenario = 'pop-' + str(pop) + '-auto'

    model.scenarioData['populations'].Population = pop
    model.updatePopulation()

    scenarios[scenario] = gatherOutputs(model)

scenarioNames = ['pop-800-auto', 'pop-1500-auto', 'pop-1700-auto']
titles = ['Population 800', 'Population 1500', 'Population 1700']

figs = []
axs = []
for i in range(7):
    fig, ax = plt.subplots(1, len(scenarioNames), figsize=(9, 4), sharey=True)
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]

    fig = figs[0]
    ax = axs[0]

    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['inflow'], color=colors['departures'],
                 label='Arrivals')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['outflow'], color=colors['arrivals'],
                 label='Departures')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles')
    ax[ind].set_title(title)

    fig = figs[1]
    ax = axs[1]
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['inflow'], axis=0),
                 color=colors['departures'],
                 label='Arrivals')
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['outflow'], axis=0),
                 color=colors['arrivals'],
                 label='Departures')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[2]
    ax = axs[2]
    ax[ind].plot(scenarios[scenario]['inflow'][1:], np.diff(scenarios[scenario]['inflow'], axis=0),
                 color=colors['departures'],
                 label='Arrivals')
    ax[ind].plot(scenarios[scenario]['outflow'][1:], np.diff(scenarios[scenario]['outflow'], axis=0),
                 color=colors['arrivals'],
                 label='Departures')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Number of vehicles')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[3]
    ax = axs[3]
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['accumulation'], color=colors['accumulation'])
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Vehicles')
    ax[ind].set_title(title)

    fig = figs[4]
    ax = axs[4]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoSpeed_timePeriod'], color='C4',
                 label='Averaged')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['autoSpeed'], color='C5', label='Instantaneous')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Speed (m/s)')
    ax[ind].set_title(title)

    fig = figs[5]
    ax = axs[5]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoTravelTime_timePeriod'],
                 color=colors['autoTime'], label='Averaged')
    ax[ind].plot(scenarios[scenario]['time'][1:], scenarios[scenario]['autoTravelTime'], color='C6',
                 label='Instantaneous')
    if ind == len(scenarios) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Travel time (min)')
    ax[ind].set_title(title)

    fig = figs[6]
    ax = axs[6]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['utils'][:, -1], color='C1', label='Averaged')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Utility')
    ax[ind].set_title(title)

groupName = 'validationForQR/1-microtype-compare-'

figs[0].savefig(groupName + 'queuing.pdf', bbox_inches='tight')
figs[1].savefig(groupName + 'inout-t.pdf', bbox_inches='tight')
figs[2].savefig(groupName + 'inout-n.pdf', bbox_inches='tight')
figs[3].savefig(groupName + 'accumulation.pdf', bbox_inches='tight')
figs[4].savefig(groupName + 'speed.pdf', bbox_inches='tight')
figs[5].savefig(groupName + 'travelTime.pdf', bbox_inches='tight')
figs[6].savefig(groupName + 'util.pdf', bbox_inches='tight')
plt.close('all')

# %%
model = Model(ROOT_DIR + "/../input-data-simpler", nSubBins=nSubBins)
model.readFiles()

pops = [2000, 2900, 3800]

for ind, pop in enumerate(pops):
    scenario = 'pop-' + str(pop) + '-all'

    model.scenarioData['populations'].Population = pop
    model.updatePopulation()

    scenarios[scenario] = gatherOutputs(model)

scenarioNames = ['pop-2000-all', 'pop-2900-all', 'pop-3800-all']
titles = ['Population 2000', 'Population 2900', 'Population 3800']

figs = []
axs = []
for i in range(8):
    fig, ax = plt.subplots(1, len(scenarioNames), figsize=(9.5, 3.75), sharey=True)
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]

    fig = figs[0]
    ax = axs[0]

    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['inflow'], color=colors['departures'],
                 label='Arrivals')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['outflow'], color=colors['arrivals'],
                 label='Departures')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles')
    ax[ind].set_title(title)

    fig = figs[1]
    ax = axs[1]
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['inflow'], axis=0),
                 color=colors['departures'],
                 label='Arrivals')
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['outflow'], axis=0),
                 color=colors['arrivals'],
                 label='Departures')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[2]
    ax = axs[2]
    ax[ind].plot(scenarios[scenario]['inflow'][1:], np.diff(scenarios[scenario]['inflow'], axis=0),
                 color=colors['departures'],
                 label='Arrivals')
    ax[ind].plot(scenarios[scenario]['outflow'][1:], np.diff(scenarios[scenario]['outflow'], axis=0),
                 color=colors['arrivals'],
                 label='Departures')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Number of vehicles')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[3]
    ax = axs[3]
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['accumulation'], color=colors['accumulation'])
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Vehicles')
    ax[ind].set_title(title)

    fig = figs[4]
    ax = axs[4]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoSpeed_timePeriod'], color='C4',
                 label='Averaged auto')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['walkSpeed_timePeriod'], color='C2',
                 label='Averaged walk')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busSpeed_timePeriod'], color='C3',
                 label='Averaged bus')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['autoSpeed'], color='C4', label='Instantaneous auto',
                 linestyle='dashed')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Speed (m/s)')
    ax[ind].set_title(title)

    fig = figs[5]
    ax = axs[5]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoTravelTime_timePeriod'],
                 color=colors['autoTime'], label='Averaged auto')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busTravelTime_timePeriod'],
                 color=colors['transitTime'], label='Averaged bus')
    ax[ind].plot(scenarios[scenario]['time'][1:], scenarios[scenario]['autoTravelTime'], color=colors['autoTime'],
                 label='Instantaneous auto', linestyle='dashed')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Travel time (min)')
    ax[ind].set_title(title)

    fig = figs[6]
    ax = axs[6]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['utils'], label='Averaged')
    ax[ind].set_xlabel('Time (h)')
    ax[ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
    ax[ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
    ax[ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(['Bus', 'Walk', 'Auto'], bbox_to_anchor=(1.05, 1), loc='upper left')
    if ind == 0:
        ax[ind].set_ylabel('Utility')
    ax[ind].set_title(title)

    fig = figs[7]
    ax = axs[7]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['modeSplit'])
    ax[ind].set_xlabel('Time (h)')
    ax[ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
    ax[ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
    ax[ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(['Bus', 'Walk', 'Auto'], bbox_to_anchor=(1.05, 1), loc='upper left')
    if ind == 0:
        ax[ind].set_ylabel('Mode split')
    ax[ind].set_title(title)

groupName = 'validationForQR/1-microtype-modes-'

figs[0].savefig(groupName + 'queuing.pdf', bbox_inches='tight')
figs[1].savefig(groupName + 'inout-t.pdf', bbox_inches='tight')
figs[2].savefig(groupName + 'inout-n.pdf', bbox_inches='tight')
figs[3].savefig(groupName + 'accumulation.pdf', bbox_inches='tight')
figs[4].savefig(groupName + 'speed.pdf', bbox_inches='tight')
figs[5].savefig(groupName + 'travelTime.pdf', bbox_inches='tight')
figs[6].savefig(groupName + 'util.pdf', bbox_inches='tight')
figs[7].savefig(groupName + 'modeSplit.pdf', bbox_inches='tight')
plt.close('all')

# %%
model = Model(ROOT_DIR + "/../input-data-simpler", nSubBins=nSubBins)
model.readFiles()

# NOTE GOT RID OF THIS BECAUSE ADDED EMTPY BUS NETWORK TO DEFAULTS
# model.scenarioData['subNetworkData'].loc[2, :] = model.scenarioData['subNetworkData'].loc[1, :].copy()
# model.scenarioData['subNetworkData'].loc[2, "Length"] = 0
# model.scenarioData['subNetworkDataFull'].loc[2, :] = model.scenarioData['subNetworkDataFull'].loc[1, :].copy()
# model.scenarioData['subNetworkDataFull'].loc[2, "Length"] = 0
# model.scenarioData['subNetworkDataFull'].loc[2, "ModesAllowed"] = 'Bus-Walk'
# model.scenarioData['subNetworkDataFull'].loc[2, "Dedicated"] = True
# newmap = model.scenarioData['modeToSubNetworkData'].iloc[1:, ].copy()
# newmap['SubnetworkID'] += 1
# model.scenarioData['modeToSubNetworkData'] = model.scenarioData['modeToSubNetworkData'].append(newmap)
# model.scenarioData['modeToSubNetworkData'].reset_index(drop=True, inplace=True)
# model.readFiles()

initialDistance = model.scenarioData['subNetworkData'].loc[1, "Length"]
routeDistance = model.scenarioData['modeData']['bus'].CoveragePortion[0] * initialDistance

portionDedicated = 0.5

busLaneDistance = portionDedicated * routeDistance

model.scenarioData['subNetworkData'].at[2, "Length"] = busLaneDistance
model.scenarioData['subNetworkData'].at[1, "Length"] = initialDistance - busLaneDistance
model.microtypes.updateNetworkData()

scenarioNames = []

for ind, pop in enumerate(pops):
    scenario = 'pop-' + str(pop) + '-all-' + "{0:.0%}".format(portionDedicated)
    scenarioNames.append(scenario)

    model.scenarioData['populations'].Population = pop
    model.updatePopulation()

    scenarios[scenario] = gatherOutputs(model)

figs = []
axs = []
for i in range(8):
    fig, ax = plt.subplots(1, len(scenarioNames), figsize=(9.5, 3.75), sharey=True)
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]

    fig = figs[0]
    ax = axs[0]

    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['inflow'], color=colors['departures'],
                 label='Departures')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['outflow'], color=colors['arrivals'],
                 label='Arrivals')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles')
    ax[ind].set_title(title)
    ax[ind].set_ylim((0, 1200))

    fig = figs[1]
    ax = axs[1]
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['inflow'], axis=0),
                 color=colors['departures'],
                 label='Departures')
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['outflow'], axis=0),
                 color=colors['arrivals'],
                 label='Arrivals')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[2]
    ax = axs[2]
    ax[ind].plot(scenarios[scenario]['inflow'][1:], np.diff(scenarios[scenario]['inflow'], axis=0),
                 color=colors['departures'],
                 label='Departures')
    ax[ind].plot(scenarios[scenario]['outflow'][1:], np.diff(scenarios[scenario]['outflow'], axis=0),
                 color=colors['arrivals'],
                 label='Arrivals')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Number of vehicles')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[3]
    ax = axs[3]
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['accumulation'], color=colors['accumulation'])
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Vehicles')
    ax[ind].set_title(title)
    # ax[ind].set_ylim((0,220))

    fig = figs[4]
    ax = axs[4]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoSpeed_timePeriod'], color='C4',
                 label='Averaged auto')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['walkSpeed_timePeriod'], color='C2',
                 label='Averaged walk')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busSpeed_timePeriod'], color='C3',
                 label='Averaged bus')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['autoSpeed'], color='C4', label='Instantaneous auto',
                 linestyle='dashed')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Speed (m/s)')
    ax[ind].set_title(title)

    fig = figs[5]
    ax = axs[5]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoTravelTime_timePeriod'],
                 color=colors['autoTime'], label='Averaged auto')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busTravelTime_timePeriod'],
                 color=colors['transitTime'], label='Averaged bus')
    ax[ind].plot(scenarios[scenario]['time'][1:], scenarios[scenario]['autoTravelTime'], color=colors['autoTime'],
                 label='Instantaneous auto', linestyle='dashed')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Travel time (min)')
    ax[ind].set_title(title)
    # ax[ind].set_ylim((0,40))

    fig = figs[6]
    ax = axs[6]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['utils'], label='Averaged')
    ax[ind].set_xlabel('Time (h)')
    ax[ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
    ax[ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
    ax[ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(['Bus', 'Walk', 'Auto'], bbox_to_anchor=(1.05, 1), loc='upper left')
    if ind == 0:
        ax[ind].set_ylabel('Utility')
    ax[ind].set_title(title)

    fig = figs[7]
    ax = axs[7]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['modeSplit'])
    ax[ind].set_xlabel('Time (h)')
    ax[ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
    ax[ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
    ax[ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(['Bus', 'Walk', 'Auto'], bbox_to_anchor=(1.05, 1), loc='upper left')
    if ind == 0:
        ax[ind].set_ylabel('Mode split')
    ax[ind].set_title(title)

groupName = 'validationForQR/1-microtype-modes-' + "{0:.0%}".format(portionDedicated) + '-'

figs[0].savefig(groupName + 'queuing.pdf', bbox_inches='tight')
figs[1].savefig(groupName + 'inout-t.pdf', bbox_inches='tight')
figs[2].savefig(groupName + 'inout-n.pdf', bbox_inches='tight')
figs[3].savefig(groupName + 'accumulation.pdf', bbox_inches='tight')
figs[4].savefig(groupName + 'speed.pdf', bbox_inches='tight')
figs[5].savefig(groupName + 'travelTime.pdf', bbox_inches='tight')
figs[6].savefig(groupName + 'util.pdf', bbox_inches='tight')
figs[7].savefig(groupName + 'modeSplit.pdf', bbox_inches='tight')
plt.close('all')

# %%
portionDedicated = 0.25

busLaneDistance = portionDedicated * routeDistance

model.scenarioData['subNetworkData'].at[2, "Length"] = busLaneDistance
model.scenarioData['subNetworkData'].at[1, "Length"] = initialDistance - busLaneDistance
model.microtypes.updateNetworkData()

scenarioNames = []

for ind, pop in enumerate(pops):
    scenario = 'pop-' + str(pop) + '-all-' + "{0:.0%}".format(portionDedicated)
    scenarioNames.append(scenario)

    model.scenarioData['populations'].Population = pop
    model.updatePopulation()

    scenarios[scenario] = gatherOutputs(model)

figs = []
axs = []
for i in range(8):
    fig, ax = plt.subplots(1, len(scenarioNames), figsize=(9.5, 3.75), sharey=True)
    figs.append(fig)
    axs.append(ax)

for ind, scenario in enumerate(scenarioNames):
    title = titles[ind]

    fig = figs[0]
    ax = axs[0]

    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['inflow'], color=colors['departures'],
                 label='Departures')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['outflow'], color=colors['arrivals'],
                 label='Arrivals')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles')
    ax[ind].set_title(title)
    ax[ind].set_ylim((0, 1200))

    fig = figs[1]
    ax = axs[1]
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['inflow'], axis=0),
                 color=colors['departures'],
                 label='Departures')
    ax[ind].plot(scenarios[scenario]['time'][1:], np.diff(scenarios[scenario]['outflow'], axis=0),
                 color=colors['arrivals'],
                 label='Arrivals')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[2]
    ax = axs[2]
    ax[ind].plot(scenarios[scenario]['inflow'][1:], np.diff(scenarios[scenario]['inflow'], axis=0),
                 color=colors['departures'],
                 label='Departures')
    ax[ind].plot(scenarios[scenario]['outflow'][1:], np.diff(scenarios[scenario]['outflow'], axis=0),
                 color=colors['arrivals'],
                 label='Arrivals')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Number of vehicles')
    if ind == 0:
        ax[ind].set_ylabel('Number of vehicles per second')
    ax[ind].set_title(title)

    fig = figs[3]
    ax = axs[3]
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['accumulation'], color=colors['accumulation'])
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Vehicles')
    ax[ind].set_title(title)
    # ax[ind].set_ylim((0,220))

    fig = figs[4]
    ax = axs[4]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoSpeed_timePeriod'], color='C4',
                 label='Averaged auto')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['walkSpeed_timePeriod'], color='C2',
                 label='Averaged walk')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busSpeed_timePeriod'], color='C3',
                 label='Averaged bus')
    ax[ind].plot(scenarios[scenario]['time'], scenarios[scenario]['autoSpeed'], color='C4', label='Instantaneous auto',
                 linestyle='dashed')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Speed (m/s)')
    ax[ind].set_title(title)

    fig = figs[5]
    ax = axs[5]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['autoTravelTime_timePeriod'],
                 color=colors['autoTime'], label='Averaged auto')
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['busTravelTime_timePeriod'],
                 color=colors['transitTime'], label='Averaged bus')
    ax[ind].plot(scenarios[scenario]['time'][1:], scenarios[scenario]['autoTravelTime'], color=colors['autoTime'],
                 label='Instantaneous auto', linestyle='dashed')
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax[ind].set_xlabel('Time (h)')
    if ind == 0:
        ax[ind].set_ylabel('Travel time (min)')
    ax[ind].set_title(title)
    # ax[ind].set_ylim((0,40))

    fig = figs[6]
    ax = axs[6]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['utils'], label='Averaged')
    ax[ind].set_xlabel('Time (h)')
    ax[ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
    ax[ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
    ax[ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(['Bus', 'Walk', 'Auto'], bbox_to_anchor=(1.05, 1), loc='upper left')
    if ind == 0:
        ax[ind].set_ylabel('Utility')
    ax[ind].set_title(title)

    fig = figs[7]
    ax = axs[7]
    ax[ind].step(scenarios[scenario]['t_timePeriod'], scenarios[scenario]['modeSplit'])
    ax[ind].set_xlabel('Time (h)')
    ax[ind].lines[model.modeToIdx['auto']].set_color(colors['autoSplit'])
    ax[ind].lines[model.modeToIdx['bus']].set_color(colors['transitSplit'])
    ax[ind].lines[model.modeToIdx['walk']].set_color(colors['walkSplit'])
    if ind == len(scenarioNames) - 1:
        ax[ind].legend(['Bus', 'Walk', 'Auto'], bbox_to_anchor=(1.05, 1), loc='upper left')
    if ind == 0:
        ax[ind].set_ylabel('Mode split')
    ax[ind].set_title(title)

groupName = 'validationForQR/1-microtype-modes-' + "{0:.0%}".format(portionDedicated) + '-'

figs[0].savefig(groupName + 'queuing.pdf', bbox_inches='tight')
figs[1].savefig(groupName + 'inout-t.pdf', bbox_inches='tight')
figs[2].savefig(groupName + 'inout-n.pdf', bbox_inches='tight')
figs[3].savefig(groupName + 'accumulation.pdf', bbox_inches='tight')
figs[4].savefig(groupName + 'speed.pdf', bbox_inches='tight')
figs[5].savefig(groupName + 'travelTime.pdf', bbox_inches='tight')
figs[6].savefig(groupName + 'util.pdf', bbox_inches='tight')
figs[7].savefig(groupName + 'modeSplit.pdf', bbox_inches='tight')

print('done')
