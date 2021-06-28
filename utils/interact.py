from math import floor, log10

import ipywidgets as widgets
from ipywidgets import Layout
import matplotlib.pyplot as plt
import numpy as np
import plotly.express.colors as col
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Interact:
    def __init__(self, model):
        self.__model = model
        self.__colors = self.generateColorDict()
        self.__fig = go.FigureWidget(
            make_subplots(rows=7, cols=2, shared_yaxes=False, column_titles=['Current', 'Reference']))
        self.__modeToHandle = dict()
        self.__dataToHandle = dict()
        self.addBlankPlots(self.__fig)
        self.copyCurrentToRef()
        self.__microtypeToMixedNetworkID = dict()
        self.__microtypeToBusNetworkID = dict()
        self.__microtypeToBusService = dict()
        self.__widgetIDtoField = dict()
        self.__plotStateWidget = None
        self.__loadingWidget = None
        self.__out = None  # print(*a, file = sys.stdout)
        self.__grid = self.generateGridSpec()

    def generateColorDict(self):
        modes = col.qualitative.Bold
        microtypes = col.qualitative.Safe
        costs = col.qualitative.Vivid
        out = {'bus': modes[0],
               'auto': modes[1],
               'walk': modes[5],
               'bike': modes[3],
               'rail': modes[4],
               'A': microtypes[0],
               'B': microtypes[1],
               'C': microtypes[2],
               'D': microtypes[3],
               '1': microtypes[0],
               '2': microtypes[1],
               '3': microtypes[2],
               '4': microtypes[3],
               '5': microtypes[4],
               '6': microtypes[5],
               '7': microtypes[6],
               '8': microtypes[7],
               'user': costs[1],
               'system': costs[1],
               'dedication': costs[1]}
        return out

    @property
    def colors(self):
        return self.__colors

    @property
    def model(self):
        return self.__model

    @property
    def data(self):
        return self.__fig.data[0]

    @property
    def layout(self):
        return self.__fig.layout

    @property
    def fig(self):
        return self.__fig

    @property
    def grid(self):
        return self.__grid

    @property
    def out(self):
        return self.__out

    def addBlankPlots(self, fig: go.FigureWidget):
        fig.update_layout(
            autosize=False,
            width=900,
            height=1500, )
        nMicrotypes = len(self.model.scenarioData['microtypeIDs'].MicrotypeID)
        self.__dataToHandle['speed'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['modeSplit'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['modeSpeed'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['cost'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['costDiff'] = dict()
        self.__dataToHandle['modeSplitDiff'] = dict()
        self.__dataToHandle['speedDiff'] = dict()
        for mID in self.model.scenarioData['microtypeIDs'].MicrotypeID:
            fig.add_scatter(x=[], y=[], visible=True, name='Microtype ' + mID, row=1, col=1, legendgroup="Speed",
                            mode='lines')
            self.__dataToHandle['speed']['current'][mID] = fig.data[-1]
            fig.data[-1].line = {"color": self.colors[mID]}
            fig.add_scatter(x=[], y=[], visible=True, name='Microtype ' + mID, row=1, col=2, legendgroup="Speed",
                            mode='lines', showlegend=False)
            self.__dataToHandle['speed']['ref'][mID] = fig.data[-1]
            fig.data[-1].line = {"color": self.colors[mID]}
            fig.add_scatter(x=[], y=[], visible=True, name='Microtype ' + mID, row=5, col=1, legendgroup="Speed",
                            mode='lines', showlegend=False)
            self.__dataToHandle['speedDiff'][mID] = fig.data[-1]
            fig.data[-1].line = {"color": self.colors[mID]}
        for mode in self.model.scenarioData['modeData'].keys():
            fig.add_scatter(x=[], y=[], visible=True, name=mode, row=2, col=1, legendgroup="Mode Split", mode='lines')
            self.__dataToHandle['modeSplit']['current'][mode] = fig.data[-1]
            fig.data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
            fig.add_scatter(x=[], y=[], visible=True, name=mode, row=2, col=2, legendgroup="Mode Split", mode='lines',
                            showlegend=False)
            self.__dataToHandle['modeSplit']['ref'][mode] = fig.data[-1]
            fig.data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
            fig.add_scatter(x=[], y=[], visible=True, name=mode, row=6, col=1, legendgroup="Mode Split", mode='lines',
                            showlegend=False)
            self.__dataToHandle['modeSplitDiff'][mode] = fig.data[-1]
            fig.data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
        for mode in self.model.scenarioData['modeData'].keys():
            self.__dataToHandle['modeSpeed']['current'][mode] = dict()
            self.__dataToHandle['modeSpeed']['ref'][mode] = dict()
            showLegend = True
            for mID in self.model.scenarioData['microtypeIDs'].MicrotypeID:
                fig.add_scatter(x=[], y=[], visible=True, name=mode, row=3, col=1, legendgroup=mode, mode='lines',
                                hovertext="Microtype " + mID + " " + mode, hoverinfo="text", showlegend=showLegend)
                self.__dataToHandle['modeSpeed']['current'][mode][mID] = fig.data[-1]
                fig.data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
                fig.add_scatter(x=[], y=[], visible=True, name=mode, row=3, col=2, legendgroup=mode, mode='lines',
                                showlegend=False, hovertext="Microtype " + mID + " " + mode, hoverinfo="text")
                self.__dataToHandle['modeSpeed']['ref'][mode][mID] = fig.data[-1]
                fig.data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
                showLegend = False
        for mID in self.model.scenarioData['microtypeIDs'].MicrotypeID:
            fig.add_bar(x=['User', 'Operator', 'Lane dedication'], y=[0.] * 3, visible=True, row=4, col=1,
                        name='Microtype ' + mID, legendgroup="Costs")
            self.__dataToHandle['cost']['current'][mID] = fig.data[-1]
            fig.data[-1].marker.color = self.colors[mID]
            fig.add_bar(x=['User', 'Operator', 'Lane dedication'], y=[0.] * 3, visible=True, row=4, col=2,
                        name='Microtype ' + mID, legendgroup="Costs", showlegend=False)
            self.__dataToHandle['cost']['ref'][mID] = fig.data[-1]
            fig.data[-1].marker.color = self.colors[mID]
            fig.add_bar(x=['User', 'Operator', 'Lane dedication'], y=[0.] * 3, visible=True, row=7, col=1,
                        name='Microtype ' + mID, legendgroup="Costs", showlegend=False)
            self.__dataToHandle['costDiff'][mID] = fig.data[-1]
            fig.data[-1].marker.color = self.colors[mID]
        # fig.update_layout(barmode='stack')
        fig.update_layout(template='simple_white')
        fig['layout']['xaxis']['title'] = 'Time (hr)'
        fig['layout']['yaxis']['title'] = 'Auto speed (m/s)'
        fig['layout']['xaxis2']['title'] = 'Time (hr)'
        fig['layout']['xaxis3']['title'] = 'Time (hr)'
        fig['layout']['yaxis3']['title'] = 'Mode split'
        fig['layout']['xaxis4']['title'] = 'Time (hr)'
        fig['layout']['yaxis5']['title'] = 'Mode speed (m/s)'
        fig['layout']['xaxis6']['title'] = 'Time (hr)'
        fig['layout']['yaxis7']['title'] = 'Cost'
        fig['layout']['yaxis9']['title'] = 'Difference in auto speed'
        fig['layout']['xaxis9']['title'] = 'Time (hr)'
        fig['layout']['yaxis11']['title'] = 'Difference in mode split'
        fig['layout']['xaxis11']['title'] = 'Time (hr)'
        fig['layout']['yaxis13']['title'] = 'Difference in cost'

    def generateGridSpec(self):
        rerunModel = widgets.Button(description="Calculate Costs",
                                    tooltip="Click to run the model with your given inputs",
                                    layout=Layout(width='100%', height='80%', justify_content='center'))
        rerunModel.on_click(self.updateCosts)

        setRef = widgets.Button(description="Update reference",
                                tooltip='Click to update reference plots on right',
                                layout=Layout(width='100%', height='80%'))
        setRef.on_click(self.copyCurrentToRef)

        populationStack = []
        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            microtypePopulations = [widgets.HTML(
                value="<center><b>Microtype " + mID + "</b></center>"
            )]
            sub = self.model.scenarioData['populations'].loc[self.model.scenarioData['populations'].MicrotypeID == mID,
                  :]
            upperBound = sub.Population.max() * 1.5
            upperBound = round(upperBound, 3 - int(floor(log10(
                abs(upperBound)))) - 1)  # https://www.kite.com/python/answers/how-to-round-a-number-to-significant-digits-in-python
            popVBox = []
            for row in sub.itertuples():
                popVBox.append(widgets.IntSlider(row.Population, 0, upperBound, upperBound / 100,
                                                 description=row.PopulationGroupTypeID,
                                                 orientation='horizontal'))
                popVBox[-1].observe(self.response, names="value")
                self.__widgetIDtoField[popVBox[-1].model_id] = ('population', (mID, row.PopulationGroupTypeID))
            microtypePopulations.append(widgets.VBox(popVBox))
            populationStack.append(widgets.HBox(microtypePopulations))

        dedicatedStack = []

        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            initialAutoData = self.model.scenarioData['subNetworkDataFull'].loc[
                              self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.contains('Auto') &
                              (self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID), :]
            self.__microtypeToMixedNetworkID[mID] = initialAutoData

            initialBusData = self.model.scenarioData['subNetworkDataFull'].loc[
                             self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.contains('Bus') &
                             (self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID) &
                             self.model.scenarioData['subNetworkDataFull'].Dedicated, :]
            self.__microtypeToBusNetworkID[mID] = initialBusData

            dedicatedStack.append(
                widgets.FloatSlider(value=0, min=0, max=1.0, step=0.02, description="Microtype " + mID))
            dedicatedStack[-1].observe(self.response, names="value")
            self.__widgetIDtoField[dedicatedStack[-1].model_id] = ('dedicated', mID)

        headwayStack = []

        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            busServiceData = self.model.scenarioData['modeData']['bus'].loc[mID]
            self.__microtypeToBusService[mID] = busServiceData
            headwayStack.append(widgets.IntSlider(busServiceData.Headway, 90, 1800, 30, description="Microtype " + mID))
            headwayStack[-1].observe(self.response, names="value")
            self.__widgetIDtoField[headwayStack[-1].model_id] = ('headway', mID)

        coverageStack = []

        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            busServiceData = self.__microtypeToBusService[mID]
            coverageStack.append(widgets.FloatSlider(value=busServiceData.CoveragePortion, min=0.02, max=1.0, step=0.02,
                                                     description="Microtype " + mID))
            coverageStack[-1].observe(self.response, names="value")
            self.__widgetIDtoField[coverageStack[-1].model_id] = ('headway', mID)

        accordionChildren = [widgets.VBox(populationStack), widgets.VBox(dedicatedStack), widgets.VBox(headwayStack),
                             widgets.VBox(coverageStack)]

        accordion = widgets.Accordion(children=accordionChildren)

        for ind, title in enumerate(
                ('Population by group', 'Bus lane dedication', 'Bus headways (s)', 'Bus service area')):
            accordion.set_title(ind, title)

        gs = widgets.GridspecLayout(4, 2)

        self.__loadingWidget = widgets.HTML(
            value="<center><i>Model Running</i></center>"
        )

        self.__plotStateWidget = widgets.Dropdown(
            options=['Auto speed', 'Mode split', 'Auto accumulation'],
            value='Auto speed',
            description='Plot type:',
            disabled=False,
        )
        self.__plotStateWidget.observe(self.updatePlots, names="value")

        hardReset = widgets.Button(description="Reset model state",
                                   tooltip="Click if everything is broken",
                                   layout=Layout(width='100%', height='80%'))
        hardReset.on_click(self.hardReset)

        gs[:, 0] = accordion
        gs[0, 1] = rerunModel
        gs[2, 1] = self.__loadingWidget
        self.__out = widgets.Output(layout={'border': '1px solid black'})
        gs[1, 1] = setRef
        gs[3, 1] = hardReset
        return gs

    def response(self, change, otherStuff=None):
        field = self.__widgetIDtoField[change.owner.model_id]
        self.modifyModel(field, change)

    def modifyModel(self, changeType, value):
        if changeType[0] == 'dedicated':
            df = self.returnBusNetworkLengths(changeType[1])
            totalLength = df.sum()
            newDedicatedLength = totalLength * value.new
            newMixedLength = totalLength * (1. - value.new)
            self.model.scenarioData['subNetworkData'].loc[df.index[0], 'Length'] = newMixedLength
            self.model.scenarioData['subNetworkData'].loc[df.index[1], 'Length'] = newDedicatedLength
        if changeType[0] == 'headway':
            self.model.scenarioData['modeData']['bus'].loc[changeType[1], 'Headway'] = value.new
        if changeType[0] == 'coverage':
            self.model.scenarioData['modeData']['bus'].loc[changeType[1], 'CoveragePortion'] = value.new
            self.model.readFiles()
        if changeType[0] == 'population':
            mask = (self.model.scenarioData['populations']['MicrotypeID'] == changeType[1][0]) & (
                    self.model.scenarioData['populations']['PopulationGroupTypeID'] == changeType[1][1])
            if sum(mask) == 1:
                self.model.scenarioData['populations'].loc[mask, 'Population'] = value.new
                self.model.updatePopulation()

    def returnBusNetworkLengths(self, mID):
        return self.model.scenarioData['subNetworkDataFull'].loc[
            self.model.scenarioData['subNetworkDataFull'].ModesAllowed.str.contains('Bus') & (
                    self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID), 'Length']

    def plotArray(self):
        x, y = self.model.plotAllDynamicStats("delay")
        fig, axs = plt.subplots(4, len(self.model.microtypes), figsize=(8., 6.), dpi=200)
        for ind, m in enumerate(self.model.microtypes):
            y1 = y[0, :, ind]
            y2 = y[1, :, ind]
            axs[0, ind].plot(x, y1, color="#800080")
            axs[0, ind].plot(x, y2, color="#00DBFF")
            axs[1, ind].plot(x, y1 - y2, color="#E56717")
            axs[2, ind].plot(x[:-1], np.interp(y1, y2, x)[:-1] / 60. - x[:-1] / 60., '#ED4337')
            axs[0, ind].set_title("Microtype " + m[0])

            axs[3, ind].clear()
            axs[3, ind].step(np.arange(len(self.model.timePeriods()) + 1), np.vstack(
                [self.model.getModeSplit('0', microtypeID=m[0])] + [self.model.getModeSplit(p, microtypeID=m[0]) for p
                                                                    in self.model.timePeriods().keys()]))
            axs[3, ind].set_ylim([0, 1])

            axs[3, ind].lines[self.model.modeToIdx['auto']].set_color('#C21807')
            axs[3, ind].lines[self.model.modeToIdx['bus']].set_color('#1338BE')
            axs[3, ind].lines[self.model.modeToIdx['walk']].set_color('#3CB043')
            axs[3, ind].lines[self.model.modeToIdx['rail']].set_color('orange')
            axs[3, ind].lines[self.model.modeToIdx['bike']].set_color('blue')

        axs[3, 0].legend(['bus', 'rail', 'walk', 'bike', 'auto'])
        axs[0, 0].set_ylabel('cumulative vehicles')
        axs[1, 0].set_ylabel('accumulation')
        axs[2, 0].set_ylabel('travel time')
        axs[3, 0].set_ylabel('mode split')

    def updateCosts(self, message=None):
        self.__loadingWidget.value = "<center><i>Model Running</i></center>"
        self.model.collectAllCharacteristics()
        self.updatePlots()
        self.__loadingWidget.value = "<center><b>Complete</b></center>"

    def updatePlots(self, message=None):
        time, spds = self.model.plotAllDynamicStats('v')
        for ind, (mode, handle) in enumerate(self.__dataToHandle['speed']['current'].items()):
            handle.y = spds[:, ind]
            handle.x = time
            handle.visible = True

        time, splits = self.model.plotAllDynamicStats('modes')
        for ind, (mode, handle) in enumerate(self.__dataToHandle['modeSplit']['current'].items()):
            handle.y = splits[:, ind]
            handle.x = time
            handle.visible = True

        time, modeSpeeds = self.model.plotAllDynamicStats('modeSpeeds')
        for mode, group in self.__dataToHandle['modeSpeed']['current'].items():
            for mID, handle in group.items():
                handle.y = modeSpeeds[(mID, mode)].values
                handle.x = time
                handle.visible = True

        mIDs, costs = self.model.plotAllDynamicStats('costs')
        for ind, (mID, handle) in enumerate(self.__dataToHandle['cost']['current'].items()):
            handle.y = costs[mID].values

        for mID, plot in self.__dataToHandle['costDiff'].items():
            yRef = np.array(self.__dataToHandle['cost']['ref'][mID].y)
            yCurrent = np.array(self.__dataToHandle['cost']['current'][mID].y)
            if len(yRef) == 0:
                plot.y = yCurrent * 0.0
            else:
                plot.y = yCurrent - yRef

        for mode, plot in self.__dataToHandle['modeSplitDiff'].items():
            yRef = np.array(self.__dataToHandle['modeSplit']['ref'][mode].y)
            yCurrent = np.array(self.__dataToHandle['modeSplit']['current'][mode].y)
            if len(yRef) == 0:
                plot.y = yCurrent * 0.0
            else:
                plot.y = yCurrent - yRef
            plot.x = self.__dataToHandle['modeSplit']['current'][mode].x

        for mID, plot in self.__dataToHandle['speedDiff'].items():
            yRef = np.array(self.__dataToHandle['speed']['ref'][mID].y)
            yCurrent = np.array(self.__dataToHandle['speed']['current'][mID].y)
            if len(yRef) == 0:
                plot.y = yCurrent * 0.0
            else:
                plot.y = yCurrent - yRef
            plot.x = self.__dataToHandle['speed']['current'][mID].x

    def copyCurrentToRef(self, message=None):
        for plotType, plots in self.__dataToHandle.items():
            if plotType.endswith("Diff"):
                continue
            elif plotType == "modeSpeed":
                for mode, group in plots['current'].items():
                    for line, value in group.items():
                        plots['ref'][mode][line].y = value.y
                        plots['ref'][mode][line].x = value.x
            else:
                for line, value in plots['current'].items():
                    plots['ref'][line].y = value.y
                    plots['ref'][line].x = value.x
        for plotType, plots in self.__dataToHandle.items():
            if plotType == "costDiff":
                for line in plots.keys():
                    # yRef = np.array(self.__dataToHandle['cost']['ref'][line].y)
                    # yCurrent = np.array(self.__dataToHandle['cost']['current'][line].y)
                    self.__dataToHandle['costDiff'][line].y = [0] * len(self.__dataToHandle['cost']['current'][line].y)
            if plotType == "modeSplitDiff":
                for line in plots.keys():
                    # yRef = np.array(self.__dataToHandle['cost']['ref'][line].y)
                    # yCurrent = np.array(self.__dataToHandle['cost']['current'][line].y)
                    self.__dataToHandle['modeSplitDiff'][line].y = [0] * len(
                        self.__dataToHandle['modeSplit']['current'][line].y)
            if plotType == "speedDiff":
                for line in plots.keys():
                    # yRef = np.array(self.__dataToHandle['cost']['ref'][line].y)
                    # yCurrent = np.array(self.__dataToHandle['cost']['current'][line].y)
                    self.__dataToHandle['speedDiff'][line].y = [0] * len(
                        self.__dataToHandle['speed']['current'][line].y)

    def init(self):
        self.updateCosts()
        self.copyCurrentToRef()

    def hardReset(self, message=None):
        self.model.scenarioData.loadData()
        self.model.scenarioData.loadModeData()
        self.model.resetNetworks()
        self.model.readFiles()
        self.updateCosts()
