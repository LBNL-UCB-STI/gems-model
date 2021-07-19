from math import floor, log10

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import plotly.express.colors as col
import plotly.graph_objects as go
from ipywidgets import Layout
from plotly.subplots import make_subplots


class Interact:
    def __init__(self, model):
        self.__model = model
        self.__colors = self.generateColorDict()
        self.__paramNames = self.generateParamDict()
        self.__fig = widgets.Accordion(children=[])  # widgets.VBox([])
        self.__modeToHandle = dict()
        self.__dataToHandle = dict()
        self.addBlankPlots(self.__fig)
        self.copyCurrentToRef()
        self.__microtypeToMixedNetworkID = dict()
        self.__microtypeToBusNetworkID = dict()
        self.__microtypeToBusService = dict()
        self.__widgetIDtoSubNetwork = dict()
        self.__widgetIDtoField = dict()
        self.__utilDropdownStatus = dict()
        self.__widgetIDtoUtil = dict()
        self.__plotStateWidget = None
        self.__loadingWidget = None
        self.__out = None  # print(*a, file = sys.stdout)
        self.__grid = self.generateGridSpec()

    def generateParamDict(self):
        out = {'Constant': 'intercept',
               'Wait time': 'wait_time',
               'In vehicle time': 'travel_time',
               'Access time': 'access_time'}
        return out

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

    def addBlankPlots(self, figContainer: widgets.Accordion):
        # fig = go.FigureWidget(
        #     make_subplots(rows=7, cols=2, shared_yaxes=False, column_titles=['Current', 'Reference']))
        # figContainer.children = [fig]
        #
        # fig.update_layout(
        #     autosize=False,
        #     width=900,
        #     height=1500, )
        microtypeFigs = []
        diffFigs = []

        for mID in self.model.scenarioData['microtypeIDs'].MicrotypeID:
            currentMicrotypeFig = go.FigureWidget(
                make_subplots(rows=4, cols=2,
                              shared_yaxes=True,
                              column_titles=['Current', 'Reference'])
            )

            currentMicrotypeFig.update_layout(
                {'autosize': False, 'width': 900, 'height': 900, 'template': 'simple_white'})

            currentMicrotypeFig['layout']['xaxis']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['yaxis']['title'] = 'Mode speed (m/s)'
            currentMicrotypeFig['layout']['xaxis2']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['xaxis3']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['yaxis3']['title'] = 'Mode split'
            currentMicrotypeFig['layout']['xaxis4']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['yaxis5']['title'] = 'Auto speed (m/s)'
            currentMicrotypeFig['layout']['xaxis6']['title'] = 'Time (hr)'
            currentMicrotypeFig['layout']['yaxis7']['title'] = 'Cost'

            microtypeFigs.append(currentMicrotypeFig)

            currentDiffFig = go.FigureWidget(
                make_subplots(rows=2, cols=1,
                              shared_yaxes=True)
            )

            currentDiffFig['layout']['yaxis']['title'] = 'Difference in mode split'
            currentDiffFig['layout']['yaxis2']['title'] = 'Difference in auto speed (m/s)'
            currentDiffFig['layout']['xaxis2']['title'] = 'Time (hr)'

            currentDiffFig.update_layout(
                {'autosize': False, 'width': 900, 'height': 600, 'template': 'simple_white'})

            diffFigs.append(currentDiffFig)

        combinedCostDiffFig = go.FigureWidget(go.Figure())
        combinedCostDiffFig['layout']['yaxis']['title'] = 'Difference in cost'
        combinedCostDiffFig.update_layout(template='simple_white')

        self.__dataToHandle['speed'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['modeSplit'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['modeSpeed'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['cost'] = {'current': dict(), 'ref': dict()}
        self.__dataToHandle['costDiff'] = dict()
        self.__dataToHandle['modeSplitDiff'] = dict()
        self.__dataToHandle['speedDiff'] = dict()
        for idx, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name='Microtype ' + mID, row=3, col=1,
                                           showlegend=False,
                                           mode='lines')
            self.__dataToHandle['speed']['current'][mID] = microtypeFigs[idx].data[-1]
            microtypeFigs[idx].data[-1].line = {"color": self.colors[mID]}
            microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name='Microtype ' + mID, row=3, col=2,
                                           mode='lines', showlegend=False)
            self.__dataToHandle['speed']['ref'][mID] = microtypeFigs[idx].data[-1]
            microtypeFigs[idx].data[-1].line = {"color": self.colors[mID]}
            diffFigs[idx].add_scatter(x=[], y=[], visible=True, name='Microtype ' + mID, row=2, col=1,
                                      legendgroup="Speed",
                                      mode='lines', showlegend=False)
            self.__dataToHandle['speedDiff'][mID] = diffFigs[idx].data[-1]
            diffFigs[idx].data[-1].line = {"color": self.colors[mID]}
        for mode in self.model.scenarioData['modeData'].keys():
            self.__dataToHandle['modeSplit']['current'][mode] = dict()
            self.__dataToHandle['modeSplit']['ref'][mode] = dict()
            self.__dataToHandle['modeSplitDiff'][mode] = dict()
            for idx, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
                microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=2, col=1, legendgroup=mode,
                                               mode='lines', showlegend=False)
                self.__dataToHandle['modeSplit']['current'][mode][mID] = microtypeFigs[idx].data[-1]
                microtypeFigs[idx].data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
                microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=2, col=2, legendgroup=mode,
                                               mode='lines',
                                               showlegend=False)
                self.__dataToHandle['modeSplit']['ref'][mode][mID] = microtypeFigs[idx].data[-1]
                microtypeFigs[idx].data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
                diffFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=1, col=1, legendgroup=mode,
                                          mode='lines',
                                          showlegend=True)
                self.__dataToHandle['modeSplitDiff'][mode][mID] = diffFigs[idx].data[-1]
                diffFigs[idx].data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
        for mode in self.model.scenarioData['modeData'].keys():
            self.__dataToHandle['modeSpeed']['current'][mode] = dict()
            self.__dataToHandle['modeSpeed']['ref'][mode] = dict()
            for idx, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
                microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=1, col=1, mode='lines',
                                               hovertext="Microtype " + mID + " " + mode, hoverinfo="text",
                                               showlegend=True, legendgroup=mode)
                self.__dataToHandle['modeSpeed']['current'][mode][mID] = microtypeFigs[idx].data[-1]
                microtypeFigs[idx].data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
                microtypeFigs[idx].add_scatter(x=[], y=[], visible=True, name=mode, row=1, col=2, mode='lines',
                                               showlegend=False, hovertext="Microtype " + mID + " " + mode,
                                               hoverinfo="text", legendgroup=mode)
                self.__dataToHandle['modeSpeed']['ref'][mode][mID] = microtypeFigs[idx].data[-1]
                microtypeFigs[idx].data[-1].line = {"shape": 'hv', "color": self.colors[mode]}
        for idx, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            microtypeFigs[idx].add_bar(x=['User', 'Operator', 'Lane dedication'], y=[0.] * 3, visible=True, row=4,
                                       col=1,
                                       name='Microtype ' + mID, legendgroup="Costs", showlegend=False)
            self.__dataToHandle['cost']['current'][mID] = microtypeFigs[idx].data[-1]
            microtypeFigs[idx].data[-1].marker.color = self.colors[mID]
            microtypeFigs[idx].add_bar(x=['User', 'Operator', 'Lane dedication'], y=[0.] * 3, visible=True, row=4,
                                       col=2,
                                       name='Microtype ' + mID, legendgroup="Costs", showlegend=False)
            self.__dataToHandle['cost']['ref'][mID] = microtypeFigs[idx].data[-1]
            microtypeFigs[idx].data[-1].marker.color = self.colors[mID]
            combinedCostDiffFig.add_bar(x=['User', 'Operator', 'Lane dedication'], y=[0.] * 3, visible=True,
                                        name='Microtype ' + mID, showlegend=True)
            self.__dataToHandle['costDiff'][mID] = combinedCostDiffFig.data[-1]
            combinedCostDiffFig.data[-1].marker.color = self.colors[mID]

        figContainer.children = microtypeFigs + diffFigs + [combinedCostDiffFig]

        tabTitles = ["Microtype " + mID + ": Outcomes" for mID in self.model.microtypeIdToIdx.keys()] + [
            "Microtype " + mID + ": Change from reference" for mID in self.model.microtypeIdToIdx.keys()] + [
                        "Change in costs"]

        for ind, title in enumerate(tabTitles):
            figContainer.set_title(ind, title)

    def generateGridSpec(self):
        rerunModel = widgets.Button(description="Calculate Costs",
                                    tooltip="Click to run the model with your given inputs",
                                    layout=Layout(width='100%', height='0.5in', justify_content='center'))
        rerunModel.on_click(self.updateCosts)

        setRef = widgets.Button(description="Update reference",
                                tooltip='Click to update reference plots on right',
                                layout=Layout(width='100%', height='0.5in'))
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

        utilStack = []
        allTripTypes = self.model.scenarioData['populationGroups'].TripPurposeID.unique()
        modes = list(self.model.modeToIdx.keys())
        for groupID in self.model.scenarioData['populationGroups'].PopulationGroupTypeID.unique():
            self.__utilDropdownStatus[groupID] = dict()
            label = widgets.HTML(
                value="<center><b>Population group</b><br>" + groupID + "</center>"
            )
            tripTypeDropdown = widgets.Dropdown(
                options=allTripTypes,
                value=allTripTypes[0],
                description='Trip purpose:',
                disabled=False,
                layout={'vertical_align': 'middle', 'width': '2.8in'},
                style={'description_width': '1.25in'}
            )
            self.__utilDropdownStatus[groupID]['tripType'] = tripTypeDropdown
            self.__widgetIDtoUtil[tripTypeDropdown.model_id] = ('tripType', groupID)
            tripTypeDropdown.observe(self.changeUtilDropdown)
            modeDropdown = widgets.Dropdown(
                options=modes,
                value=modes[0],
                description='Mode:',
                layout={'vertical_align': 'middle', 'width': '2.8in'},
                style={'description_width': '1.25in'}
            )
            self.__utilDropdownStatus[groupID]['mode'] = modeDropdown
            self.__widgetIDtoUtil[modeDropdown.model_id] = ('mode', groupID)
            modeDropdown.observe(self.changeUtilDropdown)
            paramDropdown = widgets.Dropdown(
                options=['Constant', 'Wait time', 'In vehicle time', 'Access time'],
                value='Constant',
                description='Utility param:',
                layout={'vertical_align': 'middle', 'width': '2.8in'},
                style={'description_width': '1.25in'}
            )
            self.__utilDropdownStatus[groupID]['param'] = paramDropdown
            self.__widgetIDtoUtil[paramDropdown.model_id] = ('param', groupID)
            paramDropdown.observe(self.changeUtilDropdown)
            valueBox = widgets.FloatText(value=0.0, step=0.05, description="Value: ", disabled=False,
                                         layout={'vertical_align': 'middle', 'width': '1.8in'})
            self.__utilDropdownStatus[groupID]['value'] = valueBox
            self.__widgetIDtoUtil[valueBox.model_id] = ('value', groupID)
            valueBox.observe(self.changeUtilValue)
            utilStack.append(
                widgets.HBox([label, widgets.VBox([tripTypeDropdown, modeDropdown, paramDropdown]), valueBox]))

        MFDstack = []
        for ind, mID in enumerate(self.model.scenarioData['microtypeIDs'].MicrotypeID):
            initialAutoData = self.model.scenarioData['subNetworkDataFull'].loc[
                              (self.model.scenarioData['subNetworkDataFull'].Type == 'Road') &
                              (self.model.scenarioData['subNetworkDataFull'].MicrotypeID == mID), :]
            microtypeRoadNetworks = [widgets.HTML(
                value="<center><b>Microtype " + mID + "</b></center>"
            )]
            autoVBox = []
            for row in initialAutoData.itertuples():
                roadNetworkParameters = [widgets.HTML(
                    value="<center><i>" + row.ModesAllowed + "</i></center>",
                    layout=Layout(width='1.25in', align_content='center'))]
                parameterVBox = []
                parameterVBox.append(widgets.FloatSlider(value=row.vMax, min=0, max=25, step=0.5,
                                                         description="Max speed (m/s)",
                                                         orientation='horizontal',
                                                         style={'description_width': '1.25in'})
                                     )
                parameterVBox[-1].observe(self.response, names="value")
                self.__widgetIDtoField[parameterVBox[-1].model_id] = ('vMax', row.Index)
                parameterVBox.append(widgets.FloatSlider(value=row.densityMax, min=0.1, max=0.2, step=0.002,
                                                         description="Jam density (veh/m)",
                                                         orientation='horizontal',
                                                         style={'description_width': '1.25in'}))
                parameterVBox[-1].observe(self.response, names="value")
                self.__widgetIDtoField[parameterVBox[-1].model_id] = ('densityMax', row.Index)
                roadNetworkParameters.append(widgets.VBox(parameterVBox))
                autoVBox.append(widgets.HBox(roadNetworkParameters))
            microtypeRoadNetworks.append(widgets.VBox(autoVBox))
            MFDstack.append(widgets.HBox(microtypeRoadNetworks))

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

        dataAccordion = widgets.Accordion(
            [widgets.VBox(populationStack), widgets.VBox(utilStack), widgets.VBox(MFDstack)])
        for ind, title in enumerate(('Population', 'Utility parameters', 'MFD parameters')):
            dataAccordion.set_title(ind, title)

        scenarioAccordion = widgets.Accordion(
            [widgets.VBox(dedicatedStack), widgets.VBox(headwayStack), widgets.VBox(coverageStack)])

        for ind, title in enumerate(('Bus lane dedication', 'Bus headway (s)', 'Bus service area')):
            scenarioAccordion.set_title(ind, title)

        accordionChildren = [dataAccordion, scenarioAccordion]

        accordion = widgets.Accordion(children=accordionChildren)
        for ind, title in enumerate(('Input data', 'Scenario parameters')):
            accordion.set_title(ind, title)

        # for ind, title in enumerate(
        #         ('Population by group', 'MFD Parameters', 'Bus lane dedication', 'Bus headways (s)',
        #          'Bus service area')):
        #     accordion.set_title(ind, title)

        gs = widgets.GridspecLayout(2, 3)

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

        # hardReset = widgets.Button(description="Reset model state",
        #                            tooltip="Click if everything is broken",
        #                            layout=Layout(width='100%', height='0.5in'))
        # hardReset.on_click(self.hardReset)

        gs[:, :2] = accordion
        # gs[0, 2] = rerunModel
        gs[0, 2] = self.__loadingWidget
        self.__out = widgets.Output(layout={'border': '1px solid black'})
        # gs[1, 2] = setRef
        gs[1, 2] = widgets.VBox([rerunModel, setRef])
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
        if changeType[0] == 'vMax':
            self.model.scenarioData['subNetworkData'].loc[changeType[1], 'vMax'] = value.new
        if changeType[0] == 'densityMax':
            self.model.scenarioData['subNetworkData'].loc[changeType[1], 'densityMax'] = value.new

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

        microtypeModeSplits = dict()
        for mID in self.model.microtypeIdToIdx.keys():
            time, microtypeModeSplits[mID] = self.model.plotAllDynamicStats('modes', microtype=mID)

        for modeID, (mode, group) in enumerate(self.__dataToHandle['modeSplit']['current'].items()):
            for mID, handle in group.items():
                handle.y = microtypeModeSplits[mID][:, modeID]
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

        for mode, microtypeDict in self.__dataToHandle['modeSplitDiff'].items():
            for mID, plot in microtypeDict.items():
                yRef = np.array(self.__dataToHandle['modeSplit']['ref'][mode][mID].y)
                yCurrent = np.array(self.__dataToHandle['modeSplit']['current'][mode][mID].y)
                if len(yRef) == 0:
                    plot.y = yCurrent * 0.0
                else:
                    plot.y = yCurrent - yRef
                plot.x = self.__dataToHandle['modeSplit']['current'][mode][mID].x

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
            elif (plotType == "modeSpeed") | (plotType == "modeSplit"):
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
                for mode, group in plots.items():
                    for mID in group.keys():
                        # yRef = np.array(self.__dataToHandle['cost']['ref'][line].y)
                        # yCurrent = np.array(self.__dataToHandle['cost']['current'][line].y)
                        self.__dataToHandle['modeSplitDiff'][mode][mID].y = [0] * len(
                            self.__dataToHandle['modeSplit']['current'][mode][mID].y)
            if plotType == "speedDiff":
                for line in plots.keys():
                    # yRef = np.array(self.__dataToHandle['cost']['ref'][line].y)
                    # yCurrent = np.array(self.__dataToHandle['cost']['current'][line].y)
                    self.__dataToHandle['speedDiff'][line].y = [0] * len(
                        self.__dataToHandle['speed']['current'][line].y)

    def changeUtilDropdown(self, message=None):
        menu, popGroup = self.__widgetIDtoUtil[message.owner.model_id]
        tripPurpose = self.__utilDropdownStatus[popGroup]['tripType'].value
        mode = self.__utilDropdownStatus[popGroup]['mode'].value
        param = self.__paramNames[self.__utilDropdownStatus[popGroup]['param'].value]
        mIDs, vals = self.model.getUtilityParam(param, popGroup, tripPurpose, mode)
        self.__utilDropdownStatus[popGroup]['value'].value = vals[0]

    def changeUtilValue(self, message=None):
        menu, popGroup = self.__widgetIDtoUtil[message.owner.model_id]
        tripPurpose = self.__utilDropdownStatus[popGroup]['tripType'].value
        mode = self.__utilDropdownStatus[popGroup]['mode'].value
        param = self.__paramNames[self.__utilDropdownStatus[popGroup]['param'].value]
        if isinstance(message.new, dict):
            if 'value' in message.new:
                val = message.new['value']
                self.model.updateUtilityParam(val, param, popGroup, tripPurpose, mode)
        elif type(message.new) == int or float:
            val = message.new
            self.model.updateUtilityParam(val, param, popGroup, tripPurpose, mode)


    def init(self):
        self.updateCosts()
        self.copyCurrentToRef()

    def hardReset(self, message=None):
        self.model.scenarioData.loadData()
        self.model.scenarioData.loadModeData()
        self.model.resetNetworks()
        self.model.readFiles()
        self.updateCosts()
