import pandas as pd
import os
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags
from scipy.stats import binom


miles2meters = 1609.34 # TODO: Change back to right number

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(ROOT_DIR, "..", "input-data", "MicrotypeAssignment.csv")
assignment = pd.read_csv(path)

path = os.path.join(ROOT_DIR, "..", "input-data", "Microtypes.csv")
microtypes = pd.read_csv(path)

path = os.path.join(ROOT_DIR, "..", "input-data", "DistanceBins.csv")
distanceBins = pd.read_csv(path)

binWeights = {"short": 0.5, "medium": 0.75, "long": 0.9}
binDistances = {"short": 1, "medium": 5, "long": 15}
collected = dict()
assignmentOut = dict()
for originMID, originMicrotype in enumerate(microtypes.MicrotypeID):
    startingVec = np.zeros((len(microtypes.MicrotypeID), 1))
    startingVec[originMID] = 1.0
    for destinationMID, destinationMicrotype in enumerate(microtypes.MicrotypeID):
        endingVec = np.zeros((len(microtypes.MicrotypeID), 1))
        endingVec[destinationMID] = 1.0
        for distBin in distanceBins.DistanceBinID:
            randMat = np.random.uniform(0,1,(len(microtypes.MicrotypeID),len(microtypes.MicrotypeID)))
            assignmentMatrix = np.zeros_like(randMat)
            offDiagonals = [[1.0] * (len(microtypes.MicrotypeID) - 1), [1.0] * (len(microtypes.MicrotypeID) - 1)]
            randMat += 0.5 * diags(offDiagonals, [-1, 1]).toarray()
            np.fill_diagonal(randMat, 0.0)
            randMat /= randMat.sum(axis=1)[:,None]
            tripDist = binDistances[distBin]
            for i in np.arange(len(microtypes.MicrotypeID)):
                throughDist = microtypes.DiameterInMiles.iloc[i]

                if tripDist > throughDist:
                    probLeaving = 1. - (0.5 * throughDist ** 2.0) / (tripDist * throughDist)
                else:
                    probLeaving = (0.5 * tripDist ** 2.0) / (tripDist * throughDist)
                randMat[i,:] *= probLeaving
            # randMat *= (1 - 1/binDistances[distBin])
            val, vec = eigs(randMat, k=1, which='LM')
            vec = np.real_if_close(vec)
            meanDistancePerStep = np.abs(np.real_if_close(microtypes.DiameterInMiles.values @ vec)[0])

            if tripDist > meanDistancePerStep:
                p = 1. - (0.5 * meanDistancePerStep ** 2.0) / (tripDist * meanDistancePerStep)
            else:
                p = (0.5 * tripDist ** 2.0) / (tripDist * meanDistancePerStep)

            binomial = binom.cdf(0,1, p)
            firstPart = np.sum([binom.cdf(0,i, p) * (np.linalg.matrix_power(randMat, i) @ startingVec) for i in range(10)], axis=0)
            secondPart = endingVec * (meanDistancePerStep/tripDist)
            together = firstPart + secondPart
            averaged = together / np.sum(together)
            for fromIdx, currentMicrotype in enumerate(microtypes.MicrotypeID):
                out = {mID: randMat[fromIdx, toIdx] for toIdx, mID in enumerate(microtypes.MicrotypeID)}
                collected[(originMicrotype, destinationMicrotype, distBin, currentMicrotype)] = out
                assignmentOut[(originMicrotype, destinationMicrotype, distBin, currentMicrotype)] = averaged[fromIdx]
collected = pd.DataFrame(collected)
collected.index.set_names(['To'], inplace=True)
collected.columns.set_names(['OriginMicrotypeID','DestinationMicrotypeID','DistanceBinID','From'], inplace=True)
collected = collected.transpose()

allAssignment = pd.DataFrame(assignmentOut)
allAssignment.columns.set_names(['OriginMicrotypeID','DestinationMicrotypeID','DistanceBinID','ThroughMicrotypeID'])
allAssignment = allAssignment.transpose()
allAssignment.columns.set_names(['Portion'])
path = os.path.join(ROOT_DIR, "..", "input-data", "TransitionMatrices.csv")
collected.to_csv(path)
path = os.path.join(ROOT_DIR, "..", "input-data", "MicrotypeAssignment2.csv")
allAssignment.to_csv(path)
