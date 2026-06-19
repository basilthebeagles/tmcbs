
import numpy as np
import stim
import numpy as np


class CircuitBuilderSurface:

    def __init__(self, CSSObj, d, p, n, numCodeBlocks, ebits=False):

        self.number1qGates = 0
        self.number2qGates = 0
        self.numberMeasurements = 0

        self.n = n
        self.d = d
        self.sqrtn = int(np.sqrt(n))

        self.CSSObj = CSSObj

        self.p_after_clifford_depolarization = p
        self.p_after_reset_flip_probability = p
        self.p_before_measure_flip_probability = p
        self.p_before_round_data_depolarization = p  # 0.001 * p

        self.hz = CSSObj.hz
        self.hx = CSSObj.hx

        self.hz4 = self.hz[0:-(self.sqrtn-1):, :]
        self.hz2 = self.hz[-(self.sqrtn-1):, :]
        self.hx4 = self.hx[0:-(self.sqrtn-1):, :]
        self.hx2 = self.hx[-(self.sqrtn-1):, :]

        self.cbArr = []
        o = 0

        numEachStab = int((n-1)/2)

        for i in range(numCodeBlocks):

            dataRegister = np.arange(o, o + n)
            xRegister = np.arange(o+n, o+n + numEachStab)
            zRegister = np.arange(o+n + numEachStab, o + n + (2*numEachStab))
            self.cbArr.append([dataRegister, xRegister, zRegister])
            o += (n + 2*numEachStab)

        if ebits:
            self.ebitRegister0 = np.arange(o, o + n)  # (6*n,7*n)
            self.ebitRegister1 = np.arange(o+n, o+2*n)  # (7*n,8*n)

            self.measureTracker = [
                np.zeros(o+2*n, dtype=int), np.zeros(o+2*n, dtype=int)]
        else:
            self.measureTracker = [
                np.zeros(o+2*n, dtype=int), np.zeros(o+2*n, dtype=int)]

        self.c = stim.Circuit()

        self.yCoordOffsetMultiple = 0

    def getCirc(self):
        return self.c

    def get1Loc(self, matrix, i):

        return np.nonzero(matrix[i, :])[0][0]

    def measureUpdateTracker(self, qubit):
        self.measureTracker[0] += 1
        self.measureTracker[1] += 1

        self.measureTracker[1][qubit] = self.measureTracker[0][qubit]
        self.measureTracker[0][qubit] = 1

    def initQubits(self, cbArrIndex, errors=True):

        yOffset = self.yCoordOffsetMultiple*((2*self.sqrtn) + 1)

        cb = self.cbArr[cbArrIndex]

        for i in range(len(cb[0])):

            xCord = 1 + (2 * (i % self.sqrtn))
            yCord = 1 + (2*(i // self.sqrtn))

            self.c.append("QUBIT_COORDS", cb[0][i], (xCord, yCord+yOffset))
            self.c.append("RZ", cb[0][i])
            if errors:
                self.c.append("X_ERROR", cb[0][i],
                              self.p_after_reset_flip_probability)

        for i in range(len(cb[1])):

            if i < len(self.hx4):  # in block space

                numPerRow = (self.sqrtn - 1)/2

                yCord = 2 + (2*(i // numPerRow))

                if (i // numPerRow) % 2 == 0:  # even row

                    xCord = 4 + (4 * (i % numPerRow))

                else:
                    xCord = 2 + (4 * (i % numPerRow))

            else:
                if i % 2 == 0:
                    yCord = 0

                else:
                    yCord = 2 * self.sqrtn

                xCord = 2 + (2 * (i % (self.sqrtn - 1)))

            self.c.append("QUBIT_COORDS", cb[1][i], (xCord, yCord + yOffset))

            self.c.append("RX", cb[1][i])
            if errors:
                self.c.append("Z_ERROR", cb[1][i],
                              self.p_after_reset_flip_probability)

        for i in range(len(cb[2])):

            if i < len(self.hx4):  # in block space

                numPerRow = (self.sqrtn - 1)/2

                yCord = 2 + (2*(i // numPerRow))

                if (i // numPerRow) % 2 == 1:  # even row

                    xCord = 4 + (4 * (i % numPerRow))

                else:
                    xCord = 2 + (4 * (i % numPerRow))

            else:
                if i % 2 == 1:
                    xCord = 0

                else:
                    xCord = 2 * self.sqrtn

                yCord = 2 + (2 * (i % (self.sqrtn-1)))

            self.c.append("QUBIT_COORDS", cb[2][i], (xCord, yCord + yOffset))

            self.c.append("RZ", cb[2][i])
            if errors:
                self.c.append("X_ERROR", cb[2][i],
                              self.p_after_reset_flip_probability)

        self.c.append("TICK")
        self.yCoordOffsetMultiple += 1

    def prepareEbits(self, transError=0):
        yOffset = self.yCoordOffsetMultiple*((2*self.sqrtn)+1)

        for i in range(self.n):
            xCord = 1 + (2 * (i % self.sqrtn))
            yCord = 1 + (2*(i // self.sqrtn))
            self.c.append("QUBIT_COORDS",
                          self.ebitRegister0[i], (xCord, yCord + yOffset))
            self.c.append("RZ", self.ebitRegister0[i])
        self.c.append("TICK")

        for i in range(self.n):
            self.c.append("H", self.ebitRegister0[i])

        self.c.append("TICK")

        yOffset = (self.yCoordOffsetMultiple+1)*((2*self.sqrtn) + 1)

        for i in range(self.n):
            xCord = 1 + (2 * (i % self.sqrtn))
            yCord = 1 + (2*(i // self.sqrtn))
            self.c.append("QUBIT_COORDS",
                          self.ebitRegister1[i], (xCord, yCord + yOffset))
            self.c.append("RZ", self.ebitRegister1[i])

        self.c.append("TICK")

        for i in range(self.n):

            self.c.append("CX", (self.ebitRegister0[i], self.ebitRegister1[i]))
            self.c.append(
                "DEPOLARIZE2", (self.ebitRegister0[i], self.ebitRegister1[i]), transError)

        self.yCoordOffsetMultiple += 2

    def extractionRound(self, cbArrIndicies, firstPass=False, errors=True, latticeSurgery=False, latticeSurgeryFirstTime=False, latticeSurgerySSIPObj=None, latticeSurgeryEnd=False, zBasis=True):

        if zBasis == False:
            for cbArrIndex in cbArrIndicies:
                cb = self.cbArr[cbArrIndex]
                for i in range(len(cb[1])):
                    # bit clunky but ensures basis appropriate if suddenly change...
                    self.c.append("RZ", cb[1][i])
                    if errors:
                        self.c.append("Z_ERROR" if zBasis else "X_ERROR",
                                      cb[1][i], self.p_after_reset_flip_probability)
                    self.c.append("RX", cb[2][i])
                    if errors:
                        self.c.append("X_ERROR" if zBasis else "Z_ERROR",
                                      cb[2][i], self.p_after_reset_flip_probability)

        x4order = [3, 2, 1, 0]
        z4order = [3, 1, 2, 0]

        x2order = [1, 0, 1, 0]
        z2order = [1, 0, 1, 0]

        for i in range(4):
            for cbArrIndex in cbArrIndicies:
                cb = self.cbArr[cbArrIndex]

                # print("weight 4 pairs")

                for stab in range(len(self.hx4)):

                    qubitToStabX = np.nonzero(self.hx[stab, :])[0][x4order[i]]
                    self.c.append("CX", (cb[1][stab], cb[0][qubitToStabX]))
                    if errors:
                        self.c.append(
                            "DEPOLARIZE2", (cb[1][stab], cb[0][qubitToStabX]), self.p_after_clifford_depolarization)
                    # print(cb[1][stab], cb[0][qubitToStabX])

                    qubitToStabZ = np.nonzero(self.hz[stab, :])[0][z4order[i]]
                    self.c.append("CX",  (cb[0][qubitToStabZ], cb[2][stab]))
                    if errors:
                        self.c.append(
                            "DEPOLARIZE2", (cb[0][qubitToStabZ], cb[2][stab]), self.p_after_clifford_depolarization)
                    # print( cb[0][qubitToStabZ], cb[2][stab])
                # print("weight 2 pairs")

                    self.number2qGates += 2

                for stab in range(len(self.hx4) if i < 2 else len(self.hx4)+1, len(self.hx), 2):

                    qubitToStabX = np.nonzero(self.hx[stab, :])[0][x2order[i]]
                    self.c.append("CX", (cb[1][stab], cb[0][qubitToStabX]))
                    if errors:
                        self.c.append(
                            "DEPOLARIZE2", (cb[1][stab], cb[0][qubitToStabX]), self.p_after_clifford_depolarization)

                    # print(cb[1][stab], cb[0][qubitToStabX])

                    qubitToStabZ = np.nonzero(
                        self.hz[stab+1 if i < 2 else stab-1, :])[0][z2order[i]]
                    self.c.append(
                        "CX", (cb[0][qubitToStabZ], cb[2][stab+1 if i < 2 else stab-1]))
                    if errors:
                        self.c.append(
                            "DEPOLARIZE2", (cb[0][qubitToStabZ], cb[2][stab+1 if i < 2 else stab-1]), self.p_after_clifford_depolarization)

                    self.number2qGates += 2

            self.c.append("TICK")

        # Z(X) syndrome measurement + detectors

        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]

            for i in range(len(self.hx)):

                if errors:
                    self.c.append("X_ERROR" if zBasis else "Z_ERROR",
                                  cb[2][i], self.p_before_measure_flip_probability)

                self.c.append("MRZ" if zBasis else "MRX", cb[2][i])
                self.numberMeasurements += 1

                self.measureUpdateTracker(cb[2][i])
                if errors:
                    self.c.append("X_ERROR" if zBasis else "Z_ERROR",
                                  cb[2][i], self.p_after_reset_flip_probability)

        if firstPass:
            for i in range(len(cbArrIndicies) * (len(self.hx)), 0, -1):
                self.c.append("DETECTOR", stim.target_rec(-i))
        else:
            for cbArrIndex in cbArrIndicies:
                cb = self.cbArr[cbArrIndex]
                for qubit in cb[2]:
                    self.c.append("DETECTOR", (stim.target_rec(
                        -self.measureTracker[0][qubit]), stim.target_rec(-self.measureTracker[1][qubit])))

        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(len(self.hx)):
                if errors:
                    self.c.append("Z_ERROR" if zBasis else "X_ERROR",
                                  cb[1][i], self.p_before_measure_flip_probability)
                self.c.append("MRX" if zBasis else "MRZ", cb[1][i])
                self.measureUpdateTracker(cb[1][i])
                if errors:
                    self.c.append("Z_ERROR" if zBasis else "X_ERROR",
                                  cb[1][i], self.p_after_reset_flip_probability)  # round 1 init X

                self.numberMeasurements += 1

        self.c.append("TICK")

        if errors:
            for cbArrIndex in cbArrIndicies:
                cb = self.cbArr[cbArrIndex]
                for i in range(len(cb[0])):
                    self.c.append(
                        "DEPOLARIZE1", cb[0][i], self.p_before_round_data_depolarization)

    # originally doThing()
    def endOfCircuitDetectorsForLogicalMeasurementReadout(self, cbIndex, zBasis=True):

        cb = self.cbArr[cbIndex]
        pcm = self.CSSObj.hz if zBasis else self.CSSObj.hx
        logical_pcm = self.CSSObj.lz if zBasis else self.CSSObj.lx
        stab_detector_circuit_str = ""  # stabilizers
        for i, s in enumerate(pcm):
            nnz = np.nonzero(s)[0]
            det_str = "DETECTOR"

            for ind in nnz:
                det_str += f" rec[{-self.measureTracker[0][cb[0][ind]]}]"

            det_str += f" rec[{-self.measureTracker[0][cb[2][i]]}]"
            det_str += "\n"
            stab_detector_circuit_str += det_str
        stab_detector_circuit = stim.Circuit(stab_detector_circuit_str)
        self.c += stab_detector_circuit

    def obsOffset(self, cbIndex, offsetI):

        cb = self.cbArr[cbIndex]

        logical_pcm = self.CSSObj.lz
        log_detector_circuit_str = ""  # logical operators
        for i, l in enumerate(logical_pcm):

            nnz = np.nonzero(l)[0]
            det_str = f"OBSERVABLE_INCLUDE({offsetI+i})"
            for ind in nnz:

                det_str += f" rec[{-self.measureTracker[0][cb[0][ind]]}]"

            det_str += "\n"
            log_detector_circuit_str += det_str
        log_detector_circuit = stim.Circuit(log_detector_circuit_str)
        self.c += log_detector_circuit

    def measureDataQubits(self, cbArrIndicies, zBasis=True, noise = -1):

        if noise == -1:
            noise = self.p_before_measure_flip_probability


        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(len(cb[0])):
                self.c.append("X_ERROR" if zBasis else "Z_ERROR",
                              cb[0][i], noise)  # new!
                self.c.append("MRZ" if zBasis else "MRX", cb[0][i])
                self.measureUpdateTracker(cb[0][i])

            self.c.append("TICK")

    def logicalOp(self, opType, opIndices, cbArrIndex, errors=True):

        cb = self.cbArr[cbArrIndex]

        for i in range(len(opIndices)):

            if opIndices[i] != 1:
                continue

            self.c.append(opType, cb[0][i])
            if errors:
                self.c.append(
                    "DEPOLARIZE1", cb[0][i], self.p_after_clifford_depolarization)

        self.c.append("TICK")

    def transversalOp(self, op, cbArrIndicies,  typeArr=["BB", "BB"], customNoise=-1):

        n = self.n
        if len(cbArrIndicies) == 1 and typeArr[0] == "BB":

            cb = self.cbArr[cbArrIndicies[0]]
            for i in range(len(cb[0])):
                self.c.append(op, cb[0][i])
                self.c.append(
                    "DEPOLARIZE1", cb[0][i], self.p_after_clifford_depolarization)

                self.number1qGates += 1

        elif len(cbArrIndicies) == 1 and typeArr[0] == "e":
            cb = []
            if cbArrIndicies[0] == 0:
                cb = self.ebitRegister0
            else:
                cb = self.ebitRegister1

            for i in range(n):
                self.c.append(op, cb[i])
                self.c.append(
                    "DEPOLARIZE1", cb[i], self.p_after_clifford_depolarization)
                self.number1qGates += 1

        elif len(cbArrIndicies) == 2:

            cbA = self.cbArr[cbArrIndicies[0]]
            cbB = self.cbArr[cbArrIndicies[1]]

            if typeArr[0] == "BB" and typeArr[1] == "BB":

                for i in range(len(cbA[0])):
                    self.c.append(op, (cbA[0][i], cbB[0][i]))
                    if customNoise == -1:
                        self.c.append(
                            "DEPOLARIZE2", (cbA[0][i], cbB[0][i]), self.p_after_clifford_depolarization)
                    else:
                        self.c.append(
                            "DEPOLARIZE2", (cbA[0][i], cbB[0][i]), customNoise)

                    self.number2qGates += 1

            elif typeArr[0] == "BB" and typeArr[1] == "e":
                cb = self.cbArr[cbArrIndicies[0]]
                ebitArray = []
                if cbArrIndicies[1] == 0:
                    ebitArray = self.ebitRegister0
                else:
                    ebitArray = self.ebitRegister1

                for i in range(len(cb[0])):
                    self.c.append(op, (cb[0][i], ebitArray[i]))
                    self.c.append(
                        "DEPOLARIZE2", (cb[0][i], ebitArray[i]), self.p_after_clifford_depolarization)
                    self.number2qGates += 1

            elif typeArr[0] == "e" and typeArr[1] == "BB":
                cb = self.cbArr[cbArrIndicies[1]]
                ebitArray = []
                if cbArrIndicies[0] == 0:
                    ebitArray = self.ebitRegister0
                else:
                    ebitArray = self.ebitRegister1

                for i in range(len(cb[0])):
                    self.c.append(op, (ebitArray[i], cb[0][i]))
                    self.c.append(
                        "DEPOLARIZE2", (ebitArray[i], cb[0][i]), self.p_after_clifford_depolarization)
                    self.number2qGates += 1

        self.c.append("TICK")

    def measureEbit0ThenCorrectEbit1(self):

        ebitRegister0 = self.ebitRegister0
        ebitRegister1 = self.ebitRegister1
        for i in range(self.n):
            self.c.append(
                "Z_ERROR", ebitRegister0[i], self.p_before_measure_flip_probability)
            self.c.append("M", ebitRegister0[i])
            self.measureUpdateTracker(ebitRegister0[i])
            self.c.append("CX", [stim.target_rec(-1), ebitRegister1[i]])
            self.c.append(
                "DEPOLARIZE1", ebitRegister1[i], self.p_after_clifford_depolarization)

            self.number1qGates += 1
            self.numberMeasurements += 1

        self.c.append("TICK")

    def measureEbit1ThenCorrectCB(self, cbIndex):

        cb = self.cbArr[cbIndex]
        ebitRegister1 = self.ebitRegister1

        for i in range(len(cb[0])):
            self.c.append(
                "Z_ERROR", ebitRegister1[i], self.p_before_measure_flip_probability)

            self.c.append("M", ebitRegister1[i])
            self.measureUpdateTracker(ebitRegister1[i])
            self.c.append("CZ", [stim.target_rec(-1), cb[0][i]])

            self.number1qGates += 1
            self.numberMeasurements += 1

        self.c.append("TICK")

    def measureCBThenCorrectCB(self, op, cbIndex, zbasis=True):

        cb0 = self.cbArr[cbIndex[0]]
        cb1 = self.cbArr[cbIndex[1]]

        for i in range(len(cb0[0])):
            self.c.append("Z_ERROR", cb0[0][i],
                          self.p_before_measure_flip_probability)

            self.c.append("MRZ" if zbasis else "MX", cb0[0][i])
            self.measureUpdateTracker(cb0[0][i])
            self.c.append(op, [stim.target_rec(-1), cb1[0][i]])
            self.c.append("DEPOLARIZE1", cb1[0][i],
                          self.p_after_clifford_depolarization)

            self.number1qGates += 1
            self.numberMeasurements += 1

        self.c.append("TICK")


    """
    def includeDelayDueToEbitGenTime(self, T1, T2, singleEitGenTime, cbArrIndicies ):
        print("etc")

        ebitArray0 = self.ebitRegister0
        ebitArray1 = self.ebitRegister1
        for i in range(self.n):
            errTuple = self.getDelayError(T1, T2, singleEitGenTime* (self.n - (i+1)))
            self.c.append("PAULI_CHANNEL_1", (ebitArray0[i],), errTuple)
            self.c.append("PAULI_CHANNEL_1", (ebitArray1[i],), errTuple)

        errTuple = self.getDelayError(T1, T2, singleEitGenTime* self.n)
"""

    def getDelayError(self, ratio):

        #fix this
        t2Ratio = ratio 

        p_x = 0.25 * (1- np.exp(-ratio))
        p_y = p_x
        p_z = 0.5 * (1- np.exp(-t2Ratio)) - p_x

        return (p_x, p_y, p_z)


    #def includeDelayDueToEbitGenTime(self, T1, T2, singleEitGenTime, cbArrIndicies, typeArr ):

    def includeDelayDueToEbitGenTime(self, ratio, cbArrIndicies, typeArr, numPerCycle=-1 ):


        for k in range(len(cbArrIndicies)):
            if typeArr[k] == "e":
                if cbArrIndicies[k] == 0:
                    ebitArr = self.ebitRegister0
                elif cbArrIndicies[k] == 1:
                    ebitArr = self.ebitRegister1
                
                if numPerCycle == -1:
                    for i in range(self.n):
                        errTuple = self.getDelayError(ratio* (self.n - (i+1)))
                        self.c.append("PAULI_CHANNEL_1", (ebitArr[i],), errTuple)
                else:
                 
                    if self.n % numPerCycle !=0:
                        raise ValueError

                    numCyclesNeeded = int(self.n/numPerCycle)
                    for i in range(numCyclesNeeded):
                        errTuple = self.getDelayError(ratio* (numCyclesNeeded - (i+1)))

                        for j in range(numPerCycle):
                            self.c.append("PAULI_CHANNEL_1", (ebitArr[(i * numPerCycle) + j ],), errTuple)
