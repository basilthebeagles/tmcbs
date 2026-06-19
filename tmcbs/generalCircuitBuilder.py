
import numpy as np
import stim
import numpy as np

class CircuitBuilder:


    def __init__(self, BBObject, d, p, n, numCodeBlocks, ebits=False):

        self.number1qGates = 0
        self.number2qGates = 0
        self.numberMeasurements = 0
        self.d=d
        
        self.n = n

        self.p_after_clifford_depolarization = p
        self.p_after_reset_flip_probability = p
        self.p_before_measure_flip_probability = p
        self.p_before_round_data_depolarization = p  

        code, A_list, B_list = BBObject
        self.code = code

        a1, a2, a3 = A_list
        b1, b2, b3 = B_list

        self.A1 = a1.toarray()
        self.A2 = a2.toarray()
        self.A3 = a3.toarray()

        self.B1 = b1.toarray()
        self.B2 = b2.toarray()
        self.B3 = b3.toarray()

        self.cbArr = []
        o = 0
        for i in range(numCodeBlocks):

            xRegister = np.arange(o,o+ n//2)
            LRegister = np.arange(o+ n//2, o+ n)
            RRegister = np.arange(o+n,o+ n + n//2)
            zRegister = np.arange(o+n + n//2,o+ 2*n)

            self.cbArr.append([xRegister, LRegister, RRegister, zRegister])
            o +=2*n


        if ebits:
            self.ebitRegister0 = np.arange(o,o + n)#(6*n,7*n)
            self.ebitRegister1 = np.arange(o+n,o+2*n)#(7*n,8*n)  

            self.measureTracker = [np.zeros(o+2*n, dtype=int), np.zeros(o+2*n, dtype=int)] #add ebits later
        else:
            self.measureTracker = [np.zeros(o+2*n, dtype=int), np.zeros(o+2*n, dtype=int)] #add ebits later

        self.c = stim.Circuit()



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
        for i in range(self.n//2):
            self.c.append("RX", self.cbArr[cbArrIndex][0][i])
            if errors: self.c.append("Z_ERROR", self.cbArr[cbArrIndex][0][i], self.p_after_reset_flip_probability)

        for i in range(self.n//2):
            self.c.append("RZ", self.cbArr[cbArrIndex][1][i])
            if errors: self.c.append("X_ERROR", self.cbArr[cbArrIndex][1][i], self.p_after_reset_flip_probability)

        for i in range(self.n//2):
            self.c.append("RZ", self.cbArr[cbArrIndex][2][i])
            if errors: self.c.append("X_ERROR", self.cbArr[cbArrIndex][2][i], self.p_after_reset_flip_probability)
        
        for i in range(self.n//2):
            self.c.append("RZ", self.cbArr[cbArrIndex][3][i])
            if errors: self.c.append("X_ERROR", self.cbArr[cbArrIndex][3][i], self.p_after_reset_flip_probability)

        self.c.append("TICK")


    
    def prepareEbits(self, transError=0):

        for i in range(self.n):
            self.c.append("RZ", self.ebitRegister0[i])
            self.c.append("H", self.ebitRegister0[i])
        
        for i in range(self.n):
            self.c.append("RZ", self.ebitRegister1[i])
            self.c.append("CX", (self.ebitRegister0[i], self.ebitRegister1[i]))

            self.c.append("DEPOLARIZE2",(self.ebitRegister0[i], self.ebitRegister1[i]), transError )
        
        self.c.append("TICK")

            

    def extractionRound(self, cbArrIndicies, firstPass=False, errors=True, latticeSurgery=False, latticeSurgeryFirstTime=False, latticeSurgerySSIPObj=None, latticeSurgeryEnd=False, zBasis=True):

        if zBasis == False:
            for cbArrIndex in cbArrIndicies:
                cb = self.cbArr[cbArrIndex]
                for i in range(self.n // 2):
                    #bit clunky but ensures basis appropriate if suddenly change...
                    self.c.append("RZ", cb[0][i])
                    if errors: self.c.append("Z_ERROR" if zBasis else "X_ERROR", cb[0][i], self.p_after_reset_flip_probability)
                    self.c.append("RX", cb[3][i])
                    if errors: self.c.append("X_ERROR" if zBasis else "Z_ERROR", cb[3][i], self.p_after_reset_flip_probability)



        # round 1

        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n // 2):
                self.c.append("CX", (cb[2][self.get1Loc(self.A1.T, i)], cb[3][i]))
                if errors: self.c.append("DEPOLARIZE2", (cb[2][self.get1Loc(self.A1.T, i)], cb[3][i]), self.p_after_clifford_depolarization)
                
                if errors: self.c.append("DEPOLARIZE1", cb[1][i], self.p_before_round_data_depolarization) #round 1 L idle
            self.number2qGates += self.n//2
        self.c.append("TICK")

        # round 2
        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]

            for i in range(self.n // 2):
                self.c.append("CX", (cb[0][i], cb[1][self.get1Loc(self.A2, i)]))
                if errors: self.c.append("DEPOLARIZE2", (cb[0][i], cb[1][self.get1Loc(self.A2, i)]), self.p_after_clifford_depolarization)
                self.c.append("CX", (cb[2][self.get1Loc(self.A3.T, i)], cb[3][i]))
                if errors: self.c.append("DEPOLARIZE2", (cb[2][self.get1Loc(self.A3.T, i)], cb[3][i]), self.p_after_clifford_depolarization)

            self.number2qGates += self.n

        self.c.append("TICK")

        # round 3
        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n // 2):
                self.c.append("CX", (cb[0][i], cb[2][self.get1Loc(self.B2, i)]))
                if errors: self.c.append("DEPOLARIZE2", (cb[0][i], cb[2][self.get1Loc(self.B2, i)]), self.p_after_clifford_depolarization)
                self.c.append("CX", (cb[1][self.get1Loc(self.B1.T, i)], cb[3][i]))
                if errors: self.c.append("DEPOLARIZE2", (cb[1][self.get1Loc(self.B1.T, i)], cb[3][i]), self.p_after_clifford_depolarization)

            self.number2qGates += self.n

        self.c.append("TICK")

        # round 4
        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n // 2):
                self.c.append("CX", (cb[0][i], cb[2][self.get1Loc(self.B1, i)]))
                if errors: self.c.append("DEPOLARIZE2", (cb[0][i], cb[2][self.get1Loc(self.B1, i)]), self.p_after_clifford_depolarization)
                self.c.append("CX", (cb[1][self.get1Loc(self.B2.T, i)], cb[3][i]))
                if errors: self.c.append("DEPOLARIZE2", (cb[1][self.get1Loc(self.B2.T, i)], cb[3][i]), self.p_after_clifford_depolarization)
            
            self.number2qGates += self.n

                
        self.c.append("TICK")

        # round 5
        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n // 2):
                self.c.append("CX", (cb[0][i], cb[2][self.get1Loc(self.B3, i)]))
                if errors: self.c.append("DEPOLARIZE2", (cb[0][i], cb[2][self.get1Loc(self.B3, i)]), self.p_after_clifford_depolarization)
                self.c.append("CX", (cb[1][self.get1Loc(self.B3.T, i)], cb[3][i]))
                if errors: self.c.append("DEPOLARIZE2", (cb[1][self.get1Loc(self.B3.T, i)], cb[3][i]), self.p_after_clifford_depolarization)

            self.number2qGates += self.n

            
        self.c.append("TICK")

        # round 6
        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n // 2):
                self.c.append("CX", (cb[0][i], cb[1][self.get1Loc(self.A1, i)]))
                if errors: self.c.append("DEPOLARIZE2", (cb[0][i], cb[1][self.get1Loc(self.A1, i)]), self.p_after_clifford_depolarization)
                self.c.append("CX", (cb[2][self.get1Loc(self.A2.T, i)], cb[3][i]))
                if errors: self.c.append("DEPOLARIZE2", (cb[2][self.get1Loc(self.A2.T, i)], cb[3][i]), self.p_after_clifford_depolarization)
            
            self.number2qGates += self.n

        self.c.append("TICK")

    # round 7
        
        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n // 2):
                self.c.append("CX", (cb[0][i], cb[1][self.get1Loc(self.A3, i)]))
                if errors: self.c.append("DEPOLARIZE2", (cb[0][i], cb[1][self.get1Loc(self.A3, i)]), self.p_after_clifford_depolarization)
                if errors: self.c.append("X_ERROR" if zBasis else "Z_ERROR", cb[3][i], self.p_before_measure_flip_probability)
                
                self.c.append("MRZ" if zBasis  else "MRX", cb[3][i])
                
                self.measureUpdateTracker(cb[3][i])
                if errors: self.c.append("X_ERROR" if zBasis else "Z_ERROR", cb[3][i], self.p_after_reset_flip_probability) #round 8 initZ
            
            self.number2qGates += self.n//2
            self.numberMeasurements += self.n//2


        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n // 2):
                if errors: self.c.append("DEPOLARIZE1", cb[2][i], self.p_before_round_data_depolarization) #idle r round 7

        if firstPass:
            for i in range(len(cbArrIndicies) * (self.n // 2), 0, -1):
                self.c.append("DETECTOR", stim.target_rec(-i))
        else:
            for cbArrIndex in cbArrIndicies:
                cb = self.cbArr[cbArrIndex]
                for qubit in cb[3]:
                    self.c.append("DETECTOR", (stim.target_rec(-self.measureTracker[0][qubit]), stim.target_rec(-self.measureTracker[1][qubit])))

        self.c.append("TICK")

        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n // 2):
                if errors: self.c.append("Z_ERROR" if zBasis else "X_ERROR", cb[0][i], self.p_before_measure_flip_probability)
                self.c.append("MRX" if zBasis else "MRZ", cb[0][i])
                self.measureUpdateTracker(cb[0][i])
                if errors: self.c.append("Z_ERROR" if zBasis else "X_ERROR", cb[0][i], self.p_after_reset_flip_probability)#round 1 init X

            self.numberMeasurements += self.n//2

        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n // 2):
                if errors: self.c.append("DEPOLARIZE1", cb[1][i], self.p_before_round_data_depolarization) #round 8 idle l
                if errors: self.c.append("DEPOLARIZE1", cb[2][i], self.p_before_round_data_depolarization) # round 8 idle r

        self.c.append("TICK")



    def endOfCircuitDetectorsForLogicalMeasurementReadout(self, cbIndex, zBasis = True): #originally doThing()

            cb = self.cbArr[cbIndex]
            pcm = self.code.hz if zBasis else self.code.hx
            logical_pcm = self.code.lz  if zBasis else self.code.lx
            stab_detector_circuit_str = ""  # stabilizers
            for i, s in enumerate(pcm):
                nnz = np.nonzero(s)[0]
                det_str = "DETECTOR"

                for ind in nnz:

                    if ind < self.n//2:
                        det_str += f" rec[{-self.measureTracker[0][cb[1][ind]]}]" #left register
                    else:
                        det_str += f" rec[{-self.measureTracker[0][cb[2][ind-(self.n//2)]]}]" #right register
                # if z_basis else f" rec[{-n-n//2+i}]"
                det_str += f" rec[{-self.measureTracker[0][cb[3][i]]}]"
                det_str += "\n"
                stab_detector_circuit_str += det_str
            stab_detector_circuit = stim.Circuit(stab_detector_circuit_str)
            self.c += stab_detector_circuit


    def obsOffset(self, cbIndex, offsetI):

        cb = self.cbArr[cbIndex]

        logical_pcm = self.code.lz
        log_detector_circuit_str = ""  # logical operators
        for i, l in enumerate(logical_pcm):
            
            nnz = np.nonzero(l)[0]
            det_str = f"OBSERVABLE_INCLUDE({offsetI+i})"
            for ind in nnz:
                if ind < self.n//2:
                    det_str += f" rec[{-self.measureTracker[0][cb[1][ind]]}]"
                else:
                    det_str += f" rec[{-self.measureTracker[0][cb[2][ind-(self.n//2)]]}]"
            det_str += "\n"
            log_detector_circuit_str += det_str
        log_detector_circuit = stim.Circuit(log_detector_circuit_str)
        self.c += log_detector_circuit
    


    def measureDataQubits(self, cbArrIndicies, zBasis = True, noise=-1):


        if noise == -1:
            noise = self.p_before_measure_flip_probability
        
        for cbArrIndex in cbArrIndicies:
            cb = self.cbArr[cbArrIndex]
            for i in range(self.n//2):
                self.c.append("X_ERROR" if zBasis else "Z_ERROR", cb[1][i], noise) #new!
                self.c.append("MRZ" if zBasis else "MRX", cb[1][i])
                self.measureUpdateTracker(cb[1][i])

            for i in range(self.n//2):
                self.c.append("X_ERROR" if zBasis else "Z_ERROR", cb[2][i], noise)
                self.c.append("MRZ" if zBasis else "MRX", cb[2][i])
                self.measureUpdateTracker(cb[2][i])
            
            self.c.append("TICK")

            
    def logicalOp(self,opType,opIndices ,cbArrIndex, errors=True):


        cb = self.cbArr[cbArrIndex]

        for i in range(len(opIndices)):
            
            if opIndices[i] != 1:
                continue
            
            if i <self.n//2:

                self.c.append(opType, cb[1][i])
                if errors: self.c.append("DEPOLARIZE1", cb[1][i], self.p_after_clifford_depolarization)
            elif i<self.n:
                
                self.c.append(opType, cb[2][i-self.n//2])
                if errors: self.c.append("DEPOLARIZE1", cb[2][i-self.n//2], self.p_after_clifford_depolarization)

        self.c.append("TICK")


    def transversalOp(self, op, cbArrIndicies,  typeArr=["BB", "BB"], customNoise=-1):
        
        n = self.n
        if len(cbArrIndicies) == 1 and typeArr[0]=="BB":

            cb = self.cbArr[cbArrIndicies[0  ]]
            for i in range(n//2):
                self.c.append(op, cb[1][i])
                self.c.append("DEPOLARIZE1", cb[1][i], self.p_after_clifford_depolarization)

            for i in range(n//2):
                self.c.append(op, cb[2][i])
                self.c.append("DEPOLARIZE1", cb[2][i], self.p_after_clifford_depolarization)

            self.number1qGates += self.n

        elif len(cbArrIndicies) == 1 and typeArr[0]=="e":
            cb = []
            if cbArrIndicies[0] == 0:
                cb = self.ebitRegister0
            else:
                cb = self.ebitRegister1

            for i in range(n):
                self.c.append(op, cb[i])
                self.c.append("DEPOLARIZE1", cb[i], self.p_after_clifford_depolarization)
            
            self.number1qGates += self.n


        elif len(cbArrIndicies) == 2:

            cbA = self.cbArr[cbArrIndicies[0]]
            cbB     = self.cbArr[cbArrIndicies[1]]

            if typeArr[0]=="BB" and typeArr[1]=="BB":

                for i in range(n//2):
                    self.c.append(op, (cbA[1][i],cbB[1][i] ))
                    if customNoise == -1: 
                        self.c.append("DEPOLARIZE2", (cbA[1][i],cbB[1][i] ), self.p_after_clifford_depolarization)
                    else:
                        self.c.append("DEPOLARIZE2", (cbA[1][i],cbB[1][i] ), customNoise)

                for i in range(n//2):
                    self.c.append(op, (cbA[2][i],cbB[2][i] ))

                    if customNoise == -1:
                        self.c.append("DEPOLARIZE2", (cbA[2][i],cbB[2][i] ), self.p_after_clifford_depolarization)
                    else:
                        self.c.append("DEPOLARIZE2", (cbA[2][i],cbB[2][i] ), customNoise)

                self.number2qGates += self.n

            
            elif typeArr[0]=="BB" and typeArr[1]=="e":
                cb = self.cbArr[cbArrIndicies[0]]
                ebitArray = []
                if cbArrIndicies[1] == 0:
                    ebitArray = self.ebitRegister0
                else:
                    ebitArray = self.ebitRegister1

                trueN = 0
                for i in range(n//2):
                    self.c.append(op, (cb[1][i],ebitArray[trueN] ))
                    self.c.append("DEPOLARIZE2", (cb[1][i],ebitArray[trueN] ), self.p_after_clifford_depolarization)

                    trueN += 1
                for i in range(n//2):
                    self.c.append(op, (cb[2][i],ebitArray[trueN]))
                    self.c.append("DEPOLARIZE2", (cb[2][i],ebitArray[trueN]), self.p_after_clifford_depolarization)

                    trueN += 1

                self.number2qGates += self.n


            elif typeArr[0]=="e" and typeArr[1]=="BB":
                cb = self.cbArr[cbArrIndicies[1]]
                ebitArray = []
                if cbArrIndicies[0] == 0:
                    ebitArray = self.ebitRegister0
                else:
                    ebitArray = self.ebitRegister1                
                
                trueN = 0
                for i in range(n//2):
                    self.c.append(op, (ebitArray[trueN], cb[1][i]))
                    self.c.append("DEPOLARIZE2", (ebitArray[trueN], cb[1][i]), self.p_after_clifford_depolarization)

                    trueN += 1
                for i in range(n//2):
                    self.c.append(op, (ebitArray[trueN], cb[2][i]))
                    self.c.append("DEPOLARIZE2", (ebitArray[trueN], cb[2][i]), self.p_after_clifford_depolarization)

                    trueN += 1
                self.number2qGates += self.n

        self.c.append("TICK")



    def measureEbit0ThenCorrectEbit1(self):
        
        ebitRegister0 = self.ebitRegister0
        ebitRegister1 = self.ebitRegister1
        for i in range(self.n):
            self.c.append("Z_ERROR", ebitRegister0[i], self.p_before_measure_flip_probability)
            self.c.append("M", ebitRegister0[i])
            self.measureUpdateTracker(ebitRegister0[i])
            self.c.append("CX", [stim.target_rec(-1),ebitRegister1[i]])
            self.c.append("DEPOLARIZE1", ebitRegister1[i], self.p_after_clifford_depolarization)

        self.c.append("TICK")

        self.number1qGates += self.n
        self.numberMeasurements += self.n



        
    def measureEbit1ThenCorrectCB(self, cbIndex):

        cb = self.cbArr[cbIndex]
        ebitRegister1 = self.ebitRegister1

        trueN = 0
        for i in range(self.n//2):
            self.c.append("Z_ERROR", ebitRegister1[trueN], self.p_before_measure_flip_probability)

            self.c.append("M", ebitRegister1[trueN])
            self.measureUpdateTracker(ebitRegister1[trueN])        
            self.c.append("CZ", [stim.target_rec(-1),cb[1][i]])
            trueN += 1
        for i in range(self.n//2):
            self.c.append("Z_ERROR", ebitRegister1[trueN], self.p_before_measure_flip_probability)

            self.c.append("M", ebitRegister1[trueN])
            self.measureUpdateTracker(ebitRegister1[trueN])   
            self.c.append("CZ", [stim.target_rec(-1),cb[2][i]])
            self.c.append("DEPOLARIZE1", cb[2][i], self.p_after_clifford_depolarization)

            trueN += 1
        self.number1qGates += self.n
        self.numberMeasurements += self.n
        self.c.append("TICK")


    def measureCBThenCorrectCB(self, op, cbIndex):

        cb0 = self.cbArr[cbIndex[0]]
        cb1 = self.cbArr[cbIndex[1]]

        trueN = 0
        for i in range(self.n//2):
            self.c.append("Z_ERROR", cb0[1][i], self.p_before_measure_flip_probability)

            self.c.append("M", cb0[1][i])
            self.measureUpdateTracker(cb0[1][i])        
            self.c.append(op, [stim.target_rec(-1),cb1[1][i]])
            self.c.append("DEPOLARIZE1", cb1[1][i], self.p_after_clifford_depolarization)

            trueN += 1
        for i in range(self.n//2):
            self.c.append("Z_ERROR", cb0[2][i], self.p_before_measure_flip_probability)
            self.c.append("M", cb0[2][i])
            self.measureUpdateTracker(cb0[2][i])        
            self.c.append(op, [stim.target_rec(-1),cb1[2][i]])
            self.c.append("DEPOLARIZE1", cb1[2][i], self.p_after_clifford_depolarization)

            trueN += 1
        self.c.append("TICK")

        self.number1qGates += self.n
        self.numberMeasurements += self.n

    def getDelayError(self, ratio):

        #fix this
        t2Ratio = ratio 

        p_x = 0.25 * (1- np.exp(-ratio))
        p_y = p_x
        p_z = 0.5 * (1- np.exp(-t2Ratio)) - p_x

        return (p_x, p_y, p_z)

    def includeDelayDueToEbitGenTime(self, ratio, cbArrIndicies, typeArr, numPerCycle=-1  ):

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
