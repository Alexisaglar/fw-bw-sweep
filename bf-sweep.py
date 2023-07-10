import numpy as np
import csv

Sbase=100; #MVA
Vbase=240.177; #kV
Zbase=(Vbase**2)/Sbase
phases = 3
PF = 0.95


with open('data/LineCodes.csv', newline='') as linecodefile, open('data/Lines.csv', newline='') as linesfile, open('data/Loads.csv') as loadsfile, open('data/Load Profiles/Load_Profile_1.csv') as profilefile, open('data/Buscoords.csv') as busfile:
    lines_code = np.array(list(csv.reader(linecodefile)), dtype=object)
    lines_data = np.array(list(csv.reader(linesfile)), dtype=object)
    loads_data = np.array(list(csv.reader(loadsfile)), dtype=object)
    load_profile = np.array(list(csv.reader(profilefile)), dtype=object)
    bus_coords = np.array(list(csv.reader(busfile)), dtype=object)
    
#Keep only useful columns for arrays
lines_code = lines_code[1:,[0,2,3,4,5,6,7]]
load_profile = load_profile[:,1]


#--------- Backward ordered nodes and lines -----#
#--------- First determine ending nodes
endNodes = []
backOrderedNodes = []
backOrderedLine = []
nodeA = lines_data[1:,1].astype(int)
nodeB = lines_data[1:,2].astype(int)
while len(backOrderedNodes) < len(lines_data)-1:
    endNodes = np.setdiff1d(nodeB, nodeA)
    backOrderedNodes = np.append(backOrderedNodes, endNodes)
    idx = np.intersect1d(nodeB, endNodes, return_indices=True)[1]
    for i in range(len(idx)):
        backOrderedLine = np.append(backOrderedLine, nodeA[idx[i]])
        nodeA[idx[i]]= 0
        nodeB[idx[i]]= 0

#--------- Forward ordered lines -----#
forwardOrderedLine = np.flip(backOrderedLine)

#Initiate arrays
resistance = ['ResistanceR1']
reactance = ['ReactanceX1']
impedance = ['ImpedanceZ']

#Find in line code data the specific resitance and reactance for each line
for i in range(len(lines_data)):
    if i != 0:
        resistance = np.append(resistance, lines_code[np.where(lines_data[i,6] == lines_code[:,0])[0][0], 2])
        reactance = np.append(reactance, lines_code[np.where(lines_data[i,6] == lines_code[:,0])[0][0], 3])

#Append the resistance and reactance to the array
lines_data = np.append(lines_data, resistance.reshape(906,1), axis=1)
lines_data = np.append(lines_data, reactance.reshape(906,1), axis=1)

#Calculate impedances of the lines
for i in range(len(lines_data)):
    if i !=0:
        impedance = np.append(impedance, float(lines_data[i,7])+1j*float(lines_data[i,8]))

#Append impedances to data array 
lines_data = np.append(lines_data, impedance.reshape(906,1), axis=1)


#Create array with real and reactive power for each bus with 3 phases
Pi = np.zeros((len(bus_coords)-1, phases, len(load_profile)-1))
for i in range(len(loads_data)):
        if i != 0:
            #bus = int(loads_data[i,2])
            bus_phase = loads_data[i,3]
            if bus_phase == 'A': bus_phase = 0
            if bus_phase == 'B': bus_phase = 1
            if bus_phase == 'C': bus_phase = 2
            Pi[i-1,bus_phase,:] = load_profile[1:]
#Convert power to Watts instead of kW >
Pi = Pi*1000

#Create an same size array as apparent power for reactive power 
Qi = np.zeros((len(bus_coords)-1, phases, len(load_profile)-1))
#Apply power factor to determine reactive power
Qi = (Pi/0.95) * np.sin(np.arccos(PF))


calcNodeVoltages = np.zeros((len(loads_data)-1, phases, 50))
calcLoadCurrents = np.zeros((len(loads_data)-1, phases, 50))
calcLineCurrents = np.zeros((len(lines_data)-1, phases, 50))
#----- Backward-forward sweep -----#
for i in range(len(load_profile)-1):
    #Creating starting values for algorithm
    error = 1
    epsilon = 0.000001
    iter = 1 
    nodeVoltages = np.zeros((len(bus_coords)-1, phases, 900))
    loadCurrents = np.zeros((len(bus_coords)-1, phases, 900))
    lineCurrents = np.zeros((len(lines_data)-1, phases, 900))
    
    if i == 0:
        nodeVoltages[:,:,iter-1] = np.ones((len(bus_coords)-1,3))*((1*np.exp(1j*(-2*np.pi/3))*np.exp(1j*(2*np.pi/3)))*Vbase) 
    else:
        nodeVoltages[:,:,iter] = calcNodeVoltages[:,:,i]
    
    while error > epsilon:
        iter = iter+1
        if iter > 100: 
            print('Error') 
            break
        loadCurrents[:,:,iter] = np.conj((Pi[:,:,i]+1j*Qi[:,:,i])/nodeVoltages[:,:,i]) #Calculate load current from S/V

        #----- start backward sweep -----#
        for bsweep in range(len(backOrderedNodes)-1):
            node = int(backOrderedNodes[bsweep])
            nA_n = np.where(lines_data[1:,1].astype(int) == node)[0]
            #print(nA_n)
            if np.size(nA_n) != 0:
                 outCurrents = lineCurrents[nA_n, :, iter]
            else:
                outCurrents = np.zeros(3)
            outCurrentSum = np.sum(outCurrents)
            nB_n = np.where(lines_data[1:,2].astype(int) == node)[0]
            loadCurs = loadCurrents[node-1,:,iter]
            sumCurs = loadCurs + outCurrentSum
            lineCurrents[nB_n,:,iter] = sumCurs
            print(loadCurs)