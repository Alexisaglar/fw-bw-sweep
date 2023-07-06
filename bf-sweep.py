import numpy as np
import csv

Sbase=100; #MVA
Vbase=11; #kV
Zbase=(Vbase**2)/Sbase
phases = 3

with open('data/LineCodes.csv', newline='') as linecodefile, open('data/Lines.csv', newline='') as linesfile, open('data/Loads.csv') as loadsfile, open('data/Load Profiles/Load_Profile_1.csv') as profilefile, open('data/Buscoords.csv') as busfile:
    lines_code = np.array(list(csv.reader(linecodefile)), dtype=object)
    lines_data = np.array(list(csv.reader(linesfile)), dtype=object)
    loads_data = np.array(list(csv.reader(loadsfile)), dtype=object)
    load_profile = np.array(list(csv.reader(profilefile)), dtype=object)
    bus_coords = np.array(list(csv.reader(busfile)), dtype=object)
    
#Keep only useful columns for arrays
lines_code = lines_code[1:,[0,2,3,4,5,6,7]]
load_profile = load_profile[:,1]

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
            bus = int(loads_data[i,2])
            bus_phase = loads_data[i,3]
            if bus_phase == 'A': bus_phase = 0
            if bus_phase == 'B': bus_phase = 1
            if bus_phase == 'C': bus_phase = 2
            Pi[(bus)-1,bus_phase,:] = load_profile[1:]
Qi = np.zeros((len(bus_coords)-1, phases, len(load_profile)-1))

print(Pi)

    

#print(Pi[905,1,0])
#print(Pi)

V_abc = [] 
I_abc = []
Z = []
