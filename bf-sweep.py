import numpy as np
import csv

Sbase=100; #MVA
Vbase=11; #kV
Zbase=(Vbase**2)/Sbase

with open('data/LineCodes.csv', newline='') as linecodefile, open('data/Lines.csv', newline='') as linesfile:
    lines_code = np.array(list(csv.reader(linecodefile)), dtype=object)
    lines_data = np.array(list(csv.reader(linesfile)), dtype=object)

#Keep only useful columns
lines_code = lines_code[1:,[0,2,3,4,5,6,7]]

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

#Append to impedances to data array 
lines_data = np.append(lines_data, impedance.reshape(906,1), axis=1)

V_abc = []
I_abc = []
Z = []
