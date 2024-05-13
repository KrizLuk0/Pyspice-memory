import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
import time
import matplotlib.pyplot as plt
from memory_profiler import profile
import gc

ngspice = NgSpiceShared.new_instance()
# Function to simulate a chain of inverters and measure rise/fall times
@profile
def SimulationChainOfInverters(ChainNetlist, LowThreshold, HighThreshold, SimParams):
    circuit = Circuit('Chain of Inverter')  # Create a new circuit
    circuit.include(ChainNetlist)  # Include the netlist file for the inverter chain

    # Setup and run a transient simulation
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(**SimParams)

    # Extract output voltage and time from the simulation
    OutVoltage = analysis['Vout']
    Time = np.array(analysis.time)
    Voltages = np.array(OutVoltage)

    # Measure rise and fall times using edges detection
    RiseTimeChain, FallTimeChain = MeasEdges(Time, Voltages, LowThreshold, HighThreshold)

    # Clear memory
    simulator.ngspice.reset()
    del circuit, simulator, OutVoltage, Time, Voltages, analysis
    gc.collect()

    return RiseTimeChain, FallTimeChain

# Function to detect rising and falling edges based on voltage thresholds
@profile
def MeasEdges(Time, Voltages, LowThreshold, HighThreshold):
    RiseEdge = FallEdge = None

    # Loop through voltage data to find where it crosses the low threshold upwards
    for i in range(1, len(Voltages)):
        if Voltages[i - 1] < LowThreshold <= Voltages[i]:
            StartRise = Time[i]
            for j in range(i, len(Voltages)):
                if Voltages[j - 1] < HighThreshold <= Voltages[j]:
                    RiseEdge = Time[j] - StartRise
                    break
            break

    # Loop through voltage data to find where it crosses the high threshold downwards
    for i in range(1, len(Voltages)):
        if Voltages[i - 1] > HighThreshold >= Voltages[i]:
            StartFall = Time[i]
            for j in range(i, len(Voltages)):
                if Voltages[j - 1] > LowThreshold >= Voltages[j]:
                    FallEdge = Time[j] - StartFall
                    break
            break

    return RiseEdge, FallEdge

# Function to adjust parameters of a generator based on measured rise and fall times
@profile
def ChangeParamsOfGen(RiseChain, FallChain, SimulateNetlist):
    RiseTime_ps = int(RiseChain*1e12)
    FallTime_ps = int(FallChain*1e12)

    # Calculate average parameter value for generator
    ParamOfGen = (RiseTime_ps + FallTime_ps) // 2
    UpdatedContent = []
    try:
        with open(SimulateNetlist, 'r') as file:
            lines = file.readlines()

        ParamOfGenString = f"{ParamOfGen}p"

        # Update the pulse parameters in the netlist file
        for line in lines:
            if 'pulse(' in line:
                parts = line.split('pulse(')
                PulseParams = parts[1].strip(')\n').split(' ')
                PulseParams[3] = ParamOfGenString
                PulseParams[4] = ParamOfGenString

                NewPulseParams = 'pulse(' + ' '.join(PulseParams) + ')'
                UpdateLine = parts[0] + NewPulseParams + '\n'
                UpdatedContent.append(UpdateLine)
            else:
                UpdatedContent.append(line)

        # Write the updated content back to the netlist file
        with open(SimulateNetlist, 'w') as file:
            file.writelines(UpdatedContent)
        return ParamOfGen
    except Exception as e:
        print(f"Error updating parameters of generator: {e}")

# Function to simulate the effects of changing the capacitance in the netlist
@profile
def SimulationCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold):
    circuit = Circuit('Change capacity')
    circuit.include(SimulateNetlist)  # Include the netlist file for simulation

    # Setup and run a transient simulation
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)

    analysis = simulator.transient(**SimParams)

    # Extract output voltage and time data
    OutVoltage = analysis['Vout']
    Time = np.array(analysis.time)
    Voltages = np.array(OutVoltage)
    # Measure the rise edge time to compare against target
    RiseEdge, _ = MeasEdges(Time, Voltages, LowThreshold, HighThreshold)
    time.sleep(2)
    ngspice.destroy()
    time.sleep(2)
    del circuit, simulator, OutVoltage, Time, Voltages, analysis
    gc.collect()


    return RiseEdge  # Return the length of the rise edge for comparison

# Function to update the capacitance value in the netlist
@profile
def ChangeValueOfCapacity(SimulateNetlist, NewValueOfCapacity):
    UpdatedContent = []

    try:
        with open(SimulateNetlist, 'r') as file:
            lines = file.readlines()

        # Modify the capacitance value in the netlist
        for line in lines:
            if line.startswith('C'):
                parts = line.split()
                parts[3] = f"{float(NewValueOfCapacity):.2f}f"
                UpdateLine = ' '.join(parts) + '\n'
                UpdatedContent.append(UpdateLine)
            else:
                UpdatedContent.append(line)

        # Write the updated content back to the netlist file
        with open(SimulateNetlist, 'w') as file:
            file.writelines(UpdatedContent)
    except Exception as e:
        print(f"Error updating netlist: {e}")

# Function to iteratively adjust capacitance to meet a target rise time
@profile
def MeasCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold, TargetRiseTime, LowCapacity, HighCapacity,
                 Toleration):
    OptimalCapacity = None
    # Perform a binary search to find the optimal capacitance
    while (HighCapacity - LowCapacity) > Toleration:
        MiddleCapacity = (LowCapacity + HighCapacity) / 2
        time.sleep(0.5)  # Delay to simulate realistic adjustment time
        ChangeValueOfCapacity(SimulateNetlist, MiddleCapacity)
        time.sleep(0.5)  # Another delay for realistic simulation
        CurrentRiseEdge = SimulationCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold)

        # Adjust the capacitance range based on the measured rise time
        if CurrentRiseEdge > TargetRiseTime:
            HighCapacity = MiddleCapacity
        else:
            LowCapacity = MiddleCapacity

    # Return the middle value as the optimal capacitance after tolerance is reached
    if OptimalCapacity is None:
        OptimalCapacity = (LowCapacity + HighCapacity) / 2
    return OptimalCapacity

# Function to measure delays between input and output signals
@profile
def MeasDelay(Time, InputSignal, OutputSignal, LowThreshold, HighThreshold, Mode):
    Signal50Percent = (LowThreshold + HighThreshold) / 2  # Calculate 50% threshold level
    InputRiseEdge = OutputRiseEdge = InputFallEdge = OutputFallEdge = None

    # Detect rise and fall edges in the input signal
    for i in range(1, len(InputSignal)):
        if InputSignal[i-1] < Signal50Percent <= InputSignal[i] and InputRiseEdge is None:
            InputRiseEdge = Time[i]
        elif InputSignal[i-1] > Signal50Percent >= InputSignal[i] and InputFallEdge is None:
            InputFallEdge = Time[i]
            break

    # Detect rise and fall edges in the output signal based on the mode
    if Mode == "compromise":
        for i in range(1, len(OutputSignal)):
            if OutputSignal[i-1] > Signal50Percent >= OutputSignal[i] and OutputFallEdge is None:
                OutputFallEdge = Time[i]
            elif OutputSignal[i-1] < Signal50Percent <= OutputSignal[i] and OutputRiseEdge is None:
                OutputRiseEdge = Time[i]
                break
    elif Mode == "balanced":
        for i in range(1, len(OutputSignal)):
            if OutputSignal[i-1] < Signal50Percent <= OutputSignal[i] and OutputRiseEdge is None:
                OutputRiseEdge = Time[i]
            elif OutputSignal[i-1] > Signal50Percent >= OutputSignal[i] and OutputFallEdge is None:
                OutputFallEdge = Time[i]
                break

    # Calculate rise and fall delays
    if InputRiseEdge is not None and OutputRiseEdge is not None:
        RiseDelay = abs(OutputRiseEdge - InputRiseEdge)
    else:
        RiseDelay = None

    if InputFallEdge is not None and OutputFallEdge is not None:
        FallDelay = abs(OutputFallEdge - InputFallEdge)
    else:
        FallDelay = None

    return RiseDelay, FallDelay


# Function to update transistor parameters in the netlist
@profile
def UpdateParamsOfDelay(SimulateNetlist, Width, Length):
    UpdateContent = []
    with open(SimulateNetlist, 'r') as file:
        lines = file.readlines()
    TranzistorIndex = 0
    for line in lines:
        if 'XNMOS_Delay' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.startswith('W='):
                    parts[i] = f"W={float(Width[TranzistorIndex]):.2f}"
                elif part.startswith('L='):
                    parts[i] = f"L={float(Length[TranzistorIndex]):.2f}"
            UpdateContent.append(' '.join(parts) + '\n')
        else:
            UpdateContent.append(line)

    with open(SimulateNetlist, 'w') as file:
        file.writelines(UpdateContent)


def UpdateParamsOfEdge(SimulateNetlist, Width, Length):
    UpdateContent = []
    with open(SimulateNetlist, 'r') as file:
        lines = file.readlines()
    TranzistorIndex = 0
    for line in lines:
        if 'XNMOS_Edge' in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.startswith('W='):
                    parts[i] = f"W={float(Width[TranzistorIndex]):.2f}"
                elif part.startswith('L='):
                    parts[i] = f"L={float(Length[TranzistorIndex]):.2f}"
            UpdateContent.append(' '.join(parts) + '\n')
        else:
            UpdateContent.append(line)

    with open(SimulateNetlist, 'w') as file:
        file.writelines(UpdateContent)

# Function to simulate transistors and measure delays and edges
@profile
def SimulationTranzistors(SimulateNetlist, SimParams, LowThreshold, HighThreshold, Mode):
    circuit = Circuit('Simulation transistors')
    circuit.include(SimulateNetlist)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.transient(**SimParams)

    OutVoltage = analysis['Vout']
    Time = np.array(analysis.time)
    Voltages = np.array(OutVoltage)

    Vin = analysis['Va']
    InputSignal = np.array(Vin)

    # Measure delays and edges
    RiseDelay, FallDelay = MeasDelay(Time, InputSignal, Voltages, LowThreshold, HighThreshold, Mode)
    RiseEdge, FallEdge = MeasEdges(Time, Voltages, LowThreshold, HighThreshold)
    print(RiseEdge, FallEdge)
    print(RiseDelay, FallDelay)

    simulator.ngspice.reset()
    del circuit, simulator, OutVoltage, Time, Voltages, analysis
    gc.collect()

    return RiseDelay, FallDelay, RiseEdge, FallEdge

# Function to compute a fitness score for a transistor configuration
@profile
def FitnessScoreSimulation(SimulateNetlist, NewValueOfWidthDelay, NewValueOfLengthDelay,NewValueOfWidthEdge,NewValueOfLengthEdge, SimParams, LowThreshold, HighThreshold, Mode):
    UpdateParamsOfEdge(SimulateNetlist, NewValueOfWidthDelay, NewValueOfLengthDelay)
    time.sleep(2)
    UpdateParamsOfDelay(SimulateNetlist, NewValueOfWidthEdge, NewValueOfLengthEdge)
    time.sleep(1)  # Wait for changes to take effect

    RiseDelay, FallDelay, RiseEdge, FallEdge = SimulationTranzistors(SimulateNetlist, SimParams, LowThreshold, HighThreshold, Mode)
    print(RiseEdge,RiseDelay,FallDelay,FallEdge)
    time.sleep(1)  # Wait for changes to take effect
    Delay = abs(RiseDelay - FallDelay)
    Edge = abs(RiseEdge - FallEdge)

    # Calculate fitness based on mode
    if Mode == "balanced":
        Fitness = abs(Delay + Edge)
    elif Mode == "compromise":
        if RiseDelay < FallDelay:
            MeasErrorDelay = abs(RiseDelay / FallDelay)
        else:
            MeasErrorDelay = abs(FallDelay / RiseDelay)

        if RiseEdge < FallEdge:
            MeasErrorEdge = abs(RiseEdge / FallEdge)
        else:
            MeasErrorEdge = abs(FallEdge / RiseEdge)

        if MeasErrorDelay >= 0.80 and MeasErrorEdge >= 0.80:
            Fitness = abs(MeasErrorDelay - MeasErrorEdge)
        else:
            Fitness = (1 - MeasErrorDelay) + (1 - MeasErrorEdge)

    return Fitness

# Function to count the number of transistors in a netlist
@profile
def CountTranzistorsDelay(SimulateNetlist):
    Count = 0
    try:
        with open(SimulateNetlist, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if 'XNMOS_Delay' in line:
                Count += 1
    except Exception as e:
        print(f"Error: {e}")
    return Count

def CountTranzistorsEdge(SimulateNetlist):
    Count = 0
    try:
        with open(SimulateNetlist, 'r') as file:
            lines = file.readlines()
        for line in lines:
            if 'XNMOS_Edge' in line:
                Count += 1
    except Exception as e:
        print(f"Error: {e}")
    return Count



def AlgorithmDE(SimulateNetlist, SimParams, LowThreshold, HighThreshold, PopSize,
                MaxGenerations, F, CR, WidthMin, WidthMax, LengthBase, LengthIncrease,
                AccuracyThreshold, Thresh, Mode, AcceptThresh):
    TranzistorsDelays = CountTranzistorsDelay(SimulateNetlist)
    TranzistorsEdges = CountTranzistorsEdge(SimulateNetlist)

    # Initialize populations of widths and lengths
    WidthPopulationDelay = np.random.uniform(WidthMin, WidthMax, (PopSize,TranzistorsDelays))
    WidthPopulationEdge = np.random.uniform(WidthMin, WidthMax, (PopSize,TranzistorsEdges))

    LengthPopulationDelay = LengthBase * np.ones((PopSize,TranzistorsDelays))
    LengthPopulationEdge = LengthBase * np.ones((PopSize,TranzistorsEdges))

    # WidthLock = np.zeros((PopSize, TranzistorsN), dtype=bool)
    FitnessHistory = []

    # Iterate over generations
    for Generation in range(MaxGenerations):
        FitnessScore = np.zeros(PopSize)
        # Evaluate the fitness of each individual in the population
        for i in range(PopSize):
            FitnessScore[i] = FitnessScoreSimulation(SimulateNetlist, WidthPopulationDelay[i], LengthPopulationDelay[i],WidthPopulationEdge[i],
                                                     LengthPopulationEdge[i],SimParams, LowThreshold, HighThreshold, Mode)

        # Record the best fitness score of the generation
        FitnessHistory.append(np.min(FitnessScore))
        BestIdx = np.argmin(FitnessScore)


        # Perform mutation, crossover, and selection operations
        for i in range(PopSize):
            IdxsDelay = [idxD for idxD in range(PopSize) if idxD != i]
            DelayA, DelayB, DelayC = np.random.choice(IdxsDelay, 3, replace=False)

            IdxsEdge = [idxE for idxE in range(PopSize) if idxE != i]
            EdgeA, EdgeB, EdgeC = np.random.choice(IdxsEdge, 3, replace=False)

            # Generate trial individual
            MutationWidthDelay = np.clip(WidthPopulationDelay[DelayA] + F * (WidthPopulationDelay[DelayB] - WidthPopulationDelay[DelayC]), WidthMin, WidthMax)
            CrossingDelaz = np.random.rand(TranzistorsDelays) < CR
            TrialWidthDelay = np.where(CrossingDelaz, MutationWidthDelay, WidthPopulationDelay[i])
            TrialLengthDelay = np.copy(LengthPopulationDelay[i])

            MutationWidthEdge = np.clip(WidthPopulationEdge[EdgeA] + F * (WidthPopulationEdge[EdgeB] - WidthPopulationEdge[EdgeC]), WidthMin, WidthMax)
            CrossingEdge = np.random.rand(TranzistorsEdges) < CR
            TrialWidthEdge = np.where(CrossingEdge, MutationWidthEdge, WidthPopulationEdge[i])
            TrialLengthEdge = np.copy(LengthPopulationEdge[i])

            if Mode == "balanced":
                for j in range(TranzistorsDelays):
                    if TrialWidthDelay[j] == WidthMin and FitnessScore[i] > Thresh:
                        TrialLengthDelay[j] += LengthIncrease

            elif Mode == "compromise":
                for j in range(TranzistorsDelays):
                    if TrialWidthDelay[j] == WidthMin and FitnessScore[i] >= 0.3:
                        TrialLengthDelay += LengthIncrease

            # Evaluate the trial individual
            TrialFitnessScore = FitnessScoreSimulation(SimulateNetlist, TrialWidthDelay, TrialLengthDelay,TrialWidthEdge,TrialLengthEdge, SimParams, LowThreshold,
                                                       HighThreshold, Mode)

            # Selection step
            if TrialFitnessScore < FitnessScore[i]:
                WidthPopulationDelay[i], LengthPopulationDelay[i], FitnessScore[i] = TrialWidthDelay, TrialLengthDelay, TrialFitnessScore



        # Logging the best fitness of the current generation
        print(f"Generation {Generation}: Best Fitness Score = {np.min(FitnessScore)}")
        UpdateParamsOfEdge(SimulateNetlist, WidthPopulationEdge[BestIdx], WidthPopulationEdge[BestIdx])
        UpdateParamsOfDelay(SimulateNetlist, WidthPopulationDelay[BestIdx], WidthPopulationDelay[BestIdx])


        # Check for convergence
        if (Mode == "balanced" and np.min(FitnessScore) < AccuracyThreshold) or \
           (Mode == "compromise" and np.min(FitnessScore) <= AcceptThresh):
            UpdateParamsOfEdge(SimulateNetlist, WidthPopulationEdge[BestIdx], WidthPopulationEdge[BestIdx])
            UpdateParamsOfDelay(SimulateNetlist, WidthPopulationDelay[BestIdx], WidthPopulationDelay[BestIdx])
            print(f"Required accuracy {np.min(FitnessScore)} achieved in {Generation} generation.")
            break

    # Plot the evolution of the best fitness score over generations
    plt.figure(figsize=(10, 6))
    plt.plot(FitnessHistory, marker='o', linestyle='-', color='b')
    plt.title('The Evolution of the Best Fitness Over the Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Score')
    plt.grid(True)
    plt.savefig('AND.png')

# Main function to set up and run the simulations
def main():
    # Define file paths and thresholds
    NetlistChain = '/home/vboxuser/PycharmProjects/Chain.cir'
    SimulateNetlist ='/home/vboxuser/PycharmProjects/AND.cir'
    LowThreshold = 0.18
    HighThreshold = 1.62

    # Define simulation parameters for different experiments
    LowCapacity = 1
    HighCapacity = 60
    Toleration = 0.01

    PopSize = 10
    MaxGen = 25
    F = 0.8
    CR = 0.9
    WMin = 0.42
    WMax = 1.07
    LBase = 0.15
    LIncrease = 0.05
    AcThreshold = 1e-12
    Thresh = 30e-12
    Mode = "balanced"
    OkThresh = 2e-12

    # Define simulation steps and timing parameters
    SimParams = {
        'step_time': 1e-12,
        'end_time': 2.5e-9
    }

    # Execute the simulation chain for inverters and adjust parameters based on results
    RiseTimeChain, FallTimeChain = SimulationChainOfInverters(NetlistChain, LowThreshold, HighThreshold, SimParams)
    ChangeParamsOfGen(RiseTimeChain, FallTimeChain, SimulateNetlist)
    TargetRiseTime = (RiseTimeChain + FallTimeChain) / 2
    SimulationCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold)
    MeasCapacity(SimulateNetlist, SimParams, LowThreshold, HighThreshold, TargetRiseTime, LowCapacity, HighCapacity,
                 Toleration)
    SimulationTranzistors(SimulateNetlist, SimParams, LowThreshold, HighThreshold, Mode)

    # Run the differential evolution algorithm for optimization
    AlgorithmDE(SimulateNetlist, SimParams, LowThreshold, HighThreshold, PopSize,
                MaxGen, F, CR, WMin, WMax, LBase, LIncrease,
                AcThreshold, Thresh, Mode, OkThresh)

if __name__ == "__main__":
    main()
