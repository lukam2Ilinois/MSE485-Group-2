#https://pymatgen.org/pymatgen.io.lammps.html
#Reference for the Pymatgen functions for making and parsing LAMMPS files


from pymatgen.io.lammps.outputs import parse_lammps_log,parse_lammps_dumps
from pymatgen.io.lammps.inputs import LammpsInputFile
from pymatgen.io.lammps.data import LammpsBox,LammpsData
from pandas import DataFrame
import numpy as np



def makeInFile(folder,temperature,thermoInterval=10000,dumpInterval=100,runtime=1000000):
	"""
	Writes the run_lammps.in file for the LAMMPS run
	args:
		folder         : the directory where the file will be written
		temperature    : simulation temperature
		thermoInterval : thermodynamic data will be written to the log every thermoInterval time steps (default 10000)
		dumpInterval   : structure is written to the dump file every structureDump time steps (default 100)
		runtime        : number of simulation time steps (default 1000000)
	returns:
		None
	"""
	seed=np.random.randint(10000000)
	inputFile=LammpsInputFile()
	inputFile.add_stage(
		stage_name="STYLES",commands=[
			"units metal",
			"atom_style atomic"
	])
	inputFile.add_stage(
		stage_name="SET UP CALCULATION",commands=[
			"read_data starting_config.dat",
			"pair_style sw",
			"pair_coeff * * Si.sw Si",
			f"velocity all create {temperature} {seed}"
	])
	inputFile.add_stage(
		stage_name="CALCULATION PARAMETERS",commands=[
			f"fix f1 all nvt temp {temperature:.1f} {temperature:.1f} $(100.0*dt)",
			f"thermo {thermoInterval}",
	])
	inputFile.add_stage(
		stage_name="RUN CALC AND WRITE OUTPUTS",commands=[
			f"dump d1 all custom {dumpInterval} state_dump id type x y z vx vy vz",
			f"run {runtime}",
			"write_data FINAL_STRUCTURE"
	])
	inputFile.write_file(f"{folder}/run_lammps.in")



def makeDatFile(folder,supercellSize=4,defect=None):
	"""
	Writes the starting_config.dat file for the LAMMPS run
	args:
		folder        : the directory where the file will be written
		supercellSize : dimension of cubic array. Supercell contains 8 * supercellSize**3 atoms (default 4)
		defect        : options None - no defect, "V" - one vacancy, "I" - one interstitial (default None)
	returns:
		None
	"""
	fractionalPositions=[
		np.array([0.75,0.75,0.25]),
		np.array([0.00,0.50,0.50]),
		np.array([0.75,0.25,0.75]),
		np.array([0.00,0.00,0.00]),
		np.array([0.25,0.75,0.75]),
		np.array([0.50,0.50,0.00]),
		np.array([0.25,0.25,0.25]),
		np.array([0.50,0.00,0.50])
	]
	a=5.4437025
	m=28.085
	box=LammpsBox(bounds=[[0,a*supercellSize]]*3)

	atomData=[]
	for xx in range(supercellSize):
		for yy in range(supercellSize):
			for zz in range(supercellSize):
				for pos in fractionalPositions:
					newPos=a*(pos+np.array([xx,yy,zz]))
					atomData.append([1,newPos[0],newPos[1],newPos[2]])

	if(defect=="V"):
		atomData.pop(len(atomData)//2)
	elif(defect=="I"):
		atomData.append([len(atomData)+1,0.5*a,0.5*a,0.5*a])
	
	data=LammpsData(
		box=box,
		masses=DataFrame(data=[m],columns=["mass"],index=[1]),
		atoms=DataFrame(data=atomData,columns=["type","xs","ys","zs"],index=range(1,len(atomData)+1))
	)
	data.write_file(f"{folder}/starting_config.dat")



def extractOutputs(folder,dumpFile="state_dump"):
	"""
	Extract the states from the dump file of a LAMMPS run
	args:
		folder   : the directory containing the dump file
		dumpFile : the name of the file in which the states are saved (default state_dump)
	returns:
		Generator object returning data frames of particle positions and velocities at specific time steps
		**NOTE** : This method returns a generator object rather than a list. 
				   list(extractOutputs(...)) gives the full list of time steps.
	"""
	data=parse_lammps_dumps(f"{folder}/{dumpFile}")
	return data
