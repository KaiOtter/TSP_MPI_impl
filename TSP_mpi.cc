
// Brute Force Travelling Salesman, OMP/MPI Version
//
// PROJECT - by Jeff Adkisson & Kaidi Chen
//         - for CS 7172 Parallel, Dr. Gayler Fall 2017, KSU
//         - NOTE: ONLY TESTED ON SHPC CLUSTER
//
// To compile:
// mpiCC project_omp_mpi.cc -o project_omp_mpi -g -Wall -fopenmp -std=c++0x
//
// USAGE:
// 
// mpirun -np <PROCESSES> ./project_omp_mpi <THREADS> <TOUR_SIZE> [HOME_CITY default=9,Atlanta] [RANDOM_SEED default=1]
// 
//  - PROCESSES <REQUIRED> number of MPI processes to spawn
//
//  - THREADS <REQUIRED> number of OpenMP threads to spawn per process
//
//  - TOUR_SIZE <REQUIRED> is the number of cities to visit
// 
//  - HOME_CITY [OPTIONAL] is the *index* of the city to start and end, defaults to [9, Atlanta]
//    * see cities.txt for list of cities.
//    * HOME_CITY must be between 0 and 255.
// 
//  - RANDOM_SEED [OPTIONAL ]sets the random number generator to choose cities for you.
//    * choose the same seed to choose the same cities for a given TOUR_SIZE
//      and HOME_CITY
// 
// FOR EXAMPLE:
// 
// mpirun -np 4 ./project_omp_mpi 16 4 9 1
// - this will spawn 4 mpi processs, 16 omp threads per process for a city tour of 4 
//   where the home city is ID 9 and the random seed is 1
//
// DEPENDENCIES:
// - cities.txt    : List of 256 possible cities to visit. 
//                   Use index of a row in cities.txt to choose a HOME_CITY.
// - distances.txt : A 256x256 matrix in vector form of distances between cities.
//                   (0,0) corresponds to cities.txt[0], 
//                   (1,1) corresponds to cities.txt[1], etc.

#include <vector>
#include <stack>
#include <queue>
#include <array>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <climits>
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

//globals set by command line args
unsigned int _tourSize = 0;
unsigned int _arrayTourSize = 0;
unsigned int _requestedHomeTown = 9;
unsigned long _randomSeed = 1;

//a threshold for allocate the length of MPI tasks 
unsigned int _minProcessTasks = 4;

//array position constants
const unsigned int POS_CITIES_IN_TOUR = 0;
const unsigned int POS_TOUR_TOTAL_DIST = 1;
const unsigned int POS_LAST_CITY_DIST = 2;
const unsigned int POS_TOUR_START = 3;

//hometown id in submatrix
const unsigned int HOMETOWN = 0;

//argument position constants
const signed int ARGS_MIN = 2;
const signed int ARGS_MAX = 4;
const signed int ARGS_FILENAME = 0;
const signed int ARGS_THREADS = 1;
const signed int ARGS_TOURSIZE = 2;
const signed int ARGS_HOMETOWN = 3;
const signed int ARGS_RANDOMSEED = 4;

//omp globals
unsigned int _ompThreads; 
unsigned int _ompRank;

//mpi globals
int _mpi_world_rank;
char _mpi_processor_name[MPI_MAX_PROCESSOR_NAME];
int _mpi_world_size;
unsigned int _lastDistSent = INT_MAX;

//mpi constants
const int MPI_BEST_TOUR_TAG = 10211970;


//--------------------------------------------------------------
//splits the work from getToursForTask into a single task's vectors to be added to that tasks's tour stack
vector<vector<int>> getToursForTask(int tasks, int thisTask, vector<int> &tours)
{
	vector<vector<int>> results;
	unsigned int maxLength = tours.back();
	for (unsigned int i = thisTask * maxLength; i < tours.size() - 1; i += maxLength * tasks)
	{
		vector<int> newVector;
		for (unsigned int inner = 0; inner < maxLength; inner++)
		{
			if (tours[i + inner] >= 0)
			{
				newVector.push_back(tours[i + inner]);
			}
		}
		if (newVector.size() > 0) {
			results.push_back(newVector);
		}  
	}
	return results;
}

//--------------------------------------------------------------
// accepts one or more starting tour positions and returns a vector of tours split over n tasks.
// use getToursForTask to split the tour vector into specific work for a process or thread
vector<int> splitTours(vector<vector<int>> &startFrom, unsigned int tasks, unsigned int tourSize) {
	unsigned int maxLength = 1;

	//init queue
	queue<vector<int>> queue;
	
	//setup vector start
	for (size_t v = 0; v < startFrom.size(); v++)
	{
		queue.push(startFrom[v]);
		maxLength = maxLength < startFrom[v].size() ? startFrom[v].size() : maxLength;
	}	

	bool contBfs = true;
	while (tasks > 1 && tourSize > 3 && queue.size() > 0 && queue.size() <= tasks && contBfs)
	{
		vector<int> current = queue.front();
		queue.pop();

		for (int nbr = tourSize - 1; nbr >= 1; nbr--)
		{
			vector<int> currentCopy = current;
			bool contains = false;
			bool allContains = true;
			for (unsigned int idx = 0; idx < currentCopy.size(); idx++)
			{
				contains = currentCopy[idx] == nbr;
				if (contains) break;
			}
			if (!contains)
			{
				allContains = false;
				currentCopy.push_back(nbr);
				queue.push(currentCopy);
				if (currentCopy.size() > maxLength) maxLength = currentCopy.size();
			}
			if (allContains && current.size() == tourSize) {
				queue.push(current);
				contBfs = false;
				break;
			}
		}
	}

	//create padded list where -1 = padding where length < maxLength so all tours are maxLength
	vector<int> tourData;
	while (queue.size() > 0)
	{
		vector<int> current = queue.front();
		queue.pop();

		tourData.insert(tourData.end(), current.begin(), current.end());
		for (unsigned int i = 0; i < maxLength - current.size(); i++)
		{
			tourData.push_back(-1);
		}
	}

	//add padding that indicates this should be ignored by target to make it easy to split via MPI
	while (tourData.size() % tasks != 0)
	{
		for (unsigned int i = 0; i < maxLength; i++)
		{
			tourData.push_back(-2);
		}
	}

	//add maxlength to end
	tourData.push_back(maxLength);

	return tourData;
}

//--------------------------------------------------------------
//if this is process 0, returns true
bool isRootProcess() {
  return _mpi_world_rank == 0;
}

//--------------------------------------------------------------
//load contents of cities.txt into a string vector
vector<string> loadAllCities() {
  vector<string> allCities;
  ifstream citiesFile("cities.txt");
  string city;
  if(!citiesFile) 
    {
        cerr << "Error opening cities.txt file"<< endl;
        exit(1);
    }
    while (getline(citiesFile, city))
    {
        allCities.push_back(city);
    }    
    citiesFile.close();
    return allCities;
}

//--------------------------------------------------------------
//load contents of distances.txt into a int vector
vector<int> loadAllDistances() {
  vector<int> distances;
  ifstream distancesFile("distances.txt");
  string distance;
  if(!distancesFile) 
    {
        cerr << "Error opening distances.txt file"<< endl;
        exit(1);
    }
    while (getline(distancesFile, distance))
    {
        distances.push_back(stoi(distance));
    }    
    distancesFile.close();
    return distances;
}

//--------------------------------------------------------------
//pick the hometown, plus tourSize - 1 additional psuedo-random cities
vector<int> pickCities(int cityCount) {
  vector<int> pickedCities;
  int randomCity = 0;
  srand(_randomSeed * _tourSize + _requestedHomeTown);
  pickedCities.push_back(_requestedHomeTown);
  
  while(pickedCities.size() < _tourSize) {
    randomCity = rand() % (cityCount - 1); //between 0 and maxIndex - 1
    bool notUsed = true;
    for (unsigned int i = 0; notUsed && i < pickedCities.size(); i++) {
      notUsed = pickedCities[i] != randomCity;
    }
    if (notUsed) pickedCities.push_back(randomCity);
  }
  return pickedCities;
}

//--------------------------------------------------------------
//get names of cities we picked in pickCities function
vector<string> getCityNames(vector<string> &allCities, vector<int> &pickedCities) {
  vector<string> pickedCityNames;
  for (unsigned int i = 0; i < pickedCities.size(); i++) {
    pickedCityNames.push_back(allCities[pickedCities[i]]);
  }
  return pickedCityNames;
}

//--------------------------------------------------------------
//create a smaller matrix of the distances between the cities we picked
vector<int> getPickedCityMatrix(int cityCount, vector<int> &allDistances, vector<int> &pickedCities) {
  vector<int> distances(pow(pickedCities.size(), 2));
  
	for (unsigned int fromCity = 0; fromCity < _tourSize; fromCity++)
	{
		unsigned int originalRow = (pickedCities[fromCity] * cityCount);
		for (unsigned int toCity = 0; toCity < _tourSize; toCity++)
		{			
			unsigned int originalCol = (pickedCities[toCity]);
			unsigned int transferValue = allDistances[originalRow + originalCol];
			distances[(fromCity * _tourSize) + toCity] = transferValue;
		}
	}  
  return distances;
}

//--------------------------------------------------------------
//retrieves a distance value from a city to a city from the distances vector.
unsigned int getDistance(const int *distances, unsigned int fromCity, unsigned int toCity)
{
  return distances[(fromCity * _tourSize) + toCity];
}

//pushes a copy of the current tour onto the tour stack
void pushCopy(stack<unsigned int*> *tourStack, const unsigned int *tour) {
	unsigned int* tourCopy = new unsigned int[_arrayTourSize];
	for (unsigned int idx = 0; idx < _arrayTourSize; idx++) {
		tourCopy[idx] = tour[idx];
	}
	tourStack->push(tourCopy);
}

//--------------------------------------------------------------
//adds a city to the current tour and updates the various
//metadata about the tour - distance, last distance, city count
void addCity(const int *distances, unsigned int *currentTour, int city) {
	int newCityPos = currentTour[POS_CITIES_IN_TOUR] + POS_TOUR_START;
	unsigned int distance = getDistance(distances, currentTour[newCityPos - 1], city);
	currentTour[newCityPos] = city;
	currentTour[POS_CITIES_IN_TOUR]++;	
	currentTour[POS_TOUR_TOTAL_DIST] += distance;
	currentTour[POS_LAST_CITY_DIST] = distance;
}

//--------------------------------------------------------------
//removes the last city from the current tour and updates 
//metadata (tour distance, city count)
void removeLastCity(unsigned int *currentTour) {
	int lastCityPos = currentTour[POS_CITIES_IN_TOUR] + POS_TOUR_START - 1;
	currentTour[lastCityPos] = 0;
	currentTour[POS_CITIES_IN_TOUR]--;
	currentTour[POS_TOUR_TOTAL_DIST] -= currentTour[POS_LAST_CITY_DIST];
}

//--------------------------------------------------------------
//accepts a tour vector that was configured by splitTours and
//transforms it into a properly configured array the DFS can
//use when the array is added to the process or thread stack
unsigned int* configureTourArray(vector<int> tour, const int *distances) {
	
	unsigned int* currentTour = new unsigned int[_arrayTourSize];
  for (unsigned int idx = 0; idx < _arrayTourSize; idx++) {
		currentTour[idx] = 0;
	}
	
	currentTour[POS_TOUR_TOTAL_DIST] = 0;
  currentTour[POS_CITIES_IN_TOUR] = tour.size();
  currentTour[POS_TOUR_START] = HOMETOWN;
	currentTour[_arrayTourSize - 1] = HOMETOWN; //end is always hometown
	
	for (unsigned int idx = 1; idx < tour.size(); idx++) {
		int distance = getDistance(distances, tour[idx - 1], tour[idx]);
		currentTour[POS_TOUR_START + idx] = tour[idx]; //add city
		currentTour[POS_TOUR_TOTAL_DIST] += distance;
		currentTour[POS_LAST_CITY_DIST] = distance;
	}
	
	return currentTour;
}

//--------------------------------------------------------------
//adds tours for this process to the stack
void populateTourStackForProcess(stack<unsigned int*> *tourStack, vector<int> &tourVector, const int *distances) {
	
	vector<vector<int>> forProcess = getToursForTask(_mpi_world_size, _mpi_world_rank, tourVector);	
	
 // cout<<"rank "<< _mpi_world_rank<<"  size "<< forProcess.size()<<endl;
 // for(unsigned int i =0; i<forProcess.size();i++){
   // for(unsigned int k =0; k <forProcess[i].size();k++){
  //    cout<<forProcess[i][k]<<"  ";
 //   }
  //  cout<<endl;
 // }
	vector<int> perThreadTourVector = splitTours(forProcess, _ompThreads, _tourSize);
	//cout<<"rank " <<_mpi_world_rank<<" ThreadToursize "<< perThreadTourVector.size()<<endl;
  //for(unsigned int k =0; k <perThreadTourVector.size();k++){
  //  cout<<perThreadTourVector[k]<<"  ";
  //}
  //cout<<endl;
 
	vector<vector<int>> forAllThreads;
	for	(unsigned int thread = 0; thread < _ompThreads; thread++) {
		vector<vector<int>> forProcessThread = getToursForTask(_ompThreads, thread, perThreadTourVector);
		for (unsigned int idx = 0; idx < forProcessThread.size(); idx++) {
			forAllThreads.push_back(forProcessThread[idx]);
		}
	}

	cout<<"rank "<< _mpi_world_rank<<"  forAllThreads "<< forAllThreads.size()<<endl;
  //for(unsigned int i =0; i<forAllThreads.size();i++){
   // for(unsigned int k =0; k <forAllThreads[i].size();k++){
   //   cout<<forAllThreads[i][k]<<"  ";
  //  }
  //  cout<<endl;
  //}

	//cout << "Process " << _mpi_world_rank << endl;
	for (size_t j = 0; j < forAllThreads.size(); j++)
	{
		unsigned int* currentTour = configureTourArray(forAllThreads[j], distances);
		tourStack->push(currentTour);
		//cout << "- Tour " << _mpi_world_rank + 1 << "." << j + 1 << " of " << forAllThreads.size() << endl;
		// for (size_t k = 0; k < forallthreads[j].size(); k++)
		// {
			// cout << "  * " << forallthreads[j][k] << endl;
		// }
	}
	// cout << "  * size=" << tourStack->size() << endl;
}

//a combination algorithm
void Cij(unsigned int i,unsigned int j,vector<int> &r,unsigned int num,vector<vector<int> > & result)  
{  
        //cout << n << ' ' << i << ' ' << j << endl;  
        if (j == 1)  
        {  
                for (unsigned int k = 1; k < i; k++)  
                {  
                        vector<int> temp(num);  
                        r[num - 1] = k;  
                        for (unsigned int i = 0; i < num;i++)  
                        {  
                                temp[i]=r[i];  
                                //cout << r[i] << ' ';  
                        }  
                        result.push_back(temp);  
                        //cout << endl;  
                }  
        }  
        else if (j == 0)  
        {  
                //do nothing!  
        }  
        else  
        {  
                for (unsigned int k = i; k >= j; k--)  
                {  
                        r[j-2] = k-1;  
                        Cij(k - 1, j - 1,r,num,result);  
                }  
        }  
}  

//split work for MPI more balance
vector<int> splitForMPI(unsigned int tasks, unsigned int tourSize) {
	unsigned int k = 1;
	unsigned int maxLength = 1;
	vector<int> result;
	while(k < tasks* _minProcessTasks){
		if(tourSize == maxLength){
			break;
		}
		k = k * (tourSize-maxLength);
		maxLength ++;	
	}
	if(maxLength == 1){
		result.push_back(0);
	}
	else{
		vector<int> r(maxLength-1);
		vector<vector<int>> re;
		vector<vector<int>> re_permut;
		Cij(tourSize, maxLength-1, r, maxLength-1, re);
     
     
		for(unsigned int i=0;i<re.size();i++){
			vector<int> temp =re[i];
			sort(temp.begin(),temp.end());
			do{
				re_permut.push_back(temp);
			}while(next_permutation(temp.begin(),temp.end()));
		}
    random_shuffle(re_permut.begin(), re_permut.end());
		for(unsigned int k=0;k<re_permut.size();k++){
			result.push_back(0);
			for(unsigned int t=0;t<maxLength-1;t++){
				result.push_back(re_permut[k][t]);
			}
		}

   // for(unsigned int idx=0; idx<re_permut.size();idx++){
   //   for(unsigned int t=0; t<re_permut[idx].size();t++){
    //    cout<<re_permut[idx][t]<<" ";
   //   }
    //  cout<<endl;
  //  }
	}
	result.push_back(maxLength);
	return result;
}

//--------------------------------------------------------------
//configures the tour stack, best tour and current tour
void initializeStack(stack<unsigned int*> *tourStack, unsigned int *bestTour, const int *distances) {
	unsigned int* currentTour = new unsigned int[_arrayTourSize];
  for (unsigned int idx = 0; idx < _arrayTourSize; idx++) {
		currentTour[idx] = 0;
		bestTour[idx] = 0;
	}
	
	bestTour[POS_TOUR_TOTAL_DIST] = INT_MAX;
	bestTour[POS_CITIES_IN_TOUR] = _tourSize;
	bestTour[POS_TOUR_START] = HOMETOWN;
	bestTour[_arrayTourSize - 1] = HOMETOWN; //end is always hometown
	
	currentTour[POS_TOUR_TOTAL_DIST] = 0;
  currentTour[POS_CITIES_IN_TOUR] = 1;
  currentTour[POS_TOUR_START] = HOMETOWN;
	currentTour[_arrayTourSize - 1] = HOMETOWN; //end is always hometown
	
	vector<vector<int>> startFrom;
	int toursToBcastSize;
	int* toursToBcast;
	
	if (isRootProcess()) {
		vector<int> fromHome;
		fromHome.push_back(HOMETOWN);
		startFrom.push_back(fromHome);
		//vector<int> tourDistribution = splitTours(startFrom, _mpi_world_size, _tourSize);
    vector<int> tourDistribution = splitForMPI(_mpi_world_size, _tourSize);
    toursToBcastSize = tourDistribution.size();
		toursToBcast = new int[toursToBcastSize];
		
		//copy vector to array for broadcast... vector.data() doesn't work quite right on cluster...
		for (int idx = 0; idx < toursToBcastSize; idx++) {
			toursToBcast[idx] = tourDistribution[idx];
		}
	}
	
	//send tours to perform to all mpi processes
	MPI_Bcast(&toursToBcastSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (!isRootProcess()) toursToBcast = new int[toursToBcastSize];
	MPI_Bcast(toursToBcast, toursToBcastSize, MPI_INT, 0, MPI_COMM_WORLD);

 
	//copy into tour vector
	vector<int> tourVector;
	for (int idx = 0; idx < toursToBcastSize; idx++) {
		tourVector.push_back(toursToBcast[idx]);
		//cout << _mpi_world_rank << " -- " << idx << " -- " << toursToBcast[idx] << endl;
	}
	//push tours into local process stack
	populateTourStackForProcess(tourStack, tourVector, distances);
	
	//clean up
	delete toursToBcast;
}

void saveBestTour(unsigned int* currentTour, unsigned int *bestTour) {
	#pragma omp critical(saveBestTour)
	{
		for (unsigned int idx = 0; idx < _arrayTourSize; idx++) {
			bestTour[idx] = currentTour[idx];
		}
	}
}

//--------------------------------------------------------------
//updates best tour, plus checks for best tour changes from 
//other MPI processes
void updateBestTour(unsigned int* currentTour, unsigned int *bestTour, int threadNum, bool onlyCheckOtherProcesses = false) {
	
	if (!onlyCheckOtherProcesses) {
		saveBestTour(currentTour, bestTour);
		
		//send best tour to other processes
		if (threadNum == 0 && _mpi_world_size > 0) {
			#pragma omp critical(sendBestTour) 
			{
				if (_lastDistSent > bestTour[POS_TOUR_TOTAL_DIST]) {
					for (int dest = 0; dest < _mpi_world_size; dest++) {
						if (dest != _mpi_world_rank) {
							//cout << "sending " << bestTour[POS_TOUR_TOTAL_DIST] << " to " << dest << " from " << _mpi_world_rank << endl;
							MPI_Bsend(bestTour, _arrayTourSize, MPI_INT, dest, MPI_BEST_TOUR_TAG, MPI_COMM_WORLD);
						}
					}
				}
				_lastDistSent = bestTour[POS_TOUR_TOTAL_DIST];
			}
		}
	} 
	
	//if thread 0, see if best tour updates are coming from other processes
	if (threadNum == 0 && _mpi_world_size > 0) {
		#pragma omp critical(sendBestTour) 
		{
			int msgAvail = false;
			MPI_Status status;
			MPI_Iprobe(MPI_ANY_SOURCE, MPI_BEST_TOUR_TAG, MPI_COMM_WORLD, &msgAvail, &status);
			unsigned int *maybeNewBestTour = new unsigned int[_arrayTourSize];
			while (msgAvail) {
				MPI_Recv(maybeNewBestTour, _arrayTourSize, MPI_INT, status.MPI_SOURCE, MPI_BEST_TOUR_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				//cout << _mpi_world_rank << " receiving " << maybeNewBestTour[POS_TOUR_TOTAL_DIST] << " from " << status.MPI_SOURCE << endl;
				if (maybeNewBestTour[POS_TOUR_TOTAL_DIST] < bestTour[POS_TOUR_TOTAL_DIST]) {
					// cout << "new best tour on " << _mpi_world_rank << " from "
							// << status.MPI_SOURCE << " --- " << maybeNewBestTour[POS_TOUR_TOTAL_DIST] 
							// << ", was " << bestTour[POS_TOUR_TOTAL_DIST] << endl;
					saveBestTour(maybeNewBestTour, bestTour);
				}
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_BEST_TOUR_TAG, MPI_COMM_WORLD, &msgAvail, &status);
			}
			delete maybeNewBestTour;
		}
	}
}

//--------------------------------------------------------------
//does the actual work of the brute force traversal of all 
//paths through all picked cities. returns number of iterations.
unsigned long startFindingTour(const int *distances, unsigned int *bestTour)
{ 
	stack<unsigned int*> sharedTourStack;
	unsigned int threadNeedsWork = 0;
	
	initializeStack(&sharedTourStack, bestTour, distances);
	
	//create vector of stacks we can use for each thread so
	//there is no need to lock the stacks when pushing and popping
	vector<stack<unsigned int*>> perThreadTourStacks;
	unsigned int thread;
	for (thread = 0; thread < _ompThreads; thread++) {
		perThreadTourStacks.push_back(stack<unsigned int*>());
	}
	thread = 0;
	while (!sharedTourStack.empty()) {
		perThreadTourStacks[thread].push(sharedTourStack.top());
		sharedTourStack.pop();
		thread = (thread + 1) % _ompThreads;
	}
	
	//setup mpi buffering for async calls for best tour
	char buffer[1048576];
	char* buf;
	int buf_size;
	MPI_Buffer_attach(buffer,1048576);
	
	unsigned long localGlobalIterations = 0;
	
	//start processing the stack
	#pragma omp parallel num_threads(_ompThreads) default (none) shared(perThreadTourStacks, bestTour, distances, localGlobalIterations, _tourSize, _mpi_world_size, _mpi_world_rank, cout, threadNeedsWork, _ompThreads, _arrayTourSize, sharedTourStack)
  {
		int threadNum = 0;
		#ifdef _OPENMP
			threadNum = omp_get_thread_num();
		#endif 
		stack<unsigned int*> threadTourStack = perThreadTourStacks[threadNum];
		
		bool continueProcessing = true;
		bool askedForWork = false;
		bool contains;
		unsigned int nbr;
		unsigned long iterations = 0;
		unsigned int* currentTour;
		bool justShared = false;
		while (continueProcessing)
		{
			if (threadTourStack.empty()) {
				//out of work... ask for more unless everyone is waiting for more
				if (!askedForWork) {
					#pragma omp critical (threadNeedsWork) 
					{
						threadNeedsWork++;
					}
					askedForWork = true;
					//cout << "Thread " << threadNum << " asking for work, " << threadNeedsWork << " threads asking" << endl;
				}
				if (sharedTourStack.size() > 0) {
					#pragma omp critical (threadNeedsWork) 
					{
						if (sharedTourStack.size() > 0) 
						{
							askedForWork = false;
							currentTour = sharedTourStack.top();
							sharedTourStack.pop();
							//cout << "Thread " << threadNum << " resuming work, " << threadNeedsWork << " threads asking" << endl;
						}
					}
				}
				
				//all threads continue spinning until the sharedTourStack queue is empty and all threads are asking for work
				continueProcessing = threadNeedsWork < _ompThreads || sharedTourStack.size() > 0;
				
			} else {
				
				//get current tour
				currentTour = threadTourStack.top();
				threadTourStack.pop();
				
				//if another thread is asking for work and this one has more, share via sharedTourStack
				if (!threadTourStack.empty() && threadNeedsWork > 0 && !justShared) {
					#pragma omp critical(threadNeedsWork) 
					{
						if (threadNeedsWork > 0) {
							//cout << "Thread " << threadNum << " sharing work, " << threadNeedsWork << " threads asking" << endl;
							sharedTourStack.push(threadTourStack.top());
							threadTourStack.pop();
							justShared = true;
							threadNeedsWork--;
						}
					}
				}	else if (justShared) {
					//don't share again until another full iteration... let another thread share
					justShared = false;
				}					
			}
			
			//
			if (continueProcessing && !askedForWork) {			
				//every 2500 iterations, take a moment to see if a new best tour has arrived
				//otherwise, we'll only check when this process finds a new best tour
				iterations++;
				if (iterations % 2500 == 0 && threadNum == 0 && _mpi_world_size > 0) 
				{ 
					updateBestTour(bestTour, bestTour, threadNum, true);
				}
				
				if (currentTour[POS_CITIES_IN_TOUR] == _tourSize) {
					//if tour is max size, add the home city, then see if we have a new best tour
					addCity(distances, currentTour, HOMETOWN);
					if (currentTour[POS_TOUR_TOTAL_DIST] < bestTour[POS_TOUR_TOTAL_DIST]) {
						updateBestTour(currentTour, bestTour, threadNum);
					}
				} 
				else if (currentTour[POS_TOUR_TOTAL_DIST] < bestTour[POS_TOUR_TOTAL_DIST]) 
				{	//stop processing current tour if distance exceeds current best tour			
					for (nbr = _tourSize - 1; nbr >= 1; nbr--) {
						//make sure we're not adding city to stack that makes invalid tour 
						//(visited more than once)
						for (unsigned int pos = POS_TOUR_START; 
								 pos <= currentTour[POS_CITIES_IN_TOUR] + POS_TOUR_START; 
								 pos++)
						{
							contains = currentTour[pos] == nbr;
							if (contains) break;
						}
						if (!contains)
						{
							addCity(distances, currentTour, nbr);
							pushCopy(&threadTourStack, currentTour);
							removeLastCity(currentTour);
						}
					}
				}
				delete currentTour;
			}
		}
		#pragma omp critical(globalIterations)
		{
			localGlobalIterations += iterations;
		}
	}
	
	//update best tour one last time on root in case root never checked
	MPI_Buffer_detach(&buf, &buf_size);
	MPI_Barrier(MPI_COMM_WORLD);
	updateBestTour(bestTour, bestTour, 0, isRootProcess());

	//summarize the iterations across all processes
	unsigned long globalIterations = 0;
	MPI_Reduce(&localGlobalIterations, &globalIterations, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  return globalIterations;
}

//--------------------------------------------------------------
//transmits the distance vector to all mpi processes
void getDistances(int* distances, vector<string> &pickedCityNames) {
	int distSize;
	if (isRootProcess()) {
		//load the data - city names and matrix of city distances
		vector<string> allCities = loadAllCities();
		vector<int> allDistances = loadAllDistances();
		vector<int> pickedCities = pickCities(allCities.size());
		pickedCityNames = getCityNames(allCities, pickedCities);
		vector<int> distanceVector = getPickedCityMatrix(allCities.size(), allDistances, pickedCities);
		copy(distanceVector.begin(), distanceVector.end(), distances);
		distSize = distanceVector.size();
	} 
	
	//send distance vector to all mpi processes
	MPI_Bcast(&distSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(distances, distSize, MPI_INT, 0, MPI_COMM_WORLD);	
}

//--------------------------------------------------------------
//launches the travelling sales best tour finder and 
//outputs results
unsigned long findBestTour() {
  
	//get distance and city name data
	vector<string> pickedCityNames;
	int* distances = new int[_tourSize * _tourSize];
	getDistances(distances, pickedCityNames);
  unsigned int* bestTour = new unsigned int[_arrayTourSize];
  unsigned long iterations = startFindingTour(distances, bestTour);
	
	if (isRootProcess()) {
		cout << "Best Tour:" << endl;
		for (unsigned int city = POS_TOUR_START + 1; city < _arrayTourSize; city++) {
			unsigned int dist = getDistance(distances, bestTour[city - 1], bestTour[city]);
			cout << " " << (city - POS_TOUR_START) << ". ";
			cout << pickedCityNames[bestTour[city - 1]];
			cout << " to ";
			cout << pickedCityNames[bestTour[city]];
			cout << " is ";
			cout << dist << " miles" << endl;
		}
		cout << endl;
		cout << "Total Distance: " << bestTour[POS_TOUR_TOTAL_DIST] << " miles" << endl;
		cout << endl;
		cout << "Iterations: " << iterations << endl;
		
		cout << endl;
		cout << "To View Route:" << endl;
		cout << "- Copy the following values, then open the URL below," << endl;
		cout << "- click the Copy/Paste option, paste the values, " << endl;
		cout << "- then click Preview Route to view best tour." << endl;
		cout << "- MapQuest may require you to indicate some values are cities." << endl;
		cout << "- https://www.mapquest.com/routeplanner" << endl;
		cout << endl;
		for (unsigned int city = POS_TOUR_START; city < _arrayTourSize; city++) {
			cout << " " << pickedCityNames[bestTour[city]] << endl;
		}
	}
  
  //clean up dynamic arrays
  delete distances;
  delete bestTour;
  
  return iterations;  
}

//--------------------------------------------------------------
//lists all of the configurable values
void showCommandLineArgs() {
  cout << endl;
  cout << "APPLICATION ARGS" << endl;
  cout << "-    <TOUR_SIZE>: " << _tourSize << endl;
  cout << "-    [HOME_CITY]: " << _requestedHomeTown << endl;
  cout << "-  [RANDOM_SEED]: " << _randomSeed << endl;
  cout << endl;
}

//--------------------------------------------------------------
//Says hello from each processor in the mpi world
void showMpi() {
  printf("** Hello from processor %s, rank %d"
    " out of %d processors\n",
    _mpi_processor_name, _mpi_world_rank, _mpi_world_size);
}


//--------------------------------------------------------------
//adds a couple of blank lines
void showWhiteSpace() {
	if (isRootProcess()) {
		cout << endl;
		cout << endl;    
	}
}

//--------------------------------------------------------------
//initializes MPI and sets globals about environment other 
//methods may use
void initMpi(bool sayHello) {
  //init MPI
  MPI_Init(NULL, NULL);
  
  //get num processes  
	MPI_Comm_size(MPI_COMM_WORLD, &_mpi_world_size);

	//get current process rank
	MPI_Comm_rank(MPI_COMM_WORLD, &_mpi_world_rank);

	//get name of processor
	int name_len;
	MPI_Get_processor_name(_mpi_processor_name, &name_len);
  
  if (isRootProcess()) {
		showWhiteSpace();
		cout << endl << "PROCESS ARGS" << endl;
    cout << "- MPI World Size: " << _mpi_world_size << endl;
  }
  
  if (sayHello) showMpi();
}

//--------------------------------------------------------------
//closes out the MPI operation and cleans up
void finalizeMpi() {
  //close out MPI
	showWhiteSpace();
  MPI_Finalize();
	
}

//--------------------------------------------------------------
//says hello from the current omp thread
void ompSayHello(void) {
  int my_rank = 0;
  int thread_count = 1;
  
  #ifdef _OPENMP
  my_rank = omp_get_thread_num();
  thread_count = omp_get_num_threads();  
  #endif 

  printf("MPI RANK %d says 'Hello from thread %d of %d'\n", _mpi_world_rank, my_rank, thread_count);
} 

//--------------------------------------------------------------
//sets omp rank and omp thread count
void initializeOmp(bool showHello) {

  #ifdef _OPENMP
  _ompRank = omp_get_thread_num();  
  #endif
	if (isRootProcess()) {
		cout << "-       OMP Rank: " << _ompRank << endl;
		cout << "-    OMP Threads: " << _ompThreads << endl;
	}
  
  if (showHello) {
    # pragma omp parallel num_threads(_ompThreads) 
    ompSayHello();
  }
}

//--------------------------------------------------------------
//if an error occurred, tell all processes to shutdown
bool broadcastErrorStatus(bool error) {
  int errorWhenNotZero = error ? 1 : 0;
  MPI_Bcast(&errorWhenNotZero, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return errorWhenNotZero > 0;
}

//--------------------------------------------------------------
//records and displays timings on root process and starts touring
bool startTouring() {
	std::chrono::high_resolution_clock::time_point t_start, t_end;
	bool error = false;
	
	try {
			//setup omp before starting histogram computation (but after MPI so we know rank)
			initializeOmp(false);
		
			if (isRootProcess()) {
				showCommandLineArgs();
				t_start = std::chrono::high_resolution_clock::now();
			}
			
			findBestTour();
			
			if (isRootProcess()) {
				t_end = std::chrono::high_resolution_clock::now();
				cout << endl;
				cout <<  "ELAPSED SEC...: " << (std::chrono::duration<double, std::milli>(t_end-t_start).count() / 1000.0) << " sec" << endl;
				cout <<  "ELAPSED MS....: " << (std::chrono::duration<double, std::milli>(t_end-t_start).count()) << " ms" << endl;
				cout << endl;
			}
	} catch (const std::exception& e) {
		cerr << "ERROR: " << e.what() << endl;
		error = true;
	}
	return broadcastErrorStatus(error);
}

//--------------------------------------------------------------
//main
int main(int argc, char* argv[])
{
  bool error = false;
	
	//setup MPI 
  initMpi(false);
  
  //verify arg count
  if (argc - 1 < ARGS_MIN || argc - 1 > ARGS_MAX) {
    cerr << "ERROR: This application requires between " << ARGS_MIN << " and " << ARGS_MAX << " arguments. You supplied " << (argc - 1) << "." << endl;
    cerr << "Usage: mpirun -np <PROCESSES> ./" << argv[ARGS_FILENAME] << " <THREADS> <TOUR_SIZE> [HOME_CITY, default=9] [RANDOM_SEED, default=1]" << endl;
    error = true;
  }
	
	//get requested thread count
  #ifdef _OPENMP
  try {
    if (!error) {
      stringstream convertArg(argv[ARGS_THREADS]);
      if (!(convertArg >> _ompThreads)) {
        if (isRootProcess()) cerr << "THREADS must be a valid int: " << argv[ARGS_THREADS] << endl;
        error = true;
      }
    }
  } catch (const std::exception& e) {
      error = true;
      if (isRootProcess()) cerr << "ERROR: " << e.what() << endl;
  }
  #else
    _ompThreads = 1;
  #endif
  
  //get tour size
  try {
    if (!error) {
      stringstream convertArg(argv[ARGS_TOURSIZE]);
      if (!(convertArg >> _tourSize)) {
        cerr << "ERROR: <TOUR_SIZE> must be a valid int: " << argv[ARGS_TOURSIZE] << endl;
        error = true;
      } else {
				_arrayTourSize = POS_TOUR_START + _tourSize + 1;
			}
    }
  } catch (const std::exception& e) {
      error = true;
      cerr << "ERROR: " << e.what() << endl;
  }
  
  //get optional home city index
  try {
    if (!error && argc > ARGS_HOMETOWN) {
      stringstream convertArg(argv[ARGS_HOMETOWN]);
      if (!(convertArg >> _requestedHomeTown)) {
        cerr << "ERROR: [HOME_CITY] must be a valid int: " << argv[ARGS_HOMETOWN] << endl;
        error = true;
      } else if (_requestedHomeTown < 0 || _requestedHomeTown > 255) {
        cerr << "ERROR: [HOME_CITY] must be between 0 and 255, inclusively. See cities.txt to pick an index city: " << argv[ARGS_HOMETOWN] << endl;
        error = true;
      }
    }
  } catch (const std::exception& e) {
      error = true;
      cerr << "ERROR: " << e.what() << endl;
  }
  
  //get optional random seed value
  try {
    if (!error && argc > ARGS_RANDOMSEED) {
      stringstream convertArg(argv[ARGS_RANDOMSEED]);
      if (!(convertArg >> _randomSeed)) {
        cerr << "ERROR: [RANDOM_SEED] must be a valid long: " << argv[ARGS_RANDOMSEED] << endl;
        error = true;
      } else if (_randomSeed < 1) {
        error = true;
        cerr << "ERROR: [RANDOM_SEED] must be 1 or greater: " << argv[ARGS_RANDOMSEED] << endl;
      }
    }
  } catch (const std::exception& e) {
      error = true;
      cerr << "ERROR: " << e.what() << endl;
  }
  
	//broadcast error status
  error = broadcastErrorStatus(error);
	
  if (!error) {
		error = startTouring();
  } 
		
  if (error) {
		cerr << "STOPPED. BEST TOUR NOT COMPUTED." << endl;
	}

	//shutdown MPI
  finalizeMpi();
	
  return error ? 1 : 0;
}