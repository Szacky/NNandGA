using System;
using System.Collections.Generic;
using System.Linq;
using Random = System.Random;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Newtonsoft.Json;
/*
    This code implements a genetic optimisation algorithm.
     The algorithm works by generating populations with a certain degree of randomness
     and in each generation performs selection, reproduction, mutation
     crossover operations to search for a solution to the problem.
*/

namespace NN.AI {
    /*
    Enumeration SelectionMethod: Enumeration type to specify a selection method.
    Four selection methods are defined in this enumeration:
    Competitive selection, natural selection, Roulette and Random_ForFun selection. There is also a Tournament selection.
    */
    public enum SelectionMethod {
        Competitive,
        Natural,
        Random_ForFun,
        Roulette,
        Tournament
    }

    /*
    GeneticOptimisation: the genetic optimisation class.
    Contains a set of parameters and methods for implementing genetic optimisation algorithms.
    */
    public class GeneticOptimisation {

        public SelectionMethod selectionMethod;//Select method
        public int populationCount; //Population size
        public List<IGeneticOptimizeable> population; //quantity, storing all individuals of the population, IGeneticOptimizeable is an interface that specifies the behaviour of the individuals to be optimised in the genetic algorithm.
        public double mutateProbability; //Probability of variation

        private int populationSize = 100; // Defining population size
        private double survivalRate = 0.5; // Defining survivorship
        private double sigma = 0.1;
        //private int optimizeableCount = 10; //Gaussian distribution initialisation
        private static readonly Random rndC = new Random();//Same as Randomize method 2: Xavier initialization strategy

        //random method 1
        //public static Random rnd = new Random(Guid.NewGuid().GetHashCode()); //It may generate the same seed in the short time it takes for multiple instances to be created, as the GetHashCode() method does not have enough randomness to ensure that different GUIDs generate different hash codes.
       
       //random method 2
       // For better randomness
        public static Random rnd = CreateRandom();
        private static Random CreateRandom(){
            byte[] seed = new byte[4];
            using (RandomNumberGenerator rng = RandomNumberGenerator.Create()){
                rng.GetBytes(seed);
            }
            return new Random(BitConverter.ToInt32(seed, 0));
        }



        /**
        Constructor named "GeneticOptimisation" with three parameters:
        1. populationCount population size
        2. mutateProbability The probability of variation
        3. selectionMethod selection method
        Inside the constructor, it first assigns the incoming selectionMethod, populationCount and mutateProbability to the corresponding properties of the class.
        It then creates a List object of size populationCount to represent the population. This List stores the individuals that implement the "IGeneticOptimizeable" interface.
        */
        public GeneticOptimisation(int populationCount, double mutateProbability, SelectionMethod selectionMethod) {//种群规模，变异概率，选择方法
            this.selectionMethod = selectionMethod;
            this.populationCount = populationCount;
            this.mutateProbability = mutateProbability;
            population = new List<IGeneticOptimizeable>(populationCount);
        }


        // Define tournament methods
        private int GetTournamentGen(IList<double> probs){
            int tournamentSize = 2; //Set tournament size to 2
            int winnerIndex = rnd.Next(probs.Count);

            for (int i = 1; i < tournamentSize; i++){
                int challengerIndex = rnd.Next(probs.Count);
                if (probs[challengerIndex] > probs[winnerIndex])
                {
                    winnerIndex = challengerIndex;
                }
            }

            return winnerIndex;
        } 


        /**
        GetRandomGen: selection function, roulette selection operator
        Selects an individual based on the selection method and the probability of fitness of each individual.
        That is to say implements the method used in genetic optimization algorithms to select genes for the next generation.
        The argument probs is a list containing the probabilities of each gene. The GetRandomGen method calculates and returns a subscript for the selected gene depending on the selection method selectionMethod.
        */
        private int GetRandomGen(IList<double> probs) { 
            
            /**
            The GetRandomGen method returns the index value of a number of individuals when "competitive selection" is required in a genetic algorithm.
            The method takes a parameter of type IList<double>, which contains the probability of some individuals.
            The probability here is the probability that the individuals will be selected.
            */
            int selectedIndex = -1;
            double highestProb = 0;
            // These three selection methods implement different selection strategies, and different methods can be chosen for genetic optimisation depending on the actual requirements.
            switch (selectionMethod) {
                //Competitive: a randomly selected gene with a relative bias towards the middle based on a Gaussian distribution.
                case SelectionMethod.Competitive:
                 // algorithm will select individuals by generating a random value in a Gaussian distribution. Depending on the properties of the Gaussian distribution, there is a half probability that the random number will be below the mean and a half probability that it will be above the mean.
                    var dis = 1.0 - Math.Abs(ISMath.GaussianRandomDistributed - 0.5d);
                    return (int)Math.Round(dis * (probs.Count-1));
                //Natural: accumulates the probabilities of one gene in order, based on the probability of each gene, and generates a random number in [0,1), selecting the gene corresponding to the interval in which the number is located.
                case SelectionMethod.Natural:
                    double top = 0;
                    var randomValue = rnd.NextDouble();
                    for (var i = 0; i < probs.Count; i++) {
                        var bot = top;
                        top += probs[i];
                        if (randomValue >= bot && randomValue <= top) return i;
                    }
                    return 0;
                //Random_For_Fun: random selection of a gene.
                case SelectionMethod.Random_ForFun:
                    return rnd.Next(probs.Count);
                //Tournament: A tournament in which a number of individuals are selected at random and the best ones are chosen.
                //1. The index of each individual in the Candidates list is taken and its fitness value is calculated.
                //2. Then compare it with bestProb, and if it is greater than bestProb, set it to the index and fitness value of the current best individual.
                //The index of the current best individual is returned.
                case SelectionMethod.Tournament:
                    return GetTournamentGen(probs);
                
                case SelectionMethod.Roulette:
                    double r = rnd.NextDouble();
                    double sum = 0;
                    for (int i = 0; i < probs.Count; i++){
                        sum += probs[i];
                        if (r < sum){ 
                            selectedIndex = i;
                            return selectedIndex;
                        }
                    }
                    return selectedIndex;
                // If the value of selectionMethod does not match any of the branches in the enumeration, an ArgumentOutOfRangeException is thrown, which indicates that the argument provided is out of the expected range, in this case indicating that the selection method argument passed in is not the expected enumeration value.
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }


        /**
        GetPairs: the breeding function that
        Selects a number of individuals from the current population for crossover and mutation to generate new individuals.
        In the GetPairs() method, the process of generating a new population is to first select a pair of individuals at random based on fitness probability, and then generate two new individuals by reproducing these two individuals and adding them to the result list until the number of elements in the list reaches 4 times the population size, indicating that a sufficient number of new populations have been generated.
        As the generation of new populations is dependent on the original individuals, at the end of each round we need to replace the old individuals with the newly generated ones for the next round of reproduction. This is done in the Evolve() method

        1. Create an empty list of size 4*populationCount to store all the children.
        2. Calculate the sum of the fitness of all individuals in the population.
        3. Calculate the probability of each individual being selected as a parent (i.e. the sum of fitness/adaptation) and store the probabilities in the lifeProb array.
        4. Sort the lifeProb array in descending order.
        5. Initialise a counter variable to keep track of the number of children added to the result list.
        6. In a while loop, continuously select two different individuals from the population and breed them. If the two individuals selected are the same, the loop is skipped.
        7. Add the reproduced children to the result list and add 1 to the counter.
        8. Repeat steps 6 and 7 until the number of children in the result list reaches the specified number.
        9. Return the result list.
        
        private List<IGeneticOptimizeable> GetPairs() { // Get couples of the same logarithm of the population size to use to reproduce new populations
            var res = new List<IGeneticOptimizeable>(4 * populationCount).
            //var sum = population.Sum(item => item.fitness).
            //var lifeProb = population.Select(item => item.fitness / sum).ToArray().
            //var bestLifeIndex = lifeProb.Select((prob, index) => (prob, index)).OrderByDescending(x => x.prob).First().index.
            var sum = population.Sum(item => item.fitness); // get the sum of all adaptations for this population
            var lifeProb = new double[populationCount]; // to store the probability of fitness for each individual organism;
            for (var i = 0; i < lifeProb.Length; i++) lifeProb[i] = population[i].fitness / sum.
            lifeProb.OrderByDescending(x => x).
            var counter = 0.
            while (counter < populationCount) { // loop keeps pairing up Each pair is basically different
                var leftGen = GetRandomGen(lifeProb); // get the subscript of the individual with the best adaptation;
                var rightGen = GetRandomGen(lifeProb); // get the other one The two are not the same
                if (leftGen == rightGen) continue.
                res.Add(population[leftGen].Reproduce()).
                res.Add(population[rightGen].Reproduce()).
                counter += 1.
            }
            //while (counter < populationCount) {
                //var leftGen = GetRandomGen(lifeProb).
                //var rightGen = GetRandomGen(lifeProb).
                //if (leftGen ! = rightGen) {
                    //res.Add(population[leftGen].Reproduce()).
                    //res.Add(population[rightGen].Reproduce()).
                    //counter += 1.
                //}
            ///}
            return res.
        }
        */


        // Use multiple threads to speed up
        private List<IGeneticOptimizeable> GetPairs(){
            var res = new List<IGeneticOptimizeable>(4 * populationCount);
            var sum = population.Sum(item => item.fitness);
            var lifeProb = new double[populationCount];
            for (var i = 0; i < lifeProb.Length; i++) lifeProb[i] = population[i].fitness / sum;
            lifeProb.OrderByDescending(x => x);

            var numThreads = Environment.ProcessorCount;
            var pairsPerThread = populationCount / numThreads;
            var remainingPairs = populationCount % numThreads;

            var threads = new Thread[numThreads];
            var pairsCompleted = 0;
            for (var i = 0; i < numThreads; i++){
                var numPairs = pairsPerThread;
                if (remainingPairs > 0){
                    numPairs++;
                    remainingPairs--;
                }
                var start = pairsCompleted;
                var end = start + numPairs;
                pairsCompleted = end;

                threads[i] = new Thread(() =>{
                    for (var j = start; j < end; j++){
                        var leftGen = GetRandomGen(lifeProb);
                        var rightGen = GetRandomGen(lifeProb);
                        while (leftGen == rightGen){
                            rightGen = GetRandomGen(lifeProb);
                        }
                        res.Add(population[leftGen].Reproduce());
                        res.Add(population[rightGen].Reproduce());
                    }
                });
                threads[i].Start();
            }

            foreach (var thread in threads){
                thread.Join();
            }

            return res;
        }

        /**
        ReproduceAll: method 1 generates a new population.
        Contains two processes: crossover and mutation.
        The effect is to reproduce a new generation of populations using the logarithm of the incoming population.
        The current population is first cleared using the Clear() method, then each pair of parents is crossed over using the Crossover() method to produce a new child.
        Next, the resulting children are mutated using the Mutate() method and finally added to the new population.
        When the cycle is complete, the new population contains all the new individuals that have been reproduced by the genetic manipulation.
        
        private void ReproduceAll(IReadOnlyList<IGeneticOptimizeable> pairs) { // propagate the new generation of population genes;
            // Clear the current population, as we are generating a new batch of individuals.
            population.Clear().
            // use a for loop to iterate over each pair, crossover and mutate them, and finally add them to the next generation population. Since each couple produces two offspring, in each loop we first cross over the couples, then mutate one of the offspring, and finally add them to the next generation population.
            for (var i = 0; i < pairs.Count; i += 2) {
                Crossover(pairs[i], pairs[i + 1]).
                Mutate(pairs[i]).
                population.Add(pairs[i]).
            }
        }
        */


        /**
        ReproduceAll: method 2 implements parallelised processing
        The reproduction process for each couple is placed in a lambda expression, and then each couple is traversed via the Parallel.ForEach method, while they are crossed and mutated, and added to the next generation population.
        */
        private void ReproduceAll(IReadOnlyList<IGeneticOptimizeable> pairs) {
            // Empty the current group, as we are generating a new batch of individuals
            population.Clear();

            // Use parallelization to traverse each couple, crossover and mutate them, and finally add them to the next generation population.
            Parallel.ForEach(pairs.Where(p => p is INextParentGeneticOptimizeable), pair => {
                var nextParentPair = (pair as INextParentGeneticOptimizeable).NextParent;
                Crossover(pair, nextParentPair);
                Mutate(pair);
                lock(population) {
                    population.Add(pair);
                }
            });
        }
        


        /**CreateRandomPopulation method 1
        is used to create a random population, i.e. to generate a random batch of genomically synthesised individuals at the beginning. (In genetic algorithms, the quality of the initial population has a significant impact on the final outcome. Differences in the quality of the randomly generated population can lead to different results in subsequent evolution. Therefore, some high-quality random generation algorithms, such as genetic algorithm-specific initialisation algorithms, are very important.)
        The randGen function is to generate random individuals. This function has the argument type Func<IGeneticOptimizeable>, indicating a parameterless function with a return value of type IGeneticOptimizeable.
        The CreateRandomPopulation function, uses a loop to generate a specified number of individuals, adding each newly generated individual to the population list, and eventually returning a random population.
        
        public void CreateRandomPopulation(Func<IGeneticOptimizeable> randGen) {//Create a random population
            // First clear the current population using the population.Clear() method.
            population.Clear().
            //then use a for loop to loop through populationCount a number of times, i.e. to create a population of size populationCount.
            // In the loop body, a random individual is generated using the randGen() method and added to the population (using the population.Add(randGen()) method). Eventually, the population is generated and saved in the population variable, which can be used as a starting point for subsequent evolution of the genetic algorithm.
            for (var i = 0; i < populationCount; i++) {
                population.Add(randGen()).
            }
        }
        */

        /**CreateRandomPopulation method 2
        // Optimization Multiple individuals can be created in parallel using the Parallel.For or Parallel.ForEach methods.
        // The ConcurrentBag type acts as a concurrency container, ensuring thread safety in a multi-threaded environment. During parallel execution, each thread generates an individual independently, which is eventually merged into the population via the concurrency container.
        public void CreateRandomPopulation(Func<IGeneticOptimizeable> randGen) {
            population = new ConcurrentBag<IGeneticOptimizeable>().
            Parallel.For(0, populationCount, i => {
                population.Add(randGen()).
            }).
        }
        */
        
        //CreateRandomPopulation method 3
        //Using the LINQ library, the Enumerable.Range and Select methods are used to generate a specified number of random genes
        //Range to generate a sequence of integers from 0 to populationSize-1, then use the Select method to map each integer to a randomly generated gene object, and convert the result to a List.
        public void CreateRandomPopulation(int populationSize, Func<IGeneticOptimizeable> randGen){
            population = Enumerable.Range(0, populationSize).Select(_ => randGen()).ToList();
        }


        /**
        Evolve Method 1: All-Pair Reproduction
        Evolve: the core method of genetic algorithms, all-pair propagation
        Contains the operations of selection, reproduction, mutation and crossover to achieve a generational evolutionary process.
        1. assigning a new fitness to each individual in the population.
        2. selects members of a new generation of populations based on the fitness of each individual in the current population
        3. Use crossover and mutation operations to generate individuals for the new generation of populations.
        4. Replace the original population with the new generation population.
        *newFitnesses is a floating point array of fitnesses for each individual in the new generation population
        *population is the current population, which contains all the individuals produced in the previous evolutionary generation.
        The *ReproduceAll function selects the members of the new generation population based on fitness, and then applies crossover and mutation operations to generate the individuals of the new generation population.
        *double[] newFitnesses is used to store the new fitness of each individual after evaluation.
        
        public void Evolve(double[] newFitnesses) { //evolve
            //for loop iterates over each individual, setting their fitness property to the value at the corresponding index in newFitnesses, thus updating the fitness of those individuals.
            for (var i = 0; i < newFitnesses.Length; i++) population[i].fitness = newFitnesses[i].
            ReproduceAll(GetPairs()).
        }
        */
        
        //Evolve method 2: elimination method
        public void Evolve(double[] newFitnesses){
            // Updating individual adaptability
            for (var i = 0; i < newFitnesses.Length; i++) population[i].fitness = newFitnesses[i];
            // Sorting individuals by suitability
            population = population.OrderByDescending(p => p.fitness).ToList();
            // Elimination of less adapted individuals
            int numSurvivors = (int)(populationSize * survivalRate);// Dealing with under-represented populations
            if (population.Count < numSurvivors){
            }
            population = population.OrderByDescending(p => p.fitness).Take(numSurvivors).ToList();
        }


        /**
        Mutate method 1: basic random mutation strategy The mutation probability of a gene is fixed and each mutation is carried out according to a random offset.
        Mutate: a mutation function that
        A random perturbation of certain characteristics of an individual is used to increase the diversity of a population.
        A method called Mutate is defined which implements the mutation operation in the genetic algorithm.
        Objects implementing the IGeneticOptimizeable interface are used as parameters and some of the genes in their optimisable values are randomly modified to make the search space of the genetic algorithm richer.
        1. First, the optimisable value of the incoming object is obtained and assigned to the w variable.
        2. Next, the method uses a random number generator, rnd, to generate a random number in the range [0, 1).
        3. It is then compared with a preset mutateProbability.
        4. If the random number is smaller than the mutateProbability, the method randomly selects a gene in w and adds it to a random number in the range [-1, 1].
        5. Finally, the method reassigns the mutated optimizable values to the optimizeableValues property of the incoming object.
        
        public void Mutate(IGeneticOptimizeable gen) { //mutate
            var w = gen.optimizableValues.
            for (var i = 0; i < w.Count; i++)
                //rnd.NextDouble() is a random number generator method that returns a pseudo-random number of type double whose value is in the interval [0, 1).
                if (rnd.NextDouble() < mutateProbability) w[i] += rnd.NextDouble() * 2 - 1.
            gen.optimizeableValues = w.

            //private double dertFit.
            //for (int i = elite.Count; i < children.Count(); i++)
            //{
                //if (UnityEngine.Random.Range(0f, 1f) < muta_ratio)
                //{
                    //int index = UnityEngine.Random.Range(0, children[0].Length).
                    //children[i][index] = UnityEngine.Random.Range(-1f, 1f).
                //}
            //}
        }
        */

        //Mutate2:polynomial variation
        /**
        You can perturb an individual's genes by changing the value of a gene locus. eta is a parameter that controls the degree of polynomial variation, usually taking values between 20 and 100. lowerBound and upperBound are the lower and upper limits of a gene, respectively.
        
        public void Mutate(IGeneticOptimizeable gen) {
            var w = gen.optimizeableValues.
            for (var i = 0; i < w.Count; i++) {
                if (rnd.NextDouble() < mutateProbability) {
                    // polynomial variation
                    double u = rnd.NextDouble().
                    double delta.
                    if (u < 0.5) {
                        delta = Math.Pow(2.0 * u, 1.0 / (eta + 1.0)) - 1.0.
                    } else {
                        delta = 1.0 - Math.Pow(2.0 * (1.0 - u), 1.0 / (eta + 1.0)).
                    }
                    w[i] += delta * (upperBound - lowerBound).
                    w[i] = Math.Max(w[i], lowerBound).
                    w[i] = Math.Min(w[i], upperBound).
                }       
            }
            gen.optimizeableValues = w.
        }
        */

        //Mutate method 3: Gaussian variation
        public void Mutate(IGeneticOptimizeable gen) {
            var w = gen.optimizeableValues;
            for (var i = 0; i < w.Count; i++) {
                if (rnd.NextDouble() < mutateProbability) {
                    //Gaussian variation
                    double u = rnd.NextDouble();
                    double v = rnd.NextDouble();
                    double step = Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v) * sigma;
                    w[i] += step;
                }   
            }
            gen.optimizeableValues = w;
        }



        /**
        Crossover method 1: The single point crossover method selects a random crossover point at the time of crossover and inherits the gene before the crossover point from the other individual.
        Crossover: The crossover function that
        Used to swap certain features of two individuals to produce a new individual.
        A method called Crossover is defined to implement the crossover operation in genetic algorithms.
        In genetic algorithms, the crossover operation is used to generate new individuals by swapping and combining certain characteristics of two individuals (e.g. weights) to generate new individuals in order to increase the diversity of the population.
        The method accepts two individuals mom and dad that implement the IGeneticOptimizeable interface and swaps n features of the two individuals by randomly selecting a position n to generate two new individuals.
        1. mom.optimizeableValues and dad.optimizeableValues represent the feature lists of mom and dad respectively.
        The rnd.Next(momW.Count) method is used to randomly select a position n and swap the first n features from dad to mom and the next momW.Count - n features from mom to dad.
        Finally, the method saves the modified feature values of the mom and dad individuals back into mom.optimizeableValues and dad.optimizeableValues for subsequent genetic algorithm operations...
        
        public void Crossover(IGeneticOptimizeable mom, IGeneticOptimizeable dad) { //crossover  
            var momW = mom.optimizableValues.
            var dadW = dad.optimizeableValues.
            var n = rnd.Next(momW.Count).
            //Use a loop to iterate through all the elements in the momW list, copying the first n elements from the dadW list to the momW list and the next momW.Count - n elements from the momW list to the dadW list, based on a randomly selected intersection n.
            // When the loop variable i is less than the intersection n, copying the value of dadW[i] into momW[i] means crossing the first n elements in dad into mom; otherwise, copying the value of momW[i] into dadW[i] means crossing the next momW.Count - n elements in mom into dad. In this way, a random exchange of certain characteristics of the two individuals is achieved, resulting in a new individual.
            for (var i = 0; i < momW.Count; i++) {
                if (i < n)
                    momW[i] = dadW[i].
                else
                    dadW[i] = momW[i].
            }
            mom.optimizeableValues = momW.
            dad.optimizeableValues = dadW.

            //while (true)
            //{
                //Genome dad = GetParent(eliteGenomeList).
                //Genome mum = GetParent(eliteGenomeList).
                ///double[] baby1 = null.
                //double[] baby2 = null.
                //CrossoverAtSplitPoint(dad.splitPoints, dad.weights, mum.weights, out baby1, out baby2).
                //children.Add(baby1).
                //children.Add(baby2).
                //int n = populationSize;// (int)(population * 3.6f / 4); whether to discard the worst part of the individuals and re-add to the population at random
                //if (children.Count >= n)
                //{
                    //while (true)
                    //{
                        //if (children.Count > n)
                        //{
                            //children.RemoveAt(children.Count - 1).
                        ///}
                        ///else
                        //{
                            //break.
                        //}
                    ///}
                    //break.
               // }
            //}
        }
        */

        
        /**
        Crossover method 2: Multi-point crossover strategy
        This uses a strategy of randomly generating multiple crossovers and sequencing the positions of the crossovers.
        At crossover, the genes are copied starting from one parent, then switch to another parent when a crossover is encountered and continue copying until all genes have been copied.
        */
        public void Crossover(IGeneticOptimizeable mom, IGeneticOptimizeable dad)
        {
            var momW = mom.optimizeableValues;
            var dadW = dad.optimizeableValues;
            var numPoints = rnd.Next(1, momW.Count); // Randomly generated number of intersections, at least 1
            var points = new int[numPoints];

            for (var i = 0; i < numPoints; i++){
                points[i] = rnd.Next(0, momW.Count); // Randomly generated intersection positions
            }
            Array.Sort(points); // Sort the intersection positions

            var currentParent = momW; // Current parent
            for (var i = 0; i < momW.Count; i++){
                if (points.Contains(i)) {
                    currentParent = currentParent == momW ? dadW : momW;
                }
                momW[i] = currentParent[i]; // Copy genes from current parent
            }

            mom.optimizeableValues = momW;
            dad.optimizeableValues = dadW;
        }


        /**
        Initialisation method 1: uniform random initialisation
        The initialization operation is used to generate a random set of primitive populations.
        1. The RandomizePopulation method uses a loop to iterate through all the individuals in the population list, calling the Randomize method to randomly initialise each individual with its optimisable values (i.e. the weights and biases of the neural network).
        2. min and max denote the lower and upper bounds, respectively, of the range of random numbers generated. In a genetic algorithm, this operation is called at the beginning of the algorithm to provide a random and diverse set of initial populations for the subsequent iterations.
        When using GeneticOptimization, the RandomizePopulation method is called at training initialisation and is used to generate a random set of initial populations where the optimisable values (i.e. weights and biases of the neural network) for each individual are set to a random number in the range [-1.0, 1.0].
        */
        public void RandomizePopulation(double min = -1.0, double max = 1.0) { //random population Called at training initialization
            for (var i = 0; i < populationCount; i++)
                Randomize(population[i], min, max);
        }
        
        /**
        Randomize method 2: Xavier initialization strategy
        Here, we first calculate the standard deviation, then randomly initialize the weights using a normal distribution with mean 0 and standard deviation stddev, and finally assign the randomly generated array of weight values w to the optimizeableValues property of the gen object.
        */
        private static double RandomGaussian(double mean, double stddev) {
            double u1 = 1.0 - rndC.NextDouble();
            double u2 = 1.0 - rndC.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            double randNormal = mean + stddev * randStdNormal;
            return randNormal;
        }

        private static void Randomize(IGeneticOptimizeable gen, double min, double max) {
            var w = gen.optimizeableValues;
            var nIn = w.Count;
            // The number of nodes output here is assumed to be 1, which should actually be adjusted according to the model structure
            var nOut = 1;  
            // Calculate the standard deviation
            var stddev = Math.Sqrt(2.0 / (nIn + nOut));
             //use a normal distribution with a mean of 0 and a standard deviation of stddev to randomly initialise the weights
            for (var i = 0; i < nIn; i++) {
                w[i] = RandomGaussian(0.0, stddev);
            }
            //assign the randomly generated array of weight values w to the optimizeableValues property of the gen object
            gen.optimizeableValues = w;
        }
    }


    /**
    Interface definition for genetic optimization algorithms.
    1. optimizableValues: represents a set of genes, understood as a collection of weights and offsets, and is one of the attributes of an individual in a genetic algorithm.
    2. fitness: indicates the fitness of the individual, used to evaluate the performance of the individual in problem solving, and is one of the bases for selecting individuals in the genetic algorithm;
    3. reproduce(): indicates the reproduction method of the individual, i.e. how to generate new offspring based on the crossover and mutation operations in the genetic algorithm.
    */
    public interface IGeneticOptimizeable {// Optimized genetic interface Used to store all individuals that inherit from this interface;
        List<double> optimizeableValues { get; set; } // a set of genetic genes (here also this set of weights and offsets);
        double fitness { get; set; } // fitness , used to decide who to cull;
        IGeneticOptimizeable Reproduce(); // the method of reproduction.
    }

    public interface INextParentGeneticOptimizeable : IGeneticOptimizeable {
        IGeneticOptimizeable NextParent { get; set; } 
    }
    
    
}