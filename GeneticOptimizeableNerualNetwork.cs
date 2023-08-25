using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
//inherits the "NeuralNetwork" class and implements the "IGeneticOptimizable" interface
//defines a neural network class that can be optimised by genetic algorithms and provides serialisation and deserialisation methods and replication methods for saving, loading and replicating network instances in genetic algorithms
namespace NN.AI {
	public class GeneticOptimizeableNerualNetwork : NeuralNetwork, IGeneticOptimizeable {
    //Acquisition of fitness, which can be used to evaluate the performance of the network.
    public double fitness { get; set; }
    // can optimise values, get weights and differences and update
    public List<double> optimizeableValues {         
        get { return Serialize(); }
        set { Deserialize(value);}
    }

    // indicates the bias value in the neural network
    public List<double> B { get; set; }
    
    
    public GeneticOptimizeableNerualNetwork(params int[] topology): base(topology){}// Genetic Optimisation Neural Networks
    //params, variable parameters, can be passed into int one-dimensional arrays
    public IGeneticOptimizeable Reproduce() {
        //create new instance,layers" is information about the number of layers of the neural network, which is used to initialise the new instance.
        var clone = new GeneticOptimizeableNerualNetwork(layers);
        //Deserialisation optimises the values, ensuring that the new instance has the same weights and bias values as the current instance.
        clone.Deserialize(Serialize());
        for (var i = 0; i < layers.Length; i++) {
            for (var j = 0; j < layers[i]; j++) {
                //Copy the neuron output to ensure that the new instance has the same neuron output as the current instance.
                clone.neuronsOutputs[i][j] = neuronsOutputs[i][j];
            }
        }
    
        for (var i = 0; i < layers.Length - 1; i++) {
            for (var j = 0; j < layers[i + 1]; j++) {
                //Copy the activation function to ensure that the new instance has the same activation function as the current instance.
                clone.activateFunctions[i][j] = activateFunctions[i][j];
            }
        }
        // Copy the fitness to ensure that the new instance has the same fitness value as the current instance.
        clone.fitness = fitness;
        return clone;
    }
    /* 
    Methods "Save()" and "Load()": these two methods are used to serialise and deserialise the 
    optimisable values of the neural network for saving and loading in the genetic algorithm. 
    The "Save()" method uses the "BinaryFormatter" class to serialise the optimisable values into 
    a byte array, and the "Load()" method uses the "BinaryFormatter" class to deserialise the byte 
    array into optimisable values and update the corresponding properties in the network.
    */
    public byte[] Save() {
        var bf = new BinaryFormatter();
        using (var ms = new MemoryStream()) {
            bf.Serialize(ms, optimizeableValues);
            return ms.ToArray();
        }
    }

    public void Load(byte[] data) {
        var bf = new BinaryFormatter();
        using (var ms = new MemoryStream(data)) {
            var values = (List<double>)bf.Deserialize(ms);
            Deserialize(values);
        }
    }
}
}

