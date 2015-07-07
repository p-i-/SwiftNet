//
//  main.swift
//  BackProp
//
//  Created by π on 02/07/2015.
//  Copyright © 2015 π. All rights reserved.
//

import Foundation

// PICS http://home.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html

func random_01() -> Double {
    return Double(arc4random()) / Double(UInt32.max)
}

extension SequenceType
{
    func eachIndexValue( action: (Int,Generator.Element)->Void ) {
        for (idx,element) in self.enumerate() {
            action(idx,element)
        }
    }

    // allow e.g.
    //     var Yarr = Xarr.mapi { Y($0, $1) }
    // with $0 = item, $1 = index
    func mapIndexValue( action: (Int,Generator.Element)->Any )  ->  [Any] {
        var arr = [Any] ()
        for (idx,element) in self.enumerate() {
            arr.append( action(idx,element) )
        }
        return arr
    }
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

class Neuron {
    unowned let net   : Net
    unowned let layer : Layer
            let index : Int

    var weights = [Double] ()
    var bias    = random_01()

    var sum = 0.0
    var output = 0.0
    var delta = 0.0

    init( net:Net, layer:Layer, index:Int )
    {
        (self.net, self.layer, self.index) = (net, layer, index)

        guard let prev = layer.prev else { return }

        weights = prev.neurons.map { _ in random_01() }
    }


    func calc()
    {
        sum = bias

        // input-layer neurons have no inputs, just bias
        if let prev = layer.prev {
            prev.neurons.eachIndexValue {
                self.sum += $1.output * self.weights[$0]
            }
        }

        func sigmoid(x:Double)->Double {  return 1.0 / ( 1.0 + exp(-x) )  }

        output = sigmoid(sum)
    }


    func calc_delta()
    {
        guard let next = layer.next else { return }

        /*
            Derivation http://pasteboard.co/1FkkVVBA.png
            
            NEURON P (this layer)                              NEURON Q (each neuron of next layer)
            ... -> p.sum -> sigmoid -> (p.output)      ->      * q.weights[p.index] -> q.sum -> ...
        */

        delta = 0.0
        for Q in next.neurons {
            let    dE_dQSum = Q.delta,
                dQSum_dPOut = Q.weights[index],
                dPOut_dPSum = output * (1.0-output)

            let dE_dPSum = dE_dQSum * dQSum_dPOut * dPOut_dPSum

            delta += dE_dPSum
        }
    }


    func adjust_weights()
    {
        guard let prev = layer.prev else { return }

        prev.neurons.eachIndexValue {
            self.weights[$0] -= self.net.learning_rate * self.delta * $1.output
        }
        bias -= net.learning_rate * delta
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - 

final class Layer
{
    unowned let net: Net

    weak var prev: Layer? = nil
    weak var next: Layer? = nil

    let index: Int

    var neurons = [Neuron] ()

    init( net:Net, index:Int )
    {
        (self.net, self.index) = (net, index)
    }


    // first initialise all layers, then populate all layers
    func setup()
    {
        if index > 0                  { prev = net.layers[index-1] }
        if index < net.layers.count-1 { next = net.layers[index+1] }

        neurons = (0 ..< net.neuronsInLayer[index]).map {
            Neuron( net:self.net, layer:self, index:$0 )
        }
    }


    func forward_propagate()
    {
        guard let next = next else { return }

        // update next layer from this layer's output
        for N in next.neurons {
            N.calc()
        }

        next.forward_propagate()
    }


    func back_propagate_deltas()
    {
        if index <= 1 { return }

        for N in prev!.neurons {
            N.calc_delta()
        }

        prev!.back_propagate_deltas()
    }

    func adjust_weights()
    {
        for N in neurons {
            N.adjust_weights()
        }
    }
}

// - - - - - - - - - - - - - - - - - - - - - - - - - 

public final class Net
{
    var layers = [Layer] ()
    var neuronsInLayer : [Int]

    var learning_rate = 0.9

    public init( layerSizes: [Int] )
    {
        neuronsInLayer = layerSizes

        // create layers
        layers = (0 ..< layerSizes.count).map { Layer( net:self,  index:$0 ) }

        // link
        for L in layers { L.setup() }
    }

    public func train( inputs:[Double], _ expected_outputs:[Double] ) -> Double
    {
        let (inputLayer, outputLayer) = (layers.first!, layers.last!)

        if inputs.count != inputLayer.neurons.count {
            print( "wrong number of inputs supplied" )
            return -1
        }
        if expected_outputs.count != outputLayer.neurons.count {
            print( "wrong number of outputs supplied" )
            return -1
        }

        for N in inputLayer.neurons {
            N.output = inputs[N.index]
        }

        inputLayer.forward_propagate()

        // calculate deltas for output layer
        for (N,y) in zip( outputLayer.neurons, expected_outputs )
        {
            let a = N.output,
                da_dz = a*(1.0-a),      // a = sigma(z) so da/dz = a(1-a)
                dE_da = a-y,            // E = sum{  1/2 ( a_i - y_i )^2  }  so dE/da = a_i - y_i
                dE_dz = dE_da * da_dz

            N.delta = dE_dz
        }

        outputLayer.back_propagate_deltas()

        // layers.foreach
        for L in layers {
            L.adjust_weights()
        }

        print( outputLayer.neurons.first!.output )

        return outputLayer.neurons.first!.output
    }
}

// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

// XOR
var net = Net( layerSizes: [2,3,1] )

for _ in 1 ... 3000
{
    //net.learning_rate *= 0.999

    net.train( [0,0], [0] )
    net.train( [0,1], [1] )
    net.train( [1,0], [1] )
    net.train( [1,1], [0] )

    print ("- - - - -" )
}

[0.1, 0.9]
