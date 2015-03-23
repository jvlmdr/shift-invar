package main

import "github.com/jvlmdr/go-cv/rimg64"

//	type Tree struct {
//		Root *Node
//	}
//
//	func (f Tree) Score(x *rimg64.Multi) float64 {
//		if f.Feature.Eval(x) <= f.Thresh {
//			return f.Left
//		}
//		return f.Right
//	}

// Node describes a node of a decision tree.
//
// If it is a leaf node then Feature is nil,
// otherwise Left and Right must be non-nil.
type Node struct {
	Feature Feature
	Thresh  float64
	Left    *Node
	Right   *Node
	Value   float64
}

func (node *Node) isLeaf() bool {
	return node.Feature == nil
}

func (node *Node) Score(x *rimg64.Multi) float64 {
	if node.isLeaf() {
		return node.Value
	}
	if node.Feature.Eval(x) <= node.Thresh {
		return node.Left.Score(x)
	}
	return node.Right.Score(x)
}
