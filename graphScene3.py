from manim import *
import random
import networkx as nx
import numpy as np
from collections import deque

def getOutNeighbors(DiGraph, vertex, edges):
        #We assume there are no self-loops, as we assume edge_set[i][1] is always distinct from edge_set[i][0]
        edge_set = edges
        size = len(edges)
        outNeighbors = []
        for i in range(size):
                if vertex == edge_set[i][0]: #If vertex is 1st entry of the directed edge, then this edge is an outgoing edge
                        if edge_set[i][1] not in outNeighbors:
                                outNeighbors.append(edge_set[i][1])
        return outNeighbors

def percolateDirected(graph, vertices, edges, prevInfected, prob = 1.0): #Input directed graph, infected nodes as prevInfected, probability, layout
        vertex_set = vertices
        edge_set = edges #Directed edge set
        curInfected = []
        for v in prevInfected:
            curInfected.append(v)
        print(curInfected)
        infectedEdges = []
        for v in prevInfected:
                currentNeighborSet = getOutNeighbors(graph, v, edge_set)
                for j in currentNeighborSet:
                        randomNumber = random.random()
                        if randomNumber <= prob:
                                if j not in curInfected:
                                        print(curInfected)
                                        curInfected.append(j)
                                        print(curInfected)
                                        infectedEdges.append((v, j))
        print(curInfected)
        return (curInfected, infectedEdges)

class GraphScene3(Scene):
    def construct(self):
        vertices = [1, 2, 3, 4, 5, 6]
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        initialInfected = [1]
        h = Graph(vertices, edges, labels=True, layout = "circular", vertex_config = {1: {"fill_color": RED}})
        self.play(Create(h))
        self.wait(2)
        curInfected = initialInfected
        timeSteps = 2
        probability = 1
        infectedColor = '#FC6255'
        for i in range(timeSteps):
            infectedVerticesEdges = percolateDirected(h, vertices, edges, curInfected, probability)
            print(infectedVerticesEdges[0])
            print(getOutNeighbors(h, 1, edges))
            curInfected = infectedVerticesEdges[0]
            infectedEdges = infectedVerticesEdges[1]
            for j in curInfected:
                self.play(h[j].animate.set_color(color = infectedColor, family = False), run_time = 0.2)
            self.wait(2)
            #maybe write one that actually updates the graph itself, using config and a loop to change vertex colors? or you could just do the animate

def copyNX(graph):
    #This function takes a manim graph and outputs a networkx graph object
    vertices = list(graph.vertices)
    edges = list(graph.edges)
    g = nx.Graph()
    g.add_nodes_from(vertices)
    g.add_edges_from(edges)
    return g

def getAdjMatrix(graph):
    #This function takes a manim graph and outputs an adjacency matrix numpy array
    g = copyNX(graph)
    return adjacency_matrix(g)

def graphFromNX(nxGraph, labelValue):
    #This function takes a NX graph and outputs a labeled manim graph, can decide whether labeled or unlabeled
    return Graph(nxGraph.nodes, nxGraph.edges, labels = labelValue)

def graphFromMatrix(adjMatrix):
    #This function takes an adjacency matrix (numpy array) and outputs a manim graph
    graph = nx.from_numpy_array(adjMatrix)
    output = graphFromNX(graph, False)
    return output

def cartesianProduct(G, H, labelFlag):
    #Input 2 manim graph, output the product as a manim graph, can decide whether labeled or unlabeled
    Gnx = copyNX(G)
    Hnx = copyNX(H)
    result = nx.cartesian_product(Gnx, Hnx)
    print(result)
    output = graphFromNX(result, labelFlag)
    return output

def generalProductNX(G, H, vector):
    adjG = adjacency_matrix(G)
    sizeG = adjG.ndim
    adjH = adjacency_matrix(H)
    sizeH = adjH.ndim
    complementG = nx.complement(G)
    complementG = adjacency_matrix(complementG)
    complementH = nx.complement(H)
    complementH = adjacency_matrix(complementH)
    identityG = np.ones((sizeG, sizeG))
    identityH = np.ones((sizeH, sizeH))
    resultMatrix = np.zeros(sizeG*sizeH, sizeG*sizeH, 8)
    if bool(vector[0]):
        resultMatrix[:, :, 0] = np.kron(identityG, adjH)
    if bool(vector[1]):
        resultMatrix[:, :, 1] = np.kron(identityG, complementH)
    if bool(vector[2]):
        resultMatrix[:, :, 2] = np.kron(adjG, identityH)
    if bool(vector[3]):
        resultMatrix[:, :, 3] = np.kron(adjG, adjH)
    if bool(vector[4]):
        resultMatrix[:, :, 4] = np.kron(adjG, complementH)
    if bool(vector[5]):
        resultMatrix[:, :, 5] = np.kron(complementG, identityH)
    if bool(vector[6]):
        resultMatrix[:, :, 6] = np.kron(complementG, adjH)
    if bool(vector[7]):
        resultMatrix[:, :, 7] = np.kron(complementG, complementH)
    outputAdj = np.logical_or(resultMatrix[:, :, 0], resultMatrix[:, :, 1])
    outputAdj = np.logical_or(outputAdj, resultMatrix[:, :, 2])
    outputAdj = np.logical_or(outputAdj, resultMatrix[:, :, 3])
    outputAdj = np.logical_or(outputAdj, resultMatrix[:, :, 4])
    outputAdj = np.logical_or(outputAdj, resultMatrix[:, :, 5])
    outputAdj = np.logical_or(outputAdj, resultMatrix[:, :, 6])
    outputAdj = np.logical_or(outputAdj, resultMatrix[:, :, 7])
    output = nx.from_numpy_array(outputAdj)
    return output

def iteratedProductPower(G, n, vector):
    #Input an NX graph, output its n-th power as an NX graph according to vector definition of graph product
    output = G
    for i in range(n):
        output = generalProductNX(output, output, vector)
    return output

def TriFreeCliqueList(G):
    #Input triangle-free NX graph, output list of its cliques
    numNodes = G.number_of_nodes()
    numEdges = G.number_of_edges()
    total = numNodes + numEdges + 1
    A = adjacency_matrix(G)
    #Create a dictionary to store the cliques, 1st entry is empty clique
    S = {}
    S[1] = []
    for i in range(2, numNodes+2):
        S[i] = [i-1]
    for i in range(numNodes+1, total+1):
        S[i] = list(G.edges[i-(numNodes+1)])
    return S

def isCliqueAdjacent(S1, S2):
    cliqueI = np.array(S1)
    cliqueJ = np.array(S2)
    diffIJ = np.setdiff1d(cliqueI, cliqueJ)
    diffJI = np.setdiff1d(cliqueJ, cliqueI)
    # Test if cliques i and j differ by the presence of 1 vertex, if so make them adjacent
    if i != j and ((len(diffIJ) == 1 and len(diffJI) == 0) or (len(diffJI) == 1 and len(diffIJ) == 0)):
        return True
    else:
        return False

def simplexGraph(G):
    #Input a NX graph, output its simplex graph in NX format, output labeled with integers
    cliqueList = deque(enumerate_all_cliques(G))
    cliqueList.appendleft([])
    length = len(cliqueList)
    if length == 1:
        output = nx.Graph()
        output.add_node(1)
        return output
    elif length == 2:
        output = nx.Graph()
        output.add_nodes_from([1, 2])
        output.add_edge((1, 2))
        return output
    else:
        vertices = range(1, length+1)
        edges = []
        for i in range(length-1):
            for j in range(i+1, length):
                if isCliqueAdjacent(cliqueList[i], cliqueList[j]):
                    edges.append((i, j))
        output = nx.Graph()
        output.add_nodes_from(vertices)
        output.add_edges_from(edges)
        return output

def simplexGraphTriFree(G):
    #Input a triangle-free NX graph, output its simplex graph in NX format
    #Makes use of TriFreeCliqueList. Output is labeled with integers.
    cliqueDict = TriFreeCliqueList(G)
    length = len(cliqueDict)
    #Set the values for the rows and columns NOT corresponding to 0-clique
    if length == 1:
        output = nx.Graph()
        output.add_node(1)
        return output
    elif length == 2:
        output = nx.Graph()
        output.add_nodes_from([1, 2])
        output.add_edge((1, 2))
        return output
    else:
        vertices = range(1, length+1)
        edges = []
        for i in range(length-1):
            for j in range(i+1, length):
                if isCliqueAdjacent(cliqueDist[i], cliqueDict[j]):
                    edges.append((i, j))
        output = nx.Graph()
        output.add_nodes_from(vertices)
        output.add_edges_from(edges)
        return output

def iteratedSimplex(G, n, triFreeFlag):
    #Input a NX graph and boolean value indicating whether it is triangle-free, output its n-th simplex graph in NX format
    #Makes use of simplexGraph() or simplexGraphTriFree. If you don't know whether the graph is triangle-free, just put False.
    H = G.copy()
    if triFreeFlag:
        for i in range(n):
            H = simplexGraphTriFree(H)
    else:
        if n == 1:
            return simplexGraph(H)
        else:
            H = simplexGraph(H)
            for i in range(n-1):
                H = simplexGraphTriFree(H)
    return H

def unSimplexG(G):
    #Input an NX simplex graph, output its original graph in NX format
    H = G.copy()
    nx.convert_node_labels_to_integers(H, 1)
    degreeList = np.array([val for (node, val) in H.degree()])

    #Find the max degree and any vertex with max degree and its neighbors
    maxDegreeIndex = np.argmax(degreeList)
    maxDegreeVertex = H.nodes[maxDegreeIndex]
    neighborList = list(G[maxDegreeVertex])

    #Remove maxDegreeVertex from simplex graph
    H.remove_node(maxDegreeVertex)

    #Square what remains
    H = nx.power(H, 2)

    #Get subgraph of maxDegreeVertex's neighbors
    F = subgraph(H, neighborList)
    return F

def isSimplexGraph(G):
    #Input an NX graph, output a boolean value determining whether it is or is not a simplex graph.
    #Makes use of unSimplexG() and simplexGraph()
    F = unsimplex(G)
    return is_isomorphic(simplex(F), G)

def triFreeSimplexOrder(G, n):
    #Input a triangle-free NX graph G and integer n, output the order and size of the n-th simplex graph of G
    numNodes = G.number_of_nodes
    numEdges = G.number_of_edges
    newNumNodes = 1 + numNodes + numEdges
    return newNumNodes

def triFreeSimplexSize(G, n):
    #Input a triangle-free NX graph G and integer n, output the order and size of the n-th simplex graph of G
    numNodes = G.number_of_nodes
    numEdges = G.number_of_edges
    newNumEdges = numNodes + 2*numEdges
    return newNumEdges

def iteratedSimplexOrder(G, n):
    #Input a graph, and integer n. Outputs the n-th simplex graph's number of vertices.
    firstSimplex = simplexGraph(G)
    if n == 1:
        return firstSimplex.number_of_nodes
    else:
        return triFreeSimplexOrder(firstSimplex, n-1)

def iteratedSimplexSize(G, n):
    #Input a graph, and integer n. Output n-th simplex graph's number of edges.
    firstSimplex = simplexGraph(G)
    if n == 1:
        return firstSimplex.number_of_nodes
    else:
        return triFreeSimplexSize(firstSimplex, n-1)

def iteratedSimplexDensity(G, n):
    #Input a graph, and integer n. Output n-th simplex graph's density.
    firstSimplex = simplexGraph(G)
    if n == 1:
        return firstSimplex.number_of_edges/firstSimplex.number_of_nodes
    else:
        order = iteratedSimplexOrder(firstSimplex, n-1)
        size = iteratedSimplexSize(firstSimplex, n - 1)
        return size/order

def iteratedProductSize(G, n, vector):
    #Input a graph, iteration number >= 2, product definition as vector, output size of n-th product power graph
    productGraph = iteratedProductPower(G, n, vector)
    return productGraph.number_of_edges

def iteratedProductOrder(G, n, vector):
    #Input graph, iteration number >= 2, product definition, output order of n-th product power graph
    productGraph = iteratedProductPower(G, n, vector)
    return productGraph.number_of_nodes

def iteratedProductDensity(G, n, vector):
    productGraph = iteratedProductPower(G, n, vector)
    return productGraph.number_of_edges/productGraph.number_of_nodes

def simplexGraphLimitDensity(G):
    #Input a graph, output the limit density of its simplex graph as iteration goes to infinity
    firstSimplexG = simplexGraph(G)
    v1G = firstSimplexG.number_of_nodes
    e1G = firstSimplexG.number_of_edges

    dominantVertexCoeffG = (1 + v1G + e1G + (1 - np.sqrt(5)) / 2 - v1G * (3 - np.sqrt(5)) / 2) / np.sqrt(5)
    dominantEdgeCoeffG = (v1G + 2 * e1G - (1 - np.sqrt(5)) / 2 - e1G * (3 - np.sqrt(5)) / 2) / np.sqrt(5)

    return dominantEdgeCoeffG / dominantVertexCoeffG

def optimalSimplexGraph(G, H, whatToFind):
    #Input a graph, decide which has more vertices or edges or density in the long run iteration. Makes use of simplexGraphCoefficients()
    firstSimplexG = simplexGraph(G)
    v1G = firstSimplexG.number_of_nodes
    e1G = firstSimplexG.number_of_edges

    firstSimplexH = simplexGraph(H)
    v1H = firstSimplexH.number_of_nodes
    e1H = firstSimplexH.number_of_edges

    if whatToFind == 'order':
        dominantVertexCoeffG = (1 + v1G + e1G + (1 - np.sqrt(5))/2 - v1G*(3 - np.sqrt(5))/2)/np.sqrt(5)
        dominantVertexCoeffH = (1 + v1H + e1H + (1 - np.sqrt(5))/2 - v1H*(3 - np.sqrt(5))/2) / np.sqrt(5)
        if dominantVertexCoeffG > dominantVertexCoeffH:
            return G
        elif dominantVertexCoeffG < dominantVertexCoeffH:
            return H
        else:
            return 0
    elif whatToFind == 'size':
        dominantEdgeCoeffG = (v1G + 2*e1G - (1 - np.sqrt(5))/2 - e1G*(3 - np.sqrt(5))/2)/np.sqrt(5)
        dominantEdgeCoeffH = (v1H + 2*e1H - (1 - np.sqrt(5))/2 - e1H*(3 - np.sqrt(5))/2)/np.sqrt(5)
        if dominantEdgeCoeffG > dominantEdgeCoeffH:
            return G
        elif dominantEdgeCoeffG < dominantEdgeCoeffH:
            return H
        else:
            return 0
    elif whatToFind == 'density':
        dominantVertexCoeffG = (1 + v1G + e1G + (1 - np.sqrt(5))/2 - v1G*(3 - np.sqrt(5))/2)/np.sqrt(5)
        dominantVertexCoeffH = (1 + v1H + e1H + (1 - np.sqrt(5))/2 - v1H*(3 - np.sqrt(5))/2) / np.sqrt(5)

        dominantEdgeCoeffG = (v1G + 2*e1G - (1 - np.sqrt(5))/2 - e1G*(3 - np.sqrt(5))/2)/np.sqrt(5)
        dominantEdgeCoeffH = (v1H + 2*e1H - (1 - np.sqrt(5))/2 - e1H*(3 - np.sqrt(5))/2)/np.sqrt(5)

        limitDensityG = dominantEdgeCoeffG / dominantVertexCoeffG
        limitDensityH = dominantEdgeCoeffH / dominantVertexCoeffH

        if limitDensityG > limitDensityH:
            return G
        elif limitDensityG < limitDensityH:
            return H
        else:
            return 0

def simplexGraphOrderCoefficients(G):
    #Input a graph, output the coefficients on its simplex graph order explicit formula's exponential terms, output[0] is dominant term's coefficient.
    firstSimplex = simplexGraph(G)
    v1 = firstSimplex.number_of_nodes
    e1 = firstSimplex.number_of_edges
    dominantCoeff = (1 + v1 + e1 + (1 - np.sqrt(5))/2 - v1*(3 - np.sqrt(5))/2)/np.sqrt(5)
    transientCoeff = (v1*(3 + np.sqrt(5))/2 - v1 - e1 - 1 - (1 + np.sqrt(5))/2)/np.sqrt(5)
    output = [dominantCoeff, transientCoeff]
    return output

def simplexGraphEdgeCoefficients(G):
    #Input a graph, output the coefficients on its simplex graph order explicit formula's exponential terms, output[0] is dominant term's coefficient.
    firstSimplex = simplexGraph(G)
    v1 = firstSimplex.number_of_nodes
    e1 = firstSimplex.number_of_edges
    dominantCoeff = (v1 + 2*e1 - (1 - np.sqrt(5))/2 - e1*(3 - np.sqrt(5))/2)/np.sqrt(5)
    transientCoeff = (e1*(3 + np.sqrt(5))/2 - v1 - 2*e1 + (1 + np.sqrt(5))/2)/np.sqrt(5)
    output = [dominantCoeff, transientCoeff]
    return output

def iteratedSimplexOrderEfficient(G, n):
    #Input a graph, output the order of its n-th simplex graph using the explicit formula. This only needs to calculate one simplex graph.
    if n < 2:
        #We don't want n less than 2, invalid input
        return False
    coefficients = simplexGraphOrderCoefficients(G)
    base1 = (3 + np.sqrt(5))/2
    base2 = (3 - np.sqrt(5))/2
    output = 1 + coefficients[0]*(numpy.power(base1, n-1)) + coefficients[1]*(numpy.power(base2, n-1))
    return output

def iteratedSimplexSizeEfficient(G, n):
    #Input a graph, output the size of its n-th simplex graph using the explicit formula. This only needs to calculate one simplex graph.
    if n < 2:
        #We don't want n less than 2, invalid input
        return False
    coefficients = simplexGraphSizeCoefficients(G)
    base1 = (3 + np.sqrt(5))/2
    base2 = (3 - np.sqrt(5))/2
    output = coefficients[0]*(numpy.power(base1, n-1)) + coefficients[1]*(numpy.power(base2, n-1)) - 1
    return output

def iteratedSimplexDensityEfficient(G, n):
    #Input a graph, output the density of its n-th simplex graph using the explicit formula. This only needs to calculate one simplex graph.
    if n < 2:
        return False
    firstSimplex = simplexGraph(G)
    v1 = firstSimplex.number_of_nodes
    e1 = firstSimplex.number_of_edges

    dominantEdgeCoeff = (v1 + 2 * e1 - (1 - np.sqrt(5)) / 2 - e1 * (3 - np.sqrt(5)) / 2) / np.sqrt(5)
    transientEdgeCoeff = (e1 * (3 + np.sqrt(5)) / 2 - v1 - 2 * e1 + (1 + np.sqrt(5)) / 2) / np.sqrt(5)
    dominantVertexCoeff = (1 + v1 + e1 + (1 - np.sqrt(5))/2 - v1*(3 - np.sqrt(5))/2)/np.sqrt(5)
    transientVertexCoeff = (v1*(3 + np.sqrt(5))/2 - v1 - e1 - 1 - (1 + np.sqrt(5))/2)/np.sqrt(5)
    coefficients = [dominantVertexCoeff, transientVertexCoeff, dominantEdgeCoeff, transientEdgeCoeff]

    base1 = (3 + np.sqrt(5))/2
    base2 = (3 - np.sqrt(5))/2

    order = 1 + coefficients[0]*(numpy.power(base1, n-1)) + coefficients[1]*(numpy.power(base2, n-1))
    size = coefficients[2]*(numpy.power(base1, n-1)) + coefficients[3]*(numpy.power(base2, n-1)) - 1
    return size/order

def arbitraryProduct(G, H, vector):
    #Input 2 manim graph, and binary vector of length 8, output the arbitrary product as a manim graph
    Gnx = copyNX(G)
    Hnx = copyNX(H)
    result = generalProductNX(Gnx, Hnx, vector)
    output = graphFromNX(result, False)
    return output

def nepsDistanceVector(graphVector, vertexTuple1, vertexTuple2):
    #Input two vertices of an NEPS of undirected graphs, outputs their distance vector
    length = len(vertexTuple1)
    output = np.zeros((1, length))
    for i in range(length):
        curGraph = graphVector[i]
        if vertexTuple1[i] in list(curGraph.neighbors(vertexTuple2[i])):
            output[i] = 1
    return list(output)



def NEPS(graphVector, basis):
    #Input a vector of graphs, labeled with integers, and the NEPS basis (a set of distance vectors for NEPS vertices that will be made adjacent). Needs itertools.
    #Outputs a graph labeled with integers.
    edges = []
    numberOfGraphs = length(graphVector)
    order = 1
    graphOrderList = []

    #Find order of NEPS and create a list of orders of graphs in graphVector to be used in creating the vertex set of NEPS.
    for i in range(numberOfGraphs):
        curOrder = graphVector[i].number_of_nodes
        order = order * curOrder
        graphOrderList.append(range(1, curOrder+1))

    #Get vertex set of NEPS
    vertexSet = itertools.product(*graphOrderList)

    for i in range(order-1):
        for j in range(i+1, order):
            #Test to see if distance vector between i-th vertex and j-th vertex is in basis. Note: vertexSet[i] and vertexSet[j] must be vertex names of the i-th and j-th input graphs, respectively
            if nepsDistanceVector(graphVector, vertexSet[i], vertexSet[j]) in basis:
                edges.append((i, j))
    output = nx.Graph()
    output.add_nodes_from(range(order))
    output.add_edges_from(edges)
    return output

class GraphScene4(Scene):
    def perc(self, graph, initialInfected, timesteps, waitTime, infectedColor, probability):
        #This function animates a percolation process on a graph
        vertices = list(graph.vertices)
        edges = list(graph.edges)
        curInfected = []
        for i in initialInfected:
            curInfected.append(i)
        for i in range(timesteps):
            infectedVerticesEdges = percolateDirected(graph, vertices, edges, curInfected, probability)
            curInfected = infectedVerticesEdges[0]
            infectedEdges = infectedVerticesEdges[1]
            vertexAnimations = []
            edgeAnimations = []
            for j in curInfected:
                vertexAnimations.append(graph[j].animate.set_color(RED, family = False))
            for j in infectedEdges:
                edgeAnimations.append(graph.edges[j].animate.set_color(RED, family = False))
            self.play(*edgeAnimations)
            self.play(*vertexAnimations)
            self.wait(waitTime)
    def construct(self):
        vertices = [1, 2, 3, 4, 5, 6]
        edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        initialInfected = [1]
        h = Graph(vertices, edges, labels = True, layout = "circular", vertex_config = {k: {"fill_color": RED} for k in initialInfected})
        hNX = copyNX(h)
        self.play(Create(h))
        self.wait(2)
        self.perc(h, initialInfected, 3, 1, '#FC6255', 1.0)
        self.wait(1)
        product = cartesianProduct(h, h, False)
        product.layout = "circular"
        self.play(Create(product))
        self.wait(5)
