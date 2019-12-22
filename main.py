import os
from itertools import permutations
from collections import deque
import networkx as nx
from collections import defaultdict
import sys
import pandas as pd
import matplotlib.pyplot as plt
sys.setrecursionlimit(1500)

################## START GABRIELE
path = os.getcwd()+'\\data'
path_dist = path + '\\distance.gr'
path_info = path + '\\node_info.co'
path_trav_time = path + '\\travel_time.gr'

def clean_data(path):
    with open(path, 'r') as x:
        lst = x.readlines()
        out = []
        for i in lst:
            row = i[2:-1] # deleting all elements we don't need 
            out.append(row)
        clean_out = out[7:]
        clean_out2 = []
        for j in range(len(clean_out)):
            clean_out2.append(list(map(int, clean_out[j].split())))        
    return clean_out2

all_neighbours = []
visited_nodes = []

def get_index_by_func(func):
    if func == 3:
        return 2
    elif func == 1:
        return 1
    elif func == 2:
        return 0


    
def add_neighbour(node):
    if node not in all_neighbours:
        all_neighbours.append(node)

def get_neighbours(origin, node, cur_price, d):
    for n in neighbours[node]:
        if n != origin and n not in visited_nodes:
            visited_nodes.append(n)
            temp_price = cur_price + price_nodes[(node,n)][0][price_index]
            if temp_price <= d:
                add_neighbour(n)
                get_neighbours(origin, n, temp_price, d)

def get_nodes_by_price(v, func, d):
    get_neighbours(v, v, 0, d)
################ END GABRIELE




################ START TRINA
def get_shortest_path(graph, start,end):
    G = graph
    to_visit = [start]
    path = []
    i = 0
    visited = []
    while end not in to_visit:
        if to_visit[i] not in visited:

            for k in list(G.neighbors(to_visit[i])):
                to_visit.append(k)
            path.append(list(G.neighbors(to_visit[i])))
            visited.append(to_visit[i])
        i = i + 1
    element = visited[-1]
    i = 1
    result = [end, element]
    while i <100000 and element !=start:
        p = []
        for k in range(len(path)):
            for i in range(len(path[k])):
                if path[k][i] == element:
                    p.append(k)
        element = visited[p[0]]
        i = i+1
        result.append(element)
    result.reverse()
    return(result)
        
def get_distance_nodes(function, start, end):
    a = function.loc[(function[1] == start) & (function[2] == end)]
    if len(a)>0 : 
        return int(a[3])
    else: 
        return 0
    
def get_distance(function, start, end):
    path = get_shortest_path(G, start,end)
    i =0
    sum = 0
    while i < len(path)-1:
        sum = sum + get_distance_nodes(function, path[i], path[i+1])
        i = i+1
        
    return(sum)
        

from itertools import permutations
def f2(list_nodes, function):
    l = []
    for s in permutations(list_nodes):
        case = list(s)
        sum_tot = 0
        i = 0
        while i < len(case)-1:
            sum_tot = sum_tot + get_distance(function, case[i], case[i+1])
            i = i +1
        element = (case, sum_tot)
        l.append(element)
    l = sorted(l, key=lambda tup: (tup[1]),reverse = False)
    best_net = l[0][0]

    i = 0
    list_edge = []
    while i < len(best_net)-1:
            list_edge = list_edge + get_shortest_path(G, best_net[i],best_net[i+1])
            i = i+1
    return(list_edge)
 

    
def visu_2(G,list_nodes,function):
    result = f2(list_nodes, function)
    edges = []
    for k in result:
        for i in list(G.neighbors(k)):
            edges.append((k,i))
    r = []
    i = 0
    while i < len(result) -1:
        edges.append((result[i],result[i+1]))
        r.append((result[i],result[i+1]))
        i = i+1
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)


    G1 = nx.DiGraph()
    G1.add_edges_from(edges)

    val_map = {1: 2.0,
               }

    values = [val_map.get(node, 0.25) for node in G1.nodes()]

    # Specify the edges you want here

    red_edges = r
    edge_colours = ['black' if not edge in red_edges else 'red'
                    for edge in G1.edges()]
    black_edges = [edge for edge in G1.edges() if edge not in red_edges]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G1)
    nx.draw_networkx_nodes(G1, pos, cmap=plt.get_cmap('jet'), 
                           node_color = values, node_size = 1)
    nx.draw_networkx_labels(G1, pos)
    nx.draw_networkx_edges(G1, pos, edgelist=red_edges, edge_color='red', arrows=True)
    nx.draw_networkx_edges(G1, pos, edgelist=black_edges, edge_color='black',arrows=False)
    plt.show()
	
##################### END TRINA


##################### START MOhanraj
#vertex class to store details of each node
class Vertex:
    def __init__(self,key):
        self.id = key #node name or number
        self.connectedTo = {}
        self.coordinate =()
    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr.getId()] = weight
    #def __str__(self):
        #return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])
        #return self.id
    #method to return the nodes or vertices which are connected to the vertex x
    def getConnections(self):
        return self.connectedTo.keys()
    #method to return the id of the vertex object
    def getId(self):
        return self.id
    #method to return the weight of the 
    def getWeight(self,nbr):
        if nbr not in self.connectedTo:
            return -1
        else:
            return self.connectedTo[nbr]
    #method to return coordinates of node for visualization
    #initially will be empty we have to load data to it
    def getCoord(self):
        return self.coordinate
    #to load coordinate
    def setCoord(self,longitude,latitude):
        self.coordinate = tuple([latitude,longitude])
#class "Graph" to create a graph
class Graph:
    #dictionary of list of vertices of vertex objects
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0
    #method to add vertex
    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        #creating vertex object
        newVertex = Vertex(key)
        #storing corresponding vertext object reference in dictionary
        self.vertList[key] = newVertex
        return newVertex
    #pass name of the vertex it will return corresponding vertex object
    #if none present it will return -1
    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return -1
    
    def __contains__(self,n):
        return n in self.vertList
    #method to add edges
    def addEdge(self,f,t,weight=0):
        #if either of vertex not in the graph adding vertex to graph
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        #adding edge afterwards
        self.vertList[f].addNeighbor(self.vertList[t], weight)
    #method to get vertices
    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())
    #method to find the shortest route between v1 and v2 vertices using dijsktra algorithm
    def find_shortest(self, start, end):
        if self.getVertex(start)!= -1 and self.getVertex(end)!=-1 :
            #shortest paths stored in below dictionary
            paths = {start: (None, 0)}
            #current vertex
            current = start
            #to keep track of nodes that are already visited
            visited = set()
            #till we reach destination loop through and find path
            while current != end:
                #add current vertex to the visited set
                visited.add(current)
                #get the nodes that are reachable from current vertex
                dest = self.getVertex(current).getConnections()
                #current weight of the path
                weight_current = paths[current][1]

                for next_vertex in dest:
                    #checking weights of the next reachable vertex from current vertex
                    weight = self.getVertex(current).getWeight(next_vertex)+ weight_current
                    #if its shortest path store it 
                    if next_vertex not in paths:
                        paths[next_vertex] = (current, weight)
                    else:
                        current_shortest_weight = paths[next_vertex][1]
                        if current_shortest_weight > weight:
                            paths[next_vertex] = (current, weight)
                #get all the destination from the current vertex
                next_dest = {node: paths[node] for node in paths if node not in visited}
                #if there is no rechable return -1 as it not possible
                if not next_dest:
                    return -1
                # next vertex is the destination with the lowest weight
                current = min(next_dest, key=lambda k: next_dest[k][1])

            # Walk back through dest in shortest path
            path = []
            weights = []
            while current is not None:
                path.append(current)
                next_vertex = paths[current][0]
                weights.append(paths[current][1])
                current = next_vertex
            # reverse the path and return along with weight
            path = path[::-1]
            return weights[0],path
        else:
            return -1
##################### END MOhanraj

##################### START TRINA PART 4
def get_shortest_path(graph, start,end):
    G = graph
    to_visit = [start]
    path = []
    i = 0
    visited = []
    while end not in to_visit:
        if set(to_visit) == set(visited):
            return([0])
        if to_visit[i] not in visited:

            for k in list(G.neighbors(to_visit[i])):
                to_visit.append(k)
            path.append(list(G.neighbors(to_visit[i])))
            visited.append(to_visit[i])
       
        i = i + 1
    element = visited[-1]
    i = 1
    result = [end, element]
    while i <100000 and element !=start:
        p = []
        for k in range(len(path)):
            for i in range(len(path[k])):
                if path[k][i] == element:
                    p.append(k)
        element = visited[p[0]]
        i = i+1
        result.append(element)
    result.reverse()
    return(result)
        
def get_distance_nodes(function, start, end):
    a = function.loc[(function[1] == start) & (function[2] == end)]
    if len(a)>0 : 
        return int(a[3])
    else: 
        return 0
    
def get_distance(function, start, end):
    path = get_shortest_path(G, start,end)
    i =0
    sum = 0
    while i < len(path)-1:
        sum = sum + get_distance_nodes(function, path[i], path[i+1])
        i = i+1
        
    return(sum)
        
def f4(G, H, list_nodes, function):
    l = []
    start = [H]
    end = [list_nodes[-1]]
    list_nodes = list_nodes[:-1] 
    for s in permutations(list_nodes):
        case = start + list(s) + end
        sum_tot = 0
        i = 0
        while i < len(case)-1:
            sum_tot = sum_tot + get_distance(function, case[i], case[i+1])
            i = i +1
        element = (case, sum_tot)
        l.append(element)
    l = sorted(l, key=lambda tup: (tup[1]),reverse = False)
    best_net = l[0][0]

    i = 0
    list_edge = []
    while i < len(best_net)-1:
            list_edge = list_edge + get_shortest_path(G, best_net[i],best_net[i+1])
            i = i+1
    return(list_edge)


    
def visu_4(G,H, list_nodes,function):
    result = f4(G, H,list_nodes, function)
    edges = []
    for k in result:
        for i in list(G.neighbors(k)):
            edges.append((k,i))
    r = []
    i = 0
    while i < len(result) -1:
        edges.append((result[i],result[i+1]))
        r.append((result[i],result[i+1]))
        i = i+1
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)


    G1 = nx.DiGraph()
    G1.add_edges_from(edges)

    val_map = {1: 2.0,
               }

    values = [val_map.get(node, 0.25) for node in G1.nodes()]

    # Specify the edges you want here

    red_edges = r
    edge_colours = ['black' if not edge in red_edges else 'red'
                    for edge in G1.edges()]
    black_edges = [edge for edge in G1.edges() if edge not in red_edges]

    # Need to create a layout when doing
    # separate calls to draw nodes and edges
    pos = nx.spring_layout(G1)
    nx.draw_networkx_nodes(G1, pos, cmap=plt.get_cmap('jet'), 
                           node_color = values, node_size = 1)
    nx.draw_networkx_labels(G1, pos)
    nx.draw_networkx_edges(G1, pos, edgelist=red_edges, edge_color='red', arrows=True)
    nx.draw_networkx_edges(G1, pos, edgelist=black_edges, edge_color='black',arrows=False)
    plt.show()


##################### END TRINA PART 4

if __name__ == '__main__':

    function = int(input("Please insert a number from 1 to 4 to choose your function: ")) # user can choose the function to run.

    if function == 1:
        
        dist = clean_data(path_dist)
        time = clean_data(path_trav_time)
        info = clean_data(path_info)

        neighbours = defaultdict(list)
        for i in range(len(dist)):
            neighbours[dist[i][0]].append(dist[i][1])

        price_nodes = defaultdict(list)

        for i in range(len(dist)):
            price_nodes[(dist[i][0], dist[i][1])].append([dist[i][2], time[i][2], 1])
        
        v = int(input('Insert node: '))
        func = int(input('Enter the type of distance threshold:\n 1.Time \n 2.Physical distance \n 3.Network distance\n'))
        d = int(input('Insert threshold: '))
        price_index = get_index_by_func(func)
        get_nodes_by_price(v, func, d)
        print(all_neighbours)

    elif function == 2:
        
        data_dist = pd.read_csv(r'D:\Data Science\ADM\HW05\USA-road-d.CAL.gr',sep = " ",header = None)
        data_time = pd.read_csv(r'D:\Data Science\ADM\HW05\USA-road-t.CAL.gr',sep = " ", header = None)
        node = pd.read_csv(r'D:\Data Science\ADM\HW05\USA-road-d.CAL.co',sep = " ", header = None)
        G=nx.Graph()
        for key,value in data_dist.iterrows():
            G.add_edge(value[1], value[2], dist=value[3] )
        visu_2(G,[3,2],data_dist)
    
    elif function == 3:

        #creating graph
        G1 = Graph()
        #Adding edges to graph thus making vertices too
        for i in range(len(dist)):
            #vertex1,vertex2,distance or time or even empty it will construct graph with weight 1
            G1.addEdge(dist[i][0],dist[i][1],dist[i][2])
    
        #Adding coordinates detail to the vertices 
        for i in range(len(info)):
            #mandatory step to check first whether the vertex present in graph or not if getvertex method return -1 
            if G1.getVertex(info[i][0])== -1 :
                print("Vertex not present ",info[i])
            else:
                G1.getVertex(info[i][0]).setCoord(int(info[i][1])/1000000,int(info[i][2])/1000000 )

        #Getting Input from the User
        while True:
            print("Enter Starting Node")
            H = input()
            if H.isalpha()==True:
                print("Enters Numbers only : Try Again")
                continue
            H = int(H)
            if G1.getVertex(H) == -1:
                print("Node not present in the Graph!!! Try again")
            else:
                break
            print("Try Again")
        print("Please Enter the sequence of Nodes one by one: (press enter to end)")
        p=[]
        while True:
            temp = input()
            if temp== "":
                break
            p.append(int(temp))
        print(H,p)
        #Finding Shortest ordered path
        Flag = -1
        current = H
        weight = 0 
        shortest_path = []
        for i in range(len(p)):
            A = G1.find_shortest(current,p[i])
            if  A == -1 :
                print("Not Possible")
            else:
                w = A[0]
                path = A[1]
                shortest_path+= path[:-1]
                weight+=w
                current = p[i]
        shortest_path.append(p[-1])
        print(shortest_path,weight)
    
    elif function == 4:

        data_dist = pd.read_csv(r'D:\Data Science\ADM\HW05\USA-road-d.CAL.gr',sep = " ",header = None)
        data_time = pd.read_csv(r'D:\Data Science\ADM\HW05\USA-road-t.CAL.gr',sep = " ", header = None)
        node = pd.read_csv(r'D:\Data Science\ADM\HW05\USA-road-d.CAL.co',sep = " ", header = None)
        data_dist = data_dist.drop_duplicates(keep='last')
        G=nx.Graph()
        for key,value in data_dist.iterrows():
            G.add_edge(value[1], value[2], dist=value[3] )
        visu_4(G, 1, [2], data_dist)
