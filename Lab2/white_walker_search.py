#!/usr/bin/env python

#Written by Andrew Schaffer

import networkx as nx
import sys

class white_walker_search:
    
    start = None                                                    #The city to start in
    westeros = None                                                 #The graph object passed in __init__
    goalCities = []                                                 #A list of all the names of the cities, this is needed
                                                                    #because the graph contains node objects, not just strings.
    spanningTree = None

    def __init__(self, graph, start='The Wall'):
        self.westeros = graph
        self.start = start
        self.spanningTree = nx.minimum_spanning_tree(graph)
        for city in graph.nodes():
            self.goalCities.append(city)
        #self.westeros = self.spanningTree

    # Weighted Nearest Neighbor
    # destroy
    #   Parameters:
    #       node     -   The current city in which the white walkers are visiting
    #       visited  -   A list of all the cities previously visited
    #       prevDist -   The distance from the previous city
    #   Purpose:
    #       Visit all cities on the graph and return the total distance traveled in the search
    #   Notes:
    #       This method uses a nearest neighbor approach. From the current city it chooses the next closest one 
    #       and continues until 
    def destroy(self, node=None, visited=None, prevDist=0):

        if node is None:                                            #If node and visitid is None then this should 
            node = self.start                                       #be the first time destroy() is called
        if visited is None:
            visited = []
        if set(visited) == set(self.goalCities):                    #We have reached our goal, return distance
            return {'distance':0, 'visited':visited}

        if node not in visited:
            visited.append(node)                                    #Add the city to visited
        edges = list(self.westeros.edges(node, data=True))

        accumDistance = 0                                               #The distance calculated for this recursion step
        for edge in sorted(edges, key=lambda edge:edge[2]['weight']):   #Check for shortest distance and decide path
            neighbor = edge[1]                                     #Get the name of the city from the edge
            if (neighbor not in visited):               
                print('Visiting %-28s\tdistance %d' % (neighbor, edge[2]['weight']))
                result = self.destroy(node=neighbor, visited=visited, prevDist=edge[2]['weight'])
                visited = result['visited']                         #After the recursive call, save the returned visited
                accumDistance += edge[2]['weight']                  #list and add the distance to accumDistance
                accumDistance += result['distance']
        
        if set(visited) != set(self.goalCities):                    #If not all cities are visited, we are backtracking
            accumDistance += prevDist
            print('Returning from %-20s\tdistance %d' % (node, prevDist))
        return {'distance':accumDistance, 'visited':visited}       

    
















