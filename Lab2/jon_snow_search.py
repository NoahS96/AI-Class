#!/usr/bin/env python

#Written by Andrew Schaffer

import networkx as nx
import sys

class jon_snow_search:
    
    start = None                                                    #The city to start in
    destination = None                                              #The place to reach
    westeros = None                                                 #The graph object passed in __init__
                                                                    #because the graph contains node objects, not just strings.

    def __init__(self, graph, start='Trader Town', destination='The Wall'):
        self.westeros = graph
        self.start = start
        self.destination = destination

    # Algorithm:
    # save
    #   Purpose:
    #       Reach the wall from the designated starting city and return the distance travelled.
    #   Notes:
    #      This function uses dijkstra's algorithm to reach the destination. 
    def save(self):
        path = list(nx.dijkstra_path(self.westeros, self.start, self.destination))
        totalDistance = 0
        for i in range(0, len(path)-1):
            #print('%s -> %s' % (path[i], path[i+1]))
            edgeSet = list(self.westeros.edges(path[i], data=True))
            dest = path[i+1]
            dist = 0
            for edge in edgeSet:
                if edge[1] == dest:
                    dist = edge[2]['weight']
            print('Visiting %-28s\tdistance %d' % (path[i+1], dist))
            totalDistance += dist
        return totalDistance
