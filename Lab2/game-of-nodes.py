#!/usr/bin/env python

# Written by Andrew Schaffer

import pandas as pd
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import sys
from white_walker_search import white_walker_search
from jon_snow_search import jon_snow_search

# getInt
#   Parameters:
#       value   -   The string to convert to a number
#   Purpose:
#       Convert a string to an int or return -1 for failure
def getInt(value):
    try:
        result = int(value)
    except:
        result = -1
    return result


# drawGraph
#   Parameters:
#       value   -   The graph to visualize
#   Purpose:
#       Display a windowed graph
def drawGraph(graph):
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, cmap=plt.get_cmap('jet'),
        node_size=50)
    nx.draw_networkx_labels(graph, pos, font_size=10)
    nx.draw_networkx_edges(graph, pos, arrows=False)
    plt.show()


###################
# Begin Main Body #
###################
print('Written by Andrew Schaffer')
data = pd.read_csv('data.csv', header=None)                     #Open the data file for reading
westeros = nx.Graph()                                           #Initialize the graph
sys.stdout = open('lab2Out.txt', 'w')                           #Open an output file and divert stdout to it
for index, row in data.iterrows():                              #Read each row of the file
    
    index1 = getInt(row[1])
    index2 = getInt(row[2])

    if (index1 != -1) and (index2 != -1):                       #This is a new node(city) to plot
        print('Adding Node: %s' % (row[0]))
        westeros.add_node(row[0], pos=(index1, index2))
    elif (index1 == -1) and (index2 != -1):                     #This is an edge between cities to add
        print('Adding Edge with Weight %3d: %s <-> %s' % (index2, row[0], row[1])) 
        westeros.add_edge(row[0], row[1], weight=index2)
    else:
        print("Unknown file format: %s" % (row))
        exit(1)                                                 
                                                                #End graph plotting
                                                                #Begin white walker and jon snow searches
print('##########################################\n############## White Walkers #############\n##########################################')
wws = white_walker_search(westeros, 'The Wall')
result = wws.destroy()
print('White Walker Total Distance = %d' % (result['distance']))
print('##########################################\n################ Jon Snow ################\n##########################################')
jss = jon_snow_search(westeros, 'Trader Town')
result = jss.save()
print('Jon Snow Total Distance = %d' % (result))
drawGraph(westeros)
