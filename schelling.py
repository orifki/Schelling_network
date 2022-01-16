import numpy as np #structure de données
import networkx as nx #structure réseau
from itertools import count #fonction de comptage
import matplotlib.pyplot as plt  #dessiner figures



class Area:
    def __init__(self, N, types, types_prob, preference, type_graph, r):
        self.g = self.construct_graph(N, types, types_prob, preference, type_graph, r)
        self.init_g = self.g
        self.type_g = type_graph
        self.total_nodes = len(self.g)
        self.total_unhappy = len(self.g)
        self.total_empty = len([x for x, n in self.g.nodes(data=True) if n['type'] == 0])


    def construct_graph(self, N, types, types_prob, preference, type_graph, r=2):
        if type_graph == 'grid':  q=0
        if type_graph == 'small-world': q=2
        g = nx.navigable_small_world_graph(N, p=1, q=q, r=r)

        # Add the attributes of type and preferences to nodes
        d_type={}; d_pref={}
        for n in g.nodes():
            d_type[n] = np.random.choice(types, p=types_prob)
            d_pref[n] = preference
        nx.set_node_attributes(g, d_type, 'type')
        nx.set_node_attributes(g, d_pref, 'preference')
        return g


    def init_graph(self, preference):
        self.g = self.init_g
        self.total_nodes = len(self.g)
        self.total_unhappy = len(self.g)
        self.total_empty = len([x for x, n in self.g.nodes(data=True) if n['type'] == 0])
        d_pref={}
        for n in self.g.nodes(): d_pref[n] = preference
        nx.set_node_attributes(self.g, d_pref, 'preference')


    def draw_graph(self, title, size=50):
        groups = set(nx.get_node_attributes(self.g, 'type').values()) #get unique groups
        mapping = dict(zip(sorted(groups), count()))
        colors = [mapping[self.g.nodes[n]['type']] for n in self.g.nodes()]
        pos = dict((n, n) for n in self.g.nodes())

        plt.figure()
        plt.title(title)
        nx.draw_networkx(self.g, pos=pos, with_labels=False, node_color=colors, node_size=size)
        plt.show()
        plt.savefig(title+'.png', fontsize=30)

    def communities_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.g)
        for node in self.g.nodes():
            type_node = self.g.nodes[node]['type']
            for n in self.g.neighbors(node):
                if self.hasSameType(n, type_node) and type_node != 0:
                    G.add_edge(node, n)
        return nx.number_connected_components(G)

    def isEmpty(self, node):
        return self.g.nodes[node]['type'] == 0


    def hasSameType(self, node, type):
        return self.g.nodes[node]['type'] == type


    def ratio_happiness(self, node):
        """ compute ratio of happiness of 'node' """
        empty=[]; same=[]
        type_node = self.g.nodes[node]['type']
        tot=0
        for n in self.g.neighbors(node):
            tot += 1
            if self.isEmpty(n):
                empty.append(n)
            if self.hasSameType(n, type_node):
                same.append(n)
        tot_empty = len(empty)
        tot_same = len(same)
        if tot_empty == tot:
            if self.isEmpty(node):
                return 1.0
            return 0.0
        return tot_same / (tot-tot_empty)

    def isHappy(self, node):
        """ check if 'node' is happy by comparing its ratio of happiness to its preference """
        preference = self.g.nodes[node]['preference']
        return self.ratio_happiness(node) >= preference

    def meanHapinessSociety(self):
        """ return the ratio of happiness averaged over all nodes of the network """
        sum = 0
        for n in self.g.nodes(): sum += self.ratio_happiness(n)
        return sum / len(self.g)

    def iteration(self):
        """ run one iteration of Schelling's model which in unhappy nodes
            are affected to empty positions within the network in a random manner
        """
        #1. find all unhappy anf empty nodes within the network
        empty=[]; unhappy=[]
        for n in self.g.nodes():
            if self.isEmpty(n):
                empty.append(n)
            if not self.isHappy(n):
                unhappy.append(n)
        self.total_unhappy = len(unhappy)
        np.random.shuffle(unhappy)
        np.random.shuffle(empty)

        #2. swap nodes (and their attributes type and preferences)
        total_swap = min(self.total_unhappy, self.total_empty)
        for i in range(total_swap):
            tmp = self.g.nodes[unhappy[i]]['type']
            self.g.nodes[unhappy[i]]['type'] = self.g.nodes[empty[i]]['type']
            self.g.nodes[empty[i]]['type'] = tmp


    def algo(self, threshold_hapiness=0.01, threshold_convergence=100):
        """ run algorithm of Schelling's model  """
        iter=0; iter_conv=0; previous=0
        while self.total_unhappy / self.total_nodes > threshold_hapiness \
                and iter_conv < threshold_convergence:
            if previous == self.total_unhappy / self.total_nodes:
                iter_conv += 1
            else:
                iter_conv = 0
            iter += 1
            previous=self.total_unhappy / self.total_nodes
            self.iteration()
        return iter


def fig():
    # type_graph='grid' pour Fig 3
    # type_graph='small-world' pour Fig 5
    area = Area(N=50,
                types=[0, 1, 2],
                types_prob=[0.1, 0.45, 0.45],
                preference=0.3,
                type_graph='grid',
                r=2)
    area.draw_graph('Configuration initiale')

    for pref in 0.3, 0.5, 0.7:
        area.init_graph(pref)
        area.algo()
        area.draw_graph('Preference '+str(pref*100)+'%')

def fig2():
    N=50; types=[0, 1, 2]; types_prob=[0.1, 0.45, 0.45]
    area_grid = Area(N, types, types_prob, 0.3,'grid', r=2)
    area_r0 = Area(N, types, types_prob, 0.3,'small-world', r=0)
    area_r2 = Area(N, types, types_prob, 0.3,'small-world', r=2)
    area_r4 = Area(N, types, types_prob, 0.3,'small-world', r=4)
    area_r6 = Area(N, types, types_prob, 0.3,'small-world', r=6)

    g_init_grid = area_grid.get_graph()
    g_init_r0 = area_r0.get_graph()
    g_init_r2 = area_r2.get_graph()
    g_init_r4 = area_r4.get_graph()
    g_init_r6 = area_r6.get_graph()

    satis_grid=[]; satis_r0=[];  satis_r2=[];  satis_r4=[];  satis_r6=[]
    comm_grid=[]; comm_r0=[]; comm_r2=[]; comm_r4=[]; comm_r6=[];
    x = []
    for pref in np.arange(0, 0.60, 0.01):
        x.append(pref)
        print(pref)

        area_grid.init_graph(g_init_grid, pref)
        area_r0.init_graph(g_init_r0, pref)
        area_r2.init_graph(g_init_r2, pref)
        area_r4.init_graph(g_init_r4, pref)
        area_r6.init_graph(g_init_r6, pref)

        area_grid.algo()
        area_r0.algo()
        area_r2.algo()
        area_r4.algo()
        area_r6.algo()

        satis_grid.append( area_grid.meanHapinessSociety() )
        satis_r0.append( area_r0.meanHapinessSociety() )
        satis_r2.append( area_r2.meanHapinessSociety() )
        satis_r4.append( area_r4.meanHapinessSociety() )
        satis_r6.append( area_r6.meanHapinessSociety() )

        comm_grid.append( area_grid.communities_graph() )
        comm_r0.append( area_r0.communities_graph() )
        comm_r2.append( area_r2.communities_graph() )
        comm_r4.append( area_r4.communities_graph() )
        comm_r6.append( area_r6.communities_graph() )



    plt.figure()
    #plt.xlim(0, 0.7)
    plt.plot(x, satis_grid, label = "grille")
    plt.plot(x, satis_r0, label = "petit-monde r = 0")
    plt.plot(x, satis_r2, label = "petit-monde r = 2")
    plt.plot(x, satis_r4, label = "petit-monde r = 4")
    #plt.plot(x, satis_r6, label = "petit-monde r = 6")
    plt.xlabel('ratio de preference')
    plt.ylabel('satisfaction moyenne [%]')
    plt.legend()
    plt.show()
    plt.savefig('satisfaction.png')


    plt.figure()
    #plt.xlim(0, 0.7)
    plt.plot(x, comm_grid, label = "grille")
    plt.plot(x, comm_r0, label = "petit-monde r = 0")
    plt.plot(x, comm_r2, label = "petit-monde r = 2")
    plt.plot(x, comm_r4, label = "petit-monde r = 4")
    #plt.plot(x, comm_r6, label = "petit-monde r = 6")
    plt.xlabel('ratio de preference')
    plt.ylabel('nombre de communautées')
    plt.legend()
    plt.show()
    plt.savefig('communautes.png')
