"""
=============================================================================
SOLVEUR MPVRP-CC AVEC PROGRAMMATION LINÉAIRE (MILP)
Utilise Google OR-Tools pour résoudre le problème de manière optimale
=============================================================================

Installation : pip install ortools

Auteur : Explication complète étape par étape
"""

from ortools.linear_solver import pywraplp
import math
import json
from dataclasses import dataclass
from typing import Dict, List

# =============================================================================
# PARTIE 1 : STRUCTURES DE DONNÉES
# =============================================================================

@dataclass
class Vehicle:
    """Représente un véhicule/camion"""
    id: int
    capacity: int
    initial_product: int
    garage_id: int

@dataclass
class Depot:
    """Représente un dépôt (point de chargement)"""
    id: int
    x: float
    y: float
    products: List[int]  # Liste des produits disponibles

@dataclass
class Garage:
    """Représente un garage (base des véhicules)"""
    id: int
    x: float
    y: float

@dataclass
class Station:
    """Représente une station-service (client)"""
    id: int
    x: float
    y: float
    demands: Dict[int, int]  # {produit_id: quantité}


# =============================================================================
# PARTIE 2 : LECTURE DE L'INSTANCE
# =============================================================================

class MPVRPInstance:
    """Charge et stocke toutes les données du problème"""
    
    def __init__(self, filename):
        # Initialisation des attributs
        self.filename = filename
        self.nb_products = 0
        self.nb_garages = 0
        self.nb_depots = 0
        self.nb_stations = 0
        self.nb_vehicles = 0
        self.transition_costs = []
        self.vehicles = []
        self.depots = []
        self.garages = []
        self.stations = []
        self.distance_matrix = {}
        
        # Charger et traiter les données
        self.parse_instance(filename)
        self.compute_distances()
    
    def parse_instance(self, filename):
        """Lit et parse le fichier .dat"""
        print(f"📂 Lecture du fichier: {filename}")
        
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f 
                    if line.strip() and not line.startswith('#')]
        
        idx = 0
        
        # === LIGNE 1 : Paramètres globaux ===
        params = list(map(int, lines[idx].split()))
        self.nb_products = params[0]
        self.nb_garages = params[1]
        self.nb_depots = params[2]
        self.nb_stations = params[3]
        self.nb_vehicles = params[4]
        idx += 1
        
        print(f"  Produits: {self.nb_products}")
        print(f"  Véhicules: {self.nb_vehicles}")
        print(f"  Dépôts: {self.nb_depots}")
        print(f"  Stations: {self.nb_stations}")
        
        # === MATRICE DES COÛTS DE TRANSITION ===
        self.transition_costs = []
        for i in range(self.nb_products):
            row = list(map(float, lines[idx].split()))
            self.transition_costs.append(row)
            idx += 1
        
        # === VÉHICULES ===
        for v in range(self.nb_vehicles):
            parts = list(map(int, lines[idx].split()))
            vehicle = Vehicle(
                id=parts[0],
                capacity=parts[1],
                initial_product=parts[2],
                garage_id=parts[3]
            )
            self.vehicles.append(vehicle)
            idx += 1
        
        # === DÉPÔTS ===
        for d in range(self.nb_depots):
            parts = lines[idx].split()
            depot_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            stocks = list(map(int, parts[3:]))
            
            # Un dépôt propose un produit si son stock > 0
            available_products = [p + 1 for p, stock in enumerate(stocks) 
                                if stock > 0]
            
            depot = Depot(
                id=depot_id,
                x=x,
                y=y,
                products=available_products
            )
            self.depots.append(depot)
            idx += 1
        
        # === GARAGES ===
        for g in range(self.nb_garages):
            parts = lines[idx].split()
            garage = Garage(
                id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2])
            )
            self.garages.append(garage)
            idx += 1
        
        # === STATIONS ===
        for s in range(self.nb_stations):
            parts = lines[idx].split()
            station_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            demands = {}
            
            for p in range(self.nb_products):
                demand = int(parts[3 + p])
                if demand > 0:
                    demands[p + 1] = demand
            
            station = Station(
                id=station_id,
                x=x,
                y=y,
                demands=demands
            )
            self.stations.append(station)
            idx += 1
    
    def compute_distances(self):
        """Calcule toutes les distances euclidiennes"""
        print("📏 Calcul des distances...")
        
        all_nodes = {}
        
        # Collecter tous les nœuds avec leurs coordonnées
        for g in self.garages:
            all_nodes[('G', g.id)] = (g.x, g.y)
        
        for d in self.depots:
            all_nodes[('D', d.id)] = (d.x, d.y)
        
        for s in self.stations:
            all_nodes[('S', s.id)] = (s.x, s.y)
        
        # Calculer toutes les distances
        for (type1, id1), (x1, y1) in all_nodes.items():
            for (type2, id2), (x2, y2) in all_nodes.items():
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                self.distance_matrix[((type1, id1), (type2, id2))] = dist
    
    def get_distance(self, node1, node2):
        """Récupère la distance entre deux nœuds"""
        return self.distance_matrix.get((node1, node2), 0)


# =============================================================================
# PARTIE 3 : SOLVEUR MILP
# =============================================================================

class MPVRPMILPSolver:
    """
    Solveur MILP pour le MPVRP-CC
    Construit et résout le modèle mathématique complet
    """
    
    def __init__(self, instance: MPVRPInstance, time_limit_seconds=300):
        self.instance = instance
        self.time_limit = time_limit_seconds
        
        # Créer le solver
        print("\n🔧 Initialisation du solver SCIP...")
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        
        if not self.solver:
            raise Exception("❌ Solver SCIP non disponible. Installez OR-Tools.")
        
        # Dictionnaires pour stocker les variables
        self.x_vars = {}  # Variables de routage
        self.q_vars = {}  # Variables de quantité livrée
        self.y_vars = {}  # Variables de chargement au dépôt
        self.z_vars = {}  # Variables de changement de produit
    
    def build_model(self):
        """Construit le modèle MILP complet"""
        print("\n" + "="*70)
        print("CONSTRUCTION DU MODÈLE MATHÉMATIQUE")
        print("="*70)
        
        # Estimation du nombre max de mini-routes
        max_mini_routes = 10
        
        # =====================================================================
        # ÉTAPE 1 : CRÉATION DES VARIABLES
        # =====================================================================
        print("\n📊 Étape 1/3 : Création des variables...")
        
        self._create_routing_variables(max_mini_routes)
        self._create_quantity_variables(max_mini_routes)
        self._create_loading_variables(max_mini_routes)
        self._create_changeover_variables(max_mini_routes)
        
        print(f"  ✓ Total de variables créées: {self.solver.NumVariables()}")
        
        # =====================================================================
        # ÉTAPE 2 : DÉFINITION DE LA FONCTION OBJECTIF
        # =====================================================================
        print("\n🎯 Étape 2/3 : Définition de la fonction objectif...")
        
        self._build_objective()
        
        # =====================================================================
        # ÉTAPE 3 : AJOUT DES CONTRAINTES
        # =====================================================================
        print("\n⚖️  Étape 3/3 : Ajout des contraintes...")
        
        self._add_demand_constraints()
        self._add_capacity_constraints(max_mini_routes)
        self._add_visit_constraints(max_mini_routes)
        self._add_flow_constraints(max_mini_routes)
        self._add_changeover_constraints(max_mini_routes)
        
        print(f"  ✓ Total de contraintes ajoutées: {self.solver.NumConstraints()}")
        
        print("\n" + "="*70)
        print("✅ MODÈLE CONSTRUIT AVEC SUCCÈS")
        print("="*70)
    
    def _create_routing_variables(self, max_mini_routes):
        """Crée les variables x de routage"""
        print("  🚚 Variables x (routage)...", end='')
        count = 0
        
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            
            for t in range(max_mini_routes):
                garage_node = ('G', vehicle.garage_id)
                
                # Garage → Dépôt
                for depot in self.instance.depots:
                    depot_node = ('D', depot.id)
                    for p in depot.products:
                        var_name = f'x_k{k}_t{t}_G{vehicle.garage_id}_D{depot.id}_p{p}'
                        self.x_vars[(k, t, garage_node, depot_node, p)] = \
                            self.solver.BoolVar(var_name)
                        count += 1
                
                # Dépôt → Station
                for depot in self.instance.depots:
                    depot_node = ('D', depot.id)
                    for station in self.instance.stations:
                        station_node = ('S', station.id)
                        for p in depot.products:
                            if p in station.demands:
                                var_name = f'x_k{k}_t{t}_D{depot.id}_S{station.id}_p{p}'
                                self.x_vars[(k, t, depot_node, station_node, p)] = \
                                    self.solver.BoolVar(var_name)
                                count += 1
                
                # Station → Station
                for s1 in self.instance.stations:
                    s1_node = ('S', s1.id)
                    for s2 in self.instance.stations:
                        if s1.id != s2.id:
                            s2_node = ('S', s2.id)
                            for p in range(1, self.instance.nb_products + 1):
                                var_name = f'x_k{k}_t{t}_S{s1.id}_S{s2.id}_p{p}'
                                self.x_vars[(k, t, s1_node, s2_node, p)] = \
                                    self.solver.BoolVar(var_name)
                                count += 1
                
                # Station → Garage
                for station in self.instance.stations:
                    station_node = ('S', station.id)
                    for p in range(1, self.instance.nb_products + 1):
                        var_name = f'x_k{k}_t{t}_S{station.id}_G{vehicle.garage_id}_p{p}'
                        self.x_vars[(k, t, station_node, garage_node, p)] = \
                            self.solver.BoolVar(var_name)
                        count += 1
        
        print(f" {count} variables")
    
    def _create_quantity_variables(self, max_mini_routes):
        """Crée les variables q de quantité livrée"""
        print("  📦 Variables q (quantités)...", end='')
        count = 0
        
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            for t in range(max_mini_routes):
                for station in self.instance.stations:
                    for p, demand in station.demands.items():
                        var_name = f'q_k{k}_t{t}_s{station.id}_p{p}'
                        self.q_vars[(k, t, station.id, p)] = \
                            self.solver.NumVar(0, demand, var_name)
                        count += 1
        
        print(f" {count} variables")
    
    def _create_loading_variables(self, max_mini_routes):
        """Crée les variables y de chargement au dépôt"""
        print("  🏭 Variables y (chargement)...", end='')
        count = 0
        
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            for t in range(max_mini_routes):
                for depot in self.instance.depots:
                    for p in depot.products:
                        var_name = f'y_k{k}_t{t}_d{depot.id}_p{p}'
                        self.y_vars[(k, t, depot.id, p)] = \
                            self.solver.BoolVar(var_name)
                        count += 1
        
        print(f" {count} variables")
    
    def _create_changeover_variables(self, max_mini_routes):
        """Crée les variables z de changement de produit"""
        print("  🔄 Variables z (changements)...", end='')
        count = 0
        
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            for t in range(1, max_mini_routes):
                for p1 in range(1, self.instance.nb_products + 1):
                    for p2 in range(1, self.instance.nb_products + 1):
                        if p1 != p2:
                            var_name = f'z_k{k}_t{t}_p{p1}_p{p2}'
                            self.z_vars[(k, t, p1, p2)] = \
                                self.solver.BoolVar(var_name)
                            count += 1
        
        print(f" {count} variables")
    
    def _build_objective(self):
        """Construit la fonction objectif"""
        objective = self.solver.Objective()
        
        # Partie 1 : Coût de distance
        distance_count = 0
        for (k, t, i, j, p), var in self.x_vars.items():
            distance = self.instance.get_distance(i, j)
            objective.SetCoefficient(var, distance)
            distance_count += 1
        
        # Partie 2 : Coût de changement
        changeover_count = 0
        for (k, t, p1, p2), var in self.z_vars.items():
            changeover_cost = self.instance.transition_costs[p1 - 1][p2 - 1]
            objective.SetCoefficient(var, changeover_cost)
            changeover_count += 1
        
        objective.SetMinimization()
        
        print(f"  ✓ Minimiser : (distance × {distance_count} arcs) + " + 
              f"(changement × {changeover_count} cas)")
    
    def _add_demand_constraints(self):
        """Contrainte : Toutes les demandes doivent être satisfaites"""
        print("  📋 Contrainte 1 : Satisfaction des demandes...", end='')
        count = 0
        
        for station in self.instance.stations:
            for p, demand in station.demands.items():
                constraint = self.solver.Constraint(demand, demand)
                
                for vehicle in self.instance.vehicles:
                    k = vehicle.id
                    for t in range(10):
                        if (k, t, station.id, p) in self.q_vars:
                            constraint.SetCoefficient(
                                self.q_vars[(k, t, station.id, p)], 1
                            )
                count += 1
        
        print(f" {count} contraintes")
    
    def _add_capacity_constraints(self, max_mini_routes):
        """Contrainte : Respecter les capacités des véhicules"""
        print("  🚛 Contrainte 2 : Capacités des véhicules...", end='')
        count = 0
        
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            for t in range(max_mini_routes):
                constraint = self.solver.Constraint(
                    -self.solver.infinity(), 
                    vehicle.capacity
                )
                
                for station in self.instance.stations:
                    for p in station.demands.keys():
                        if (k, t, station.id, p) in self.q_vars:
                            constraint.SetCoefficient(
                                self.q_vars[(k, t, station.id, p)], 1
                            )
                count += 1
        
        print(f" {count} contraintes")
    
    def _add_visit_constraints(self, max_mini_routes):
        """Contrainte : Livrer seulement si on visite (Big M)"""
        print("  🎯 Contrainte 3 : Livraison si visite (Big M)...", end='')
        count = 0
        M = 100000  # Big M
        
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            for t in range(max_mini_routes):
                for station in self.instance.stations:
                    s_node = ('S', station.id)
                    
                    for p in station.demands.keys():
                        if (k, t, station.id, p) not in self.q_vars:
                            continue
                        
                        constraint = self.solver.Constraint(
                            -self.solver.infinity(), 0
                        )
                        
                        # q avec coefficient +1
                        constraint.SetCoefficient(
                            self.q_vars[(k, t, station.id, p)], 1
                        )
                        
                        # Arcs entrants avec coefficient -M
                        for depot in self.instance.depots:
                            d_node = ('D', depot.id)
                            if (k, t, d_node, s_node, p) in self.x_vars:
                                constraint.SetCoefficient(
                                    self.x_vars[(k, t, d_node, s_node, p)], -M
                                )
                        
                        for s2 in self.instance.stations:
                            if s2.id != station.id:
                                s2_node = ('S', s2.id)
                                if (k, t, s2_node, s_node, p) in self.x_vars:
                                    constraint.SetCoefficient(
                                        self.x_vars[(k, t, s2_node, s_node, p)], -M
                                    )
                        count += 1
        
        print(f" {count} contraintes")
    
    def _add_flow_constraints(self, max_mini_routes):
        """Contraintes : Départ du garage, retour au garage"""
        print("  🔄 Contrainte 4 : Conservation du flux...", end='')
        count = 0
        
        # Départ du garage
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            garage_node = ('G', vehicle.garage_id)
            
            constraint = self.solver.Constraint(0, 1)
            for depot in self.instance.depots:
                d_node = ('D', depot.id)
                for p in depot.products:
                    if (k, 0, garage_node, d_node, p) in self.x_vars:
                        constraint.SetCoefficient(
                            self.x_vars[(k, 0, garage_node, d_node, p)], 1
                        )
            count += 1
        
        # Retour au garage
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            garage_node = ('G', vehicle.garage_id)
            
            constraint = self.solver.Constraint(0, 1)
            for t in range(max_mini_routes):
                for station in self.instance.stations:
                    s_node = ('S', station.id)
                    for p in range(1, self.instance.nb_products + 1):
                        if (k, t, s_node, garage_node, p) in self.x_vars:
                            constraint.SetCoefficient(
                                self.x_vars[(k, t, s_node, garage_node, p)], 1
                            )
            count += 1
        
        print(f" {count} contraintes")
    
    def _add_changeover_constraints(self, max_mini_routes):
        """Contrainte : Détecter les changements de produit"""
        print("  🔀 Contrainte 5 : Détection des changements...", end='')
        count = 0
        
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            for t in range(1, max_mini_routes):
                for p1 in range(1, self.instance.nb_products + 1):
                    for p2 in range(1, self.instance.nb_products + 1):
                        if p1 == p2:
                            continue
                        
                        if (k, t, p1, p2) not in self.z_vars:
                            continue
                        
                        for depot1 in self.instance.depots:
                            if p1 not in depot1.products:
                                continue
                            
                            for depot2 in self.instance.depots:
                                if p2 not in depot2.products:
                                    continue
                                
                                if ((k, t-1, depot1.id, p1) in self.y_vars and
                                    (k, t, depot2.id, p2) in self.y_vars):
                                    
                                    constraint = self.solver.Constraint(
                                        -self.solver.infinity(), 1
                                    )
                                    
                                    constraint.SetCoefficient(
                                        self.z_vars[(k, t, p1, p2)], -1
                                    )
                                    constraint.SetCoefficient(
                                        self.y_vars[(k, t-1, depot1.id, p1)], 1
                                    )
                                    constraint.SetCoefficient(
                                        self.y_vars[(k, t, depot2.id, p2)], 1
                                    )
                                    count += 1
        
        print(f" {count} contraintes")
    
    def solve(self):
        """Résout le modèle MILP"""
        print("\n" + "="*70)
        print("RÉSOLUTION DU MODÈLE")
        print("="*70)
        print(f"⏱️  Limite de temps: {self.time_limit}s")
        print("🚀 Lancement de la résolution...\n")
        
        self.solver.SetTimeLimit(self.time_limit * 1000)
        status = self.solver.Solve()
        
        print("\n" + "="*70)
        if status == pywraplp.Solver.OPTIMAL:
            print("✅ SOLUTION OPTIMALE TROUVÉE !")
        elif status == pywraplp.Solver.FEASIBLE:
            print("✅ SOLUTION RÉALISABLE TROUVÉE (pas forcément optimale)")
        else:
            print("❌ AUCUNE SOLUTION TROUVÉE")
            print("="*70)
            return None
        
        print("="*70)
        print(f"\n💰 Coût total: {self.solver.Objective().Value():.2f}")
        print(f"⏱️  Temps de résolution: {self.solver.WallTime() / 1000:.2f}s")
        
        return self.extract_solution()
    
    def extract_solution(self):
        """Extrait la solution du modèle résolu"""
        print("\n📤 Extraction de la solution...")
        
        solution = {
            'total_cost': self.solver.Objective().Value(),
            'total_distance': 0,
            'total_changeover': 0,
            'routes': {},
            'detailed_routes': {}
        }
        
        # Calculer les coûts séparément
        for (k, t, i, j, p), var in self.x_vars.items():
            if var.solution_value() > 0.5:
                distance = self.instance.get_distance(i, j)
                solution['total_distance'] += distance
        
        for (k, t, p1, p2), var in self.z_vars.items():
            if var.solution_value() > 0.5:
                cost = self.instance.transition_costs[p1 - 1][p2 - 1]
                solution['total_changeover'] += cost
        
        # Extraire les routes de manière améliorée
        for vehicle in self.instance.vehicles:
            k = vehicle.id
            solution['routes'][k] = []
            solution['detailed_routes'][k] = []
            
            # Collecter tous les arcs utilisés par ce véhicule
            vehicle_arcs = []
            for (veh_id, t, i, j, p), var in self.x_vars.items():
                if veh_id == k and var.solution_value() > 0.5:
                    vehicle_arcs.append((t, i, j, p))
            
            if not vehicle_arcs:
                continue
            
            # Trier par temps
            vehicle_arcs.sort(key=lambda x: x[0])
            
            # Reconstruire les mini-routes
            current_mini_route = None
            
            for t, i, j, p in vehicle_arcs:
                i_type, i_id = i
                j_type, j_id = j
                
                # Début d'une nouvelle mini-route (départ du dépôt)
                if i_type == 'D':
                    if current_mini_route and current_mini_route['visits']:
                        solution['routes'][k].append(current_mini_route)
                    
                    current_mini_route = {
                        'time': t,
                        'depot_id': i_id,
                        'product_id': p,
                        'visits': []
                    }
                
                # Livraison à une station
                if j_type == 'S' and current_mini_route:
                    station_id = j_id
                    # Vérifier la quantité livrée
                    if (k, t, station_id, p) in self.q_vars:
                        qty = self.q_vars[(k, t, station_id, p)].solution_value()
                        if qty > 0.01:
                            current_mini_route['visits'].append((station_id, int(round(qty))))
                
                # Enregistrer l'arc détaillé
                solution['detailed_routes'][k].append({
                    'time': t,
                    'from': f"{i_type}{i_id}",
                    'to': f"{j_type}{j_id}",
                    'product': p,
                    'distance': self.instance.get_distance(i, j)
                })
            
            # Ajouter la dernière mini-route
            if current_mini_route and current_mini_route['visits']:
                solution['routes'][k].append(current_mini_route)
        
        return solution


# =============================================================================
# PARTIE 4 : PROGRAMME PRINCIPAL
# =============================================================================

def format_solution(solution):
    """Formate joliment la solution pour l'affichage"""
    print("\n" + "="*70)
    print("SOLUTION DÉTAILLÉE")
    print("="*70)
    
    print(f"\n💰 COÛTS:")
    print(f"  Distance totale:      {solution['total_distance']:.2f}")
    print(f"  Coûts de changement:  {solution['total_changeover']:.2f}")
    print(f"  COÛT TOTAL:          {solution['total_cost']:.2f}")
    
    print(f"\n🚛 ROUTES DES VÉHICULES:")
    vehicles_used = 0
    
    for vehicle_id, routes in solution['routes'].items():
        if routes:
            vehicles_used += 1
            print(f"\n  ━━━ Véhicule {vehicle_id} ━━━")
            
            for i, route in enumerate(routes, 1):
                print(f"    📦 Mini-route {i} (temps={route['time']}):")
                print(f"       Dépôt: D{route['depot_id']} | Produit: P{route['product_id']}")
                print(f"       Livraisons:")
                total_delivered = 0
                for station_id, qty in route['visits']:
                    print(f"         → Station S{station_id}: {qty} unités")
                    total_delivered += qty
                print(f"       Total livré: {total_delivered} unités")
    
    if vehicles_used == 0:
        print("\n  ⚠️  Aucun véhicule utilisé détecté dans les mini-routes.")
        print("  Regardons les trajets détaillés...")
        
        for vehicle_id, detailed in solution['detailed_routes'].items():
            if detailed:
                print(f"\n  ━━━ Véhicule {vehicle_id} - Trajets détaillés ━━━")
                for arc in detailed:
                    print(f"    t={arc['time']}: {arc['from']} → {arc['to']} " +
                          f"(produit {arc['product']}, distance: {arc['distance']:.2f})")
    else:
        print(f"\n  ✅ Nombre de véhicules utilisés: {vehicles_used}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("="*70)
    print("SOLVEUR MPVRP-CC AVEC PROGRAMMATION LINÉAIRE EN NOMBRES ENTIERS")
    print("="*70)
    
    # Charger l'instance
    instance_file = "MPVRP_M_036_s52_d4_p6.dat"
    instance = MPVRPInstance(instance_file)
    
    # Créer le solveur et construire le modèle
    solver = MPVRPMILPSolver(instance, time_limit_seconds=1000)
    solver.build_model()
    
    # Résoudre
    solution = solver.solve()
    
    # Afficher la solution
    if solution:
        format_solution(solution)
        
        # Sauvegarder en JSON
        output_file = "solution_milp.json"
        with open(output_file, 'w') as f:
            json.dump(solution, f, indent=2)
        print(f"\n💾 Solution sauvegardée dans: {output_file}")
    
    print("\n✅ TERMINÉ !")