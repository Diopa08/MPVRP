"""
Solveur MPVRP-CC (Multi-Product Vehicle Routing Problem with Changeover Cost)
Utilise OR-Tools CP-SAT pour la résolution
Format d'entrée: fichiers .dat selon la spécification
Format de sortie: fichiers .dat selon la spécification
"""

import os
import math
import time
import uuid
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from ortools.sat.python import cp_model
import sys

# ============================================================================
# STRUCTURES DE DONNÉES POUR LES INSTANCES
# ============================================================================

@dataclass
class Vehicle:
    id: int
    capacity: float
    home_garage: int
    initial_product: int

@dataclass
class Depot:
    id: int
    x: float
    y: float
    stocks: List[float]  # stock pour chaque produit

@dataclass
class Garage:
    id: int
    x: float
    y: float

@dataclass
class Station:
    id: int
    x: float
    y: float
    demands: List[float]  # demande pour chaque produit

@dataclass
class Instance:
    """Représente une instance complète du problème MPVRP-CC"""
    uuid: str
    nb_products: int
    nb_depots: int
    nb_garages: int
    nb_stations: int
    nb_vehicles: int
    transition_cost: List[List[float]]  # matrice carrée nb_products x nb_products
    vehicles: List[Vehicle]
    depots: List[Depot]
    garages: List[Garage]
    stations: List[Station]
    
    def __post_init__(self):
        # Créer des dictionnaires pour un accès rapide
        self.depot_dict = {d.id: d for d in self.depots}
        self.garage_dict = {g.id: g for g in self.garages}
        self.station_dict = {s.id: s for s in self.stations}
        self.vehicle_dict = {v.id: v for v in self.vehicles}

# ============================================================================
# LECTURE DES INSTANCES
# ============================================================================

def read_instance(file_path: str) -> Instance:
    """
    Lit un fichier d'instance au format spécifié.
    
    Format:
    Ligne 1: UUID
    Ligne 2: NbProducts NbDepots NbGarages NbStations NbVehicles
    Matrice de coût de transition (NbProducts lignes, NbProducts valeurs par ligne)
    Véhicules (NbVehicles lignes: ID Capacity HomeGarage InitialProduct)
    Dépôts (NbDepots lignes: ID X Y Stock_P1 Stock_P2 ... Stock_Pp)
    Garages (NbGarages lignes: ID X Y)
    Stations (NbStations lignes: ID X Y Demand_P1 Demand_P2 ... Demand_Pp)
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    
    # Nettoyer les lignes vides et les commentaires
    cleaned_lines = []
    for line in lines:
        if line and not line.startswith('#'):
            # Supprimer les commentaires en ligne
            if '#' in line:
                line = line.split('#')[0].strip()
            cleaned_lines.append(line)
    
    lines = cleaned_lines
    
    # UUID (ligne 1)
    instance_uuid = lines[0]
    
    # Paramètres globaux (ligne 2)
    params = list(map(int, lines[1].split()))
    if len(params) != 5:
        raise ValueError(f"Ligne 2 doit avoir 5 valeurs, trouvé {len(params)}")
    
    nb_products, nb_depots, nb_garages, nb_stations, nb_vehicles = params
    
    # Indice de lecture courant
    idx = 2
    
    # Lire la matrice de coût de transition
    transition_cost = []
    for i in range(nb_products):
        row = list(map(float, lines[idx].split()))
        if len(row) != nb_products:
            raise ValueError(f"La matrice de transition doit être carrée ({nb_products}x{nb_products})")
        transition_cost.append(row)
        idx += 1
    
    # Lire les véhicules
    vehicles = []
    for i in range(nb_vehicles):
        parts = lines[idx].split()
        if len(parts) < 4:
            raise ValueError(f"Ligne véhicule {i+1} doit avoir au moins 4 valeurs")
        
        vehicle_id = int(parts[0])
        capacity = float(parts[1])
        home_garage = int(parts[2])
        initial_product = int(parts[3])
        
        vehicles.append(Vehicle(vehicle_id, capacity, home_garage, initial_product))
        idx += 1
    
    # Lire les dépôts
    depots = []
    for i in range(nb_depots):
        parts = list(map(float, lines[idx].split()))
        if len(parts) < 3 + nb_products:
            raise ValueError(f"Dépôt {i+1} doit avoir {3 + nb_products} valeurs")
        
        depot_id = int(parts[0])
        x = parts[1]
        y = parts[2]
        stocks = parts[3:3 + nb_products]
        
        depots.append(Depot(depot_id, x, y, stocks))
        idx += 1
    
    # Lire les garages
    garages = []
    for i in range(nb_garages):
        parts = list(map(float, lines[idx].split()))
        if len(parts) < 3:
            raise ValueError(f"Garage {i+1} doit avoir 3 valeurs")
        
        garage_id = int(parts[0])
        x = parts[1]
        y = parts[2]
        
        garages.append(Garage(garage_id, x, y))
        idx += 1
    
    # Lire les stations
    stations = []
    for i in range(nb_stations):
        parts = list(map(float, lines[idx].split()))
        if len(parts) < 3 + nb_products:
            raise ValueError(f"Station {i+1} doit avoir {3 + nb_products} valeurs")
        
        station_id = int(parts[0])
        x = parts[1]
        y = parts[2]
        demands = parts[3:3 + nb_products]
        
        stations.append(Station(station_id, x, y, demands))
        idx += 1
    
    return Instance(
        uuid=instance_uuid,
        nb_products=nb_products,
        nb_depots=nb_depots,
        nb_garages=nb_garages,
        nb_stations=nb_stations,
        nb_vehicles=nb_vehicles,
        transition_cost=transition_cost,
        vehicles=vehicles,
        depots=depots,
        garages=garages,
        stations=stations
    )

# ============================================================================
# MODÈLE MPVRP-CC AVEC OR-TOOLS
# ============================================================================

class MPVRPSolver:
    """Solveur pour le problème MPVRP-CC utilisant OR-Tools CP-SAT."""
    
    def __init__(self, instance: Instance, time_limit_seconds: int = 300):
        self.instance = instance
        self.time_limit = time_limit_seconds
        
        # Modèle CP-SAT
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # Configurer le solveur
        self.solver.parameters.max_time_in_seconds = time_limit_seconds
        self.solver.parameters.num_search_workers = 8
        
        # Structures pour le modèle
        self._setup_nodes()
        self._create_variables()
        self._add_constraints()
        self._set_objective()
    
    def _setup_nodes(self):
        """Prépare la liste de tous les nœuds avec leurs types."""
        self.all_nodes = []
        self.node_types = {}  # 'G', 'D', 'S'
        self.node_coords = {}  # (x, y)
        self.node_indices = {}  # nom -> index
        self.reverse_indices = {}  # index -> nom
        
        index = 0
        
        # Ajouter les garages
        for garage in self.instance.garages:
            node_id = f"G{garage.id}"
            self.all_nodes.append(node_id)
            self.node_types[node_id] = 'G'
            self.node_coords[node_id] = (garage.x, garage.y)
            self.node_indices[node_id] = index
            self.reverse_indices[index] = node_id
            index += 1
        
        # Ajouter les dépôts
        for depot in self.instance.depots:
            node_id = f"D{depot.id}"
            self.all_nodes.append(node_id)
            self.node_types[node_id] = 'D'
            self.node_coords[node_id] = (depot.x, depot.y)
            self.node_indices[node_id] = index
            self.reverse_indices[index] = node_id
            index += 1
        
        # Ajouter les stations
        for station in self.instance.stations:
            node_id = f"S{station.id}"
            self.all_nodes.append(node_id)
            self.node_types[node_id] = 'S'
            self.node_coords[node_id] = (station.x, station.y)
            self.node_indices[node_id] = index
            self.reverse_indices[index] = node_id
            index += 1
        
        self.nb_nodes = len(self.all_nodes)
        
        # Pré-calculer les distances
        self.distances = {}
        for i in range(self.nb_nodes):
            for j in range(self.nb_nodes):
                if i != j:
                    node_i = self.reverse_indices[i]
                    node_j = self.reverse_indices[j]
                    x1, y1 = self.node_coords[node_i]
                    x2, y2 = self.node_coords[node_j]
                    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    self.distances[(i, j)] = dist
    
    def _create_variables(self):
        """Crée toutes les variables du modèle."""
        # Variables binaires X_ijk (véhicule k va de i à j)
        self.x_vars = {}
        for i in range(self.nb_nodes):
            for j in range(self.nb_nodes):
                if i != j:
                    for k in range(self.instance.nb_vehicles):
                        var_name = f"X_{i}_{j}_{k}"
                        self.x_vars[(i, j, k)] = self.model.NewBoolVar(var_name)
        
        # Variables de flux f_ijkp (quantité de produit p transportée de i à j par k)
        self.f_vars = {}
        for i in range(self.nb_nodes):
            for j in range(self.nb_nodes):
                if i != j:
                    for k in range(self.instance.nb_vehicles):
                        vehicle_capacity = self.instance.vehicles[k].capacity
                        for p in range(self.instance.nb_products):
                            var_name = f"f_{i}_{j}_{k}_{p}"
                            self.f_vars[(i, j, k, p)] = self.model.NewIntVar(
                                0, int(vehicle_capacity), var_name)
        
        # Variables de changement de produit t_p1p2k
        self.t_vars = {}
        for p1 in range(self.instance.nb_products):
            for p2 in range(self.instance.nb_products):
                if p1 != p2:  # Pas de coût pour rester sur le même produit
                    for k in range(self.instance.nb_vehicles):
                        var_name = f"t_{p1}_{p2}_{k}"
                        self.t_vars[(p1, p2, k)] = self.model.NewBoolVar(var_name)
        
        # Variables pour le produit courant dans chaque véhicule à chaque nœud
        self.current_product_vars = {}
        for k in range(self.instance.nb_vehicles):
            for i in range(self.nb_nodes):
                var_name = f"current_prod_{k}_{i}"
                self.current_product_vars[(k, i)] = self.model.NewIntVar(
                    0, self.instance.nb_products - 1, var_name)
    
    def _add_constraints(self):
        """Ajoute toutes les contraintes du modèle."""
        # 1. Contrainte de continuité du trafic
        for k in range(self.instance.nb_vehicles):
            for i in range(self.nb_nodes):
                node_type = self.node_types[self.reverse_indices[i]]
                if node_type in ['S', 'D']:  # Pour les stations et dépôts
                    incoming = []
                    outgoing = []
                    
                    for j in range(self.nb_nodes):
                        if i != j:
                            incoming.append(self.x_vars[(j, i, k)])
                            outgoing.append(self.x_vars[(i, j, k)])
                    
                    if incoming:  # S'il y a des arcs possibles
                        self.model.Add(sum(incoming) == sum(outgoing))
        
        # 2. Livraisons obligatoires (chaque station doit être visitée au moins une fois)
        station_indices = [i for i in range(self.nb_nodes) 
                          if self.node_types[self.reverse_indices[i]] == 'S']
        
        for station_idx in station_indices:
            incoming_arcs = []
            for k in range(self.instance.nb_vehicles):
                for i in range(self.nb_nodes):
                    if i != station_idx:
                        incoming_arcs.append(self.x_vars[(i, station_idx, k)])
            
            if incoming_arcs:
                self.model.Add(sum(incoming_arcs) >= 1)
        
        # 3. Satisfaction de la demande
        for station in self.instance.stations:
            station_node = f"S{station.id}"
            if station_node in self.node_indices:
                station_idx = self.node_indices[station_node]
                
                for p in range(self.instance.nb_products):
                    if station.demands[p] > 0:
                        # Flux entrant - flux sortant = demande
                        incoming_flow = []
                        outgoing_flow = []
                        
                        for k in range(self.instance.nb_vehicles):
                            for i in range(self.nb_nodes):
                                if i != station_idx:
                                    incoming_flow.append(self.f_vars[(i, station_idx, k, p)])
                                    outgoing_flow.append(self.f_vars[(station_idx, i, k, p)])
                        
                        if incoming_flow:
                            demand_value = int(station.demands[p])
                            self.model.Add(sum(incoming_flow) - sum(outgoing_flow) == demand_value)
        
        # 4. Contraintes de flux pour les garages
        for k, vehicle in enumerate(self.instance.vehicles):
            garage_node = f"G{vehicle.home_garage}"
            if garage_node in self.node_indices:
                garage_idx = self.node_indices[garage_node]
                
                # Flux sortant du garage (vers les dépôts)
                outgoing_from_garage = []
                for j in range(self.nb_nodes):
                    if j != garage_idx and self.node_types[self.reverse_indices[j]] == 'D':
                        outgoing_from_garage.append(self.x_vars[(garage_idx, j, k)])
                
                if outgoing_from_garage:
                    self.model.Add(sum(outgoing_from_garage) <= 1)
                
                # Flux entrant au garage (depuis les stations)
                incoming_to_garage = []
                for i in range(self.nb_nodes):
                    if i != garage_idx and self.node_types[self.reverse_indices[i]] == 'S':
                        incoming_to_garage.append(self.x_vars[(i, garage_idx, k)])
                
                if incoming_to_garage:
                    self.model.Add(sum(incoming_to_garage) <= 1)
                
                # Nombre de sorties = nombre d'entrées
                if outgoing_from_garage and incoming_to_garage:
                    self.model.Add(sum(outgoing_from_garage) == sum(incoming_to_garage))
        
        # 5. Contraintes de capacité
        for k, vehicle in enumerate(self.instance.vehicles):
            vehicle_capacity = vehicle.capacity
            
            for i in range(self.nb_nodes):
                for j in range(self.nb_nodes):
                    if i != j:
                        # La somme des produits transportés sur cet arc doit être <= capacité
                        flow_vars = [self.f_vars[(i, j, k, p)] for p in range(self.instance.nb_products)]
                        if flow_vars:
                            self.model.Add(sum(flow_vars) <= int(vehicle_capacity) * self.x_vars[(i, j, k)])
        
        # 6. Contraintes liées aux dépôts
        depot_indices = [i for i in range(self.nb_nodes) 
                        if self.node_types[self.reverse_indices[i]] == 'D']
        
        for depot_idx in depot_indices:
            for k in range(self.instance.nb_vehicles):
                for p in range(self.instance.nb_products):
                    # Flux sortant >= flux entrant (on charge au dépôt)
                    incoming_flow = []
                    outgoing_flow = []
                    
                    for i in range(self.nb_nodes):
                        if i != depot_idx:
                            incoming_flow.append(self.f_vars[(i, depot_idx, k, p)])
                            outgoing_flow.append(self.f_vars[(depot_idx, i, k, p)])
                    
                    if incoming_flow and outgoing_flow:
                        self.model.Add(sum(outgoing_flow) >= sum(incoming_flow))
        
        # 7. Contraintes de trafic interdites
        # Pas de garage -> station
        garage_indices = [i for i in range(self.nb_nodes) 
                         if self.node_types[self.reverse_indices[i]] == 'G']
        station_indices = [i for i in range(self.nb_nodes) 
                          if self.node_types[self.reverse_indices[i]] == 'S']
        
        for garage_idx in garage_indices:
            for station_idx in station_indices:
                for k in range(self.instance.nb_vehicles):
                    self.model.Add(self.x_vars[(garage_idx, station_idx, k)] == 0)
        
        # Pas de dépôt -> garage
        depot_indices = [i for i in range(self.nb_nodes) 
                        if self.node_types[self.reverse_indices[i]] == 'D']
        
        for depot_idx in depot_indices:
            for garage_idx in garage_indices:
                for k in range(self.instance.nb_vehicles):
                    self.model.Add(self.x_vars[(depot_idx, garage_idx, k)] == 0)
        
        # Pas de dépôt -> dépôt
        for depot1_idx in depot_indices:
            for depot2_idx in depot_indices:
                if depot1_idx != depot2_idx:
                    for k in range(self.instance.nb_vehicles):
                        self.model.Add(self.x_vars[(depot1_idx, depot2_idx, k)] == 0)
        
        # 8. Contraintes sur les changements de produit
        # Un véhicule ne peut transporter qu'un seul produit à la fois
        for k in range(self.instance.nb_vehicles):
            for i in range(self.nb_nodes):
                for j in range(self.nb_nodes):
                    if i != j:
                        # Variables indicatrices pour chaque produit sur cet arc
                        product_vars = []
                        for p in range(self.instance.nb_products):
                            # Variable binaire indiquant si le produit p est transporté sur cet arc
                            prod_var = self.model.NewBoolVar(f"prod_{i}_{j}_{k}_{p}")
                            # Si f_ijkp > 0, alors prod_var = 1
                            self.model.Add(self.f_vars[(i, j, k, p)] > 0).OnlyEnforceIf(prod_var)
                            self.model.Add(self.f_vars[(i, j, k, p)] == 0).OnlyEnforceIf(prod_var.Not())
                            product_vars.append(prod_var)
                        
                        # Au plus un produit transporté sur cet arc
                        if product_vars:
                            self.model.Add(sum(product_vars) <= 1)
    
    def _set_objective(self):
        """Définit la fonction objectif à minimiser."""
        # Terme de distance
        distance_terms = []
        for i in range(self.nb_nodes):
            for j in range(self.nb_nodes):
                if i != j:
                    for k in range(self.instance.nb_vehicles):
                        if (i, j) in self.distances:
                            # Multiplier par 100 pour éviter les problèmes de précision avec les flottants
                            coeff = int(self.distances[(i, j)] * 100)
                            distance_terms.append(coeff * self.x_vars[(i, j, k)])
        
        # Terme de coût de transition
        transition_terms = []
        for p1 in range(self.instance.nb_products):
            for p2 in range(self.instance.nb_products):
                if p1 != p2:
                    cost = self.instance.transition_cost[p1][p2]
                    for k in range(self.instance.nb_vehicles):
                        coeff = int(cost * 100)
                        transition_terms.append(coeff * self.t_vars[(p1, p2, k)])
        
        # Minimiser la somme
        if distance_terms or transition_terms:
            self.model.Minimize(sum(distance_terms) + sum(transition_terms))
    
    def solve(self):
        """Résout le modèle et retourne une solution."""
        print(f"Début de la résolution...")
        print(f"Nombre de nœuds: {self.nb_nodes}")
        print(f"Nombre de véhicules: {self.instance.nb_vehicles}")
        print(f"Limite de temps: {self.time_limit} secondes")
        
        start_time = time.time()
        status = self.solver.Solve(self.model)
        solving_time = time.time() - start_time
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            status_str = "OPTIMAL" if status == cp_model.OPTIMAL else "FAISABLE"
            print(f"Solution {status_str} trouvée en {solving_time:.2f} secondes")
            
            # Extraire la solution
            solution = self._extract_solution(solving_time)
            return solution
        else:
            print("Aucune solution trouvée")
            return None
    
    def _extract_solution(self, solving_time):
        """Extrait la solution du modèle résolu."""
        from collections import defaultdict
        
        # Structure pour stocker les routes
        class RouteInfo:
            def __init__(self, vehicle_id):
                self.vehicle_id = vehicle_id
                self.arcs = []  # Liste de (from, to, product, quantity)
                self.total_distance = 0.0
                self.product_changes = 0
        
        # Reconstruire les routes pour chaque véhicule
        routes_by_vehicle = {}
        
        for k, vehicle in enumerate(self.instance.vehicles):
            route = RouteInfo(vehicle.id)
            
            # Trouver le départ du véhicule (garage -> dépôt)
            garage_node = f"G{vehicle.home_garage}"
            if garage_node in self.node_indices:
                garage_idx = self.node_indices[garage_node]
                
                # Chercher le premier arc sortant du garage
                for j in range(self.nb_nodes):
                    if self.solver.Value(self.x_vars[(garage_idx, j, k)]) == 1:
                        # Suivre la route
                        current_node = j
                        prev_node = garage_idx
                        visited = set()
                        
                        while current_node != garage_idx and current_node not in visited:
                            visited.add(current_node)
                            
                            # Trouver le produit transporté sur cet arc
                            product_on_arc = None
                            quantity_on_arc = 0
                            
                            for p in range(self.instance.nb_products):
                                qty = self.solver.Value(self.f_vars[(prev_node, current_node, k, p)])
                                if qty > 0:
                                    product_on_arc = p
                                    quantity_on_arc = qty
                                    break
                            
                            if product_on_arc is not None:
                                from_node = self.reverse_indices[prev_node]
                                to_node = self.reverse_indices[current_node]
                                distance = self.distances.get((prev_node, current_node), 0)
                                
                                route.arcs.append((from_node, to_node, product_on_arc, quantity_on_arc))
                                route.total_distance += distance
                            
                            # Trouver le prochain nœud
                            next_node = None
                            for n in range(self.nb_nodes):
                                if n != current_node and self.solver.Value(self.x_vars[(current_node, n, k)]) == 1:
                                    next_node = n
                                    break
                            
                            if next_node is None:
                                break
                            
                            prev_node = current_node
                            current_node = next_node
            
            # Compter les changements de produit
            if len(route.arcs) > 1:
                current_product = route.arcs[0][2]  # produit du premier arc
                for i in range(1, len(route.arcs)):
                    if route.arcs[i][2] != current_product:
                        route.product_changes += 1
                        current_product = route.arcs[i][2]
            
            routes_by_vehicle[vehicle.id] = route
        
        # Calculer les métriques globales
        total_distance = sum(r.total_distance for r in routes_by_vehicle.values())
        total_product_changes = sum(r.product_changes for r in routes_by_vehicle.values())
        
        # Calculer le coût total de transition (approximatif)
        total_transition_cost = 0
        for (p1, p2, k), var in self.t_vars.items():
            if self.solver.Value(var) == 1:
                total_transition_cost += self.instance.transition_cost[p1][p2]
        
        # Créer l'objet solution
        solution = {
            'routes': routes_by_vehicle,
            'total_distance': total_distance,
            'total_transition_cost': total_transition_cost,
            'total_product_changes': total_product_changes,
            'solving_time': solving_time,
            'status': 'OPTIMAL' if self.solver.StatusName() == 'OPTIMAL' else 'FEASIBLE'
        }
        
        return solution

# ============================================================================
# ÉCRITURE DES SOLUTIONS
# ============================================================================

def write_solution_file(instance, solution, output_path):
    """
    Écrit un fichier de solution au format spécifié.
    
    Format:
    Pour chaque véhicule utilisé:
      Ligne 1: ID : Garage - Dépôt [Qty] - Station (Qty) - ... - Garage
      Ligne 2: séquence des produits et coûts
      Ligne vide
    
    Puis 6 lignes de métriques:
      1. Nombre de véhicules utilisés
      2. Nombre total de changements de produit
      3. Coût total de transition
      4. Distance totale
      5. Modèle du processeur
      6. Temps de résolution
    """
    routes = solution['routes']
    
    with open(output_path, 'w') as f:
        # Écrire les routes pour chaque véhicule
        for vehicle_id, route in sorted(routes.items()):
            if not route.arcs:
                continue  # Véhicule non utilisé
            
            # Ligne 1: séquence de visites
            line1_parts = [f"{vehicle_id}"]
            
            # Reconstruire la séquence complète du trajet
            sequence = []
            
            # Ajouter le garage de départ
            vehicle_obj = instance.vehicle_dict[vehicle_id]
            sequence.append(str(vehicle_obj.home_garage))
            
            # Parcourir les arcs et construire la séquence
            for from_node, to_node, product, quantity in route.arcs:
                node_type = from_node[0]  # Premier caractère: G, D ou S
                node_id = from_node[1:]   # Le reste est l'ID
                
                if node_type == 'D' and to_node[0] == 'S':
                    # Chargement au dépôt
                    sequence.append(f"{node_id}[{int(quantity)}]")
                
                if to_node[0] == 'S':
                    # Livraison à la station
                    station_id = to_node[1:]
                    sequence.append(f"{station_id}({int(quantity)})")
            
            # Ajouter le garage de retour
            sequence.append(str(vehicle_obj.home_garage))
            
            # Joindre la séquence
            line1 = f"{vehicle_id} " + " - ".join(sequence)
            f.write(line1 + "\n")
            
            # Ligne 2: séquence des produits
            line2_parts = []
            for _, _, product, _ in route.arcs:
                line2_parts.append(f"{product}(0,0)")  # Les coûts sont dans les métriques
            
            line2 = " ".join(line2_parts)
            f.write(line2 + "\n\n")
        
        # Écrire les métriques
        vehicles_used = len([r for r in routes.values() if r.arcs])
        f.write(f"{vehicles_used}\n")
        f.write(f"{solution['total_product_changes']}\n")
        f.write(f"{solution['total_transition_cost']:.2f}\n")
        f.write(f"{solution['total_distance']:.2f}\n")
        f.write("Intel Core i7-10700K\n")  # À adapter selon votre configuration
        f.write(f"{solution['solving_time']:.3f}\n")

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def solve_instance(instance_file_path, output_dir="solutions", time_limit=300):
    """
    Résout une instance MPVRP-CC et génère un fichier de solution.
    
    Args:
        instance_file_path: Chemin vers le fichier d'instance .dat
        output_dir: Répertoire de sortie pour les solutions
        time_limit: Limite de temps en secondes
    """
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Lire l'instance
    print(f"Lecture de l'instance: {instance_file_path}")
    try:
        instance = read_instance(instance_file_path)
        print(f"Instance chargée: {instance.uuid}")
        print(f"  Produits: {instance.nb_products}, Dépôts: {instance.nb_depots}")
        print(f"  Garages: {instance.nb_garages}, Stations: {instance.nb_stations}")
        print(f"  Véhicules: {instance.nb_vehicles}")
    except Exception as e:
        print(f"Erreur lors de la lecture de l'instance: {e}")
        return None
    
    # Résoudre le problème
    solver = MPVRPSolver(instance, time_limit)
    solution = solver.solve()
    
    if solution:
        # Générer le nom du fichier de sortie
        base_name = os.path.basename(instance_file_path)
        if base_name.endswith('.dat'):
            base_name = base_name[:-4]
        
        solution_file = os.path.join(output_dir, f"Sol_{base_name}.dat")
        
        # Écrire la solution
        write_solution_file(instance, solution, solution_file)
        print(f"Solution écrite dans: {solution_file}")
        
        # Afficher un résumé
        print("\n=== RÉSUMÉ DE LA SOLUTION ===")
        print(f"Véhicules utilisés: {len([r for r in solution['routes'].values() if r.arcs])}")
        print(f"Distance totale: {solution['total_distance']:.2f}")
        print(f"Changements de produit: {solution['total_product_changes']}")
        print(f"Coût de transition: {solution['total_transition_cost']:.2f}")
        print(f"Coût total (distance + transition): {solution['total_distance'] + solution['total_transition_cost']:.2f}")
        print(f"Temps de résolution: {solution['solving_time']:.2f}s")
        
        return solution
    else:
        print("Aucune solution trouvée pour cette instance")
        return None

def batch_solve(instance_dir, output_dir="solutions", time_limit=300):
    """
    Résout toutes les instances dans un répertoire.
    
    Args:
        instance_dir: Répertoire contenant les fichiers d'instance .dat
        output_dir: Répertoire de sortie pour les solutions
        time_limit: Limite de temps par instance (secondes)
    """
    # Lister tous les fichiers .dat
    instance_files = []
    for file in os.listdir(instance_dir):
        if file.endswith('.dat'):
            instance_files.append(os.path.join(instance_dir, file))
    
    if not instance_files:
        print(f"Aucun fichier .dat trouvé dans {instance_dir}")
        return
    
    print(f"Trouvé {len(instance_files)} instance(s) à résoudre")
    
    # Résoudre chaque instance
    solutions = []
    for i, instance_file in enumerate(instance_files, 1):
        print(f"\n[{i}/{len(instance_files)}] Résolution de {os.path.basename(instance_file)}")
        solution = solve_instance(instance_file, output_dir, time_limit)
        if solution:
            solutions.append((instance_file, solution))
    
    # Afficher un rapport final
    if solutions:
        print("\n" + "="*60)
        print("RAPPORT FINAL")
        print("="*60)
        
        total_distance = 0
        total_transition_cost = 0
        total_time = 0
        
        for instance_file, solution in solutions:
            filename = os.path.basename(instance_file)
            print(f"\n{filename}:")
            print(f"  Véhicules: {len([r for r in solution['routes'].values() if r.arcs])}")
            print(f"  Distance: {solution['total_distance']:.2f}")
            print(f"  Transition: {solution['total_transition_cost']:.2f}")
            print(f"  Temps: {solution['solving_time']:.2f}s")
            
            total_distance += solution['total_distance']
            total_transition_cost += solution['total_transition_cost']
            total_time += solution['solving_time']
        
        print("\n" + "="*60)
        print(f"TOTAUX:")
        print(f"  Distance totale: {total_distance:.2f}")
        print(f"  Coût de transition total: {total_transition_cost:.2f}")
        print(f"  Coût total: {total_distance + total_transition_cost:.2f}")
        print(f"  Temps total de résolution: {total_time:.2f}s")
        print("="*60)

# ============================================================================
# INTERFACE UTILISATEUR
# ============================================================================

def main():
    """Fonction principale avec interface en ligne de commande."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Solveur MPVRP-CC (Multi-Product Vehicle Routing Problem with Changeover Cost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s instance.dat                     # Résoudre une instance
  %(prog)s instances/ --batch               # Résoudre toutes les instances d'un répertoire
  %(prog)s instance.dat --time-limit 600    # Résoudre avec une limite de 10 minutes
        """
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        help="Fichier d'instance .dat ou répertoire contenant des instances"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="solutions",
        help="Répertoire de sortie pour les solutions (défaut: solutions)"
    )
    
    parser.add_argument(
        "--time-limit", "-t",
        type=int,
        default=300,
        help="Limite de temps par instance en secondes (défaut: 300)"
    )
    
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Traiter tous les fichiers .dat du répertoire d'entrée"
    )
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        return
    
    if args.batch:
        # Mode batch: traiter un répertoire
        if not os.path.isdir(args.input):
            print(f"Erreur: {args.input} n'est pas un répertoire valide")
            return
        
        batch_solve(args.input, args.output_dir, args.time_limit)
    else:
        # Mode single: traiter un fichier
        if not os.path.isfile(args.input):
            print(f"Erreur: {args.input} n'est pas un fichier valide")
            return
        
        solve_instance(args.input, args.output_dir, args.time_limit)

if __name__ == "__main__":
    main()