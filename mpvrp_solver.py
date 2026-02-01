import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Vehicle:
    """Représente un camion"""
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
    products: List[int]  # Liste des produits disponibles avec stock
    
@dataclass
class Garage:
    """Représente un garage"""
    id: int
    x: float
    y: float
    
@dataclass
class Station:
    """Représente une station-service (client)"""
    id: int
    x: float
    y: float
    demands: Dict[int, int]  # {product_id: quantity}
    
class MPVRPInstance:
    """Classe pour stocker une instance du problème"""
    
    def __init__(self, filename):
        self.filename = filename
        self.uuid = ""
        self.nb_products = 0
        self.nb_garages = 0
        self.nb_depots = 0
        self.nb_stations = 0
        self.nb_vehicles = 0
        self.transition_costs = []  # Matrice des coûts de changement
        self.distance_matrix = {}  # Matrice des distances
        self.vehicles = []
        self.depots = []
        self.garages = []
        self.stations = []
        
        self.parse_instance(filename)
        self.compute_distance_matrix()
        
    def parse_instance(self, filename):
        """Lit et parse le fichier d'instance"""
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        idx = 0
        
        # Ligne 1: Paramètres globaux
        params = list(map(int, lines[idx].split()))
        self.nb_products = params[0]
        self.nb_garages = params[1]
        self.nb_depots = params[2]
        self.nb_stations = params[3]
        self.nb_vehicles = params[4]
        idx += 1
        
        # Matrice des coûts de transition (nb_products x nb_products)
        self.transition_costs = []
        for i in range(self.nb_products):
            row = list(map(float, lines[idx].split()))
            self.transition_costs.append(row)
            idx += 1
        
        # Véhicules
        for v in range(self.nb_vehicles):
            parts = list(map(int, lines[idx].split()))
            # Format: VehicleID Capacity InitialProduct GarageID
            vehicle = Vehicle(
                id=parts[0],
                capacity=parts[1],
                initial_product=parts[2],
                garage_id=parts[3]
            )
            self.vehicles.append(vehicle)
            idx += 1
        
        # Dépôts
        for d in range(self.nb_depots):
            parts = lines[idx].split()
            depot_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            
            # Les stocks pour chaque produit
            stocks = list(map(int, parts[3:]))
            
            # Un dépôt propose un produit si son stock est > 0
            available_products = [p + 1 for p, stock in enumerate(stocks) if stock > 0]
            
            depot = Depot(
                id=depot_id,
                x=x,
                y=y,
                products=available_products
            )
            self.depots.append(depot)
            idx += 1
            
            print(f"  Dépôt {depot_id}: produits disponibles = {available_products}, stocks = {stocks}")
        
        # Garages
        for g in range(self.nb_garages):
            parts = lines[idx].split()
            garage = Garage(
                id=int(parts[0]),
                x=float(parts[1]),
                y=float(parts[2])
            )
            self.garages.append(garage)
            idx += 1
        
        # Stations
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
            
            station = Station(id=station_id, x=x, y=y, demands=demands)
            self.stations.append(station)
            idx += 1
    
    def distance(self, x1, y1, x2, y2):
        """Calcule la distance euclidienne entre deux points"""
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def compute_distance_matrix(self):
        """Précalcule toutes les distances entre sites"""
        all_locations = {}
        
        # Garages: ('G', id)
        for g in self.garages:
            all_locations[('G', g.id)] = (g.x, g.y)
        
        # Dépôts: ('D', id)
        for d in self.depots:
            all_locations[('D', d.id)] = (d.x, d.y)
        
        # Stations: ('S', id)
        for s in self.stations:
            all_locations[('S', s.id)] = (s.x, s.y)
        
        # Calcul de toutes les distances
        for loc1, coords1 in all_locations.items():
            for loc2, coords2 in all_locations.items():
                dist = self.distance(coords1[0], coords1[1], coords2[0], coords2[1])
                self.distance_matrix[(loc1, loc2)] = dist
    
    def get_distance(self, from_type, from_id, to_type, to_id):
        """Récupère la distance entre deux sites"""
        return self.distance_matrix.get(((from_type, from_id), (to_type, to_id)), 0)

class Solution:
    """Représente une solution au problème"""
    
    def __init__(self, instance):
        self.instance = instance
        self.routes = {v.id: [] for v in instance.vehicles}  # {vehicle_id: [mini-routes]}
        self.total_distance = 0
        self.total_changeover_cost = 0
        self.total_cost = 0
        
    def add_mini_route(self, vehicle_id, depot_id, product_id, stations_visits):
        """
        Ajoute une mini-tournée à un véhicule
        stations_visits: [(station_id, quantity_delivered), ...]
        """
        mini_route = {
            'depot_id': depot_id,
            'product_id': product_id,
            'visits': stations_visits
        }
        self.routes[vehicle_id].append(mini_route)
    
    def calculate_cost(self):
        """Calcule le coût total de la solution"""
        total_distance = 0
        total_changeover = 0
        
        for vehicle in self.instance.vehicles:
            v_id = vehicle.id
            current_product = vehicle.initial_product
            current_location = ('G', vehicle.garage_id)  # Départ du garage
            
            for mini_route in self.routes[v_id]:
                depot_id = mini_route['depot_id']
                product_id = mini_route['product_id']
                visits = mini_route['visits']
                
                # Changement de produit au dépôt si nécessaire
                if current_product != product_id:
                    changeover_cost = self.instance.transition_costs[current_product - 1][product_id - 1]
                    total_changeover += changeover_cost
                    current_product = product_id
                
                # Distance vers le dépôt
                total_distance += self.instance.get_distance(
                    current_location[0], current_location[1],
                    'D', depot_id
                )
                current_location = ('D', depot_id)
                
                # Livraisons
                for station_id, qty in visits:
                    total_distance += self.instance.get_distance(
                        current_location[0], current_location[1],
                        'S', station_id
                    )
                    current_location = ('S', station_id)
            
            # Retour au garage
            total_distance += self.instance.get_distance(
                current_location[0], current_location[1],
                'G', vehicle.garage_id
            )
        
        self.total_distance = total_distance
        self.total_changeover_cost = total_changeover
        self.total_cost = total_distance + total_changeover
        
        return self.total_cost

class MPVRPSolver:
    """Solveur pour le MPVRP-CC"""
    
    def __init__(self, instance):
        self.instance = instance
        
    def solve(self):
        """Résout le problème avec une heuristique constructive"""
        solution = Solution(self.instance)
        
        # Créer une copie des demandes non satisfaites
        unmet_demands = {}
        for station in self.instance.stations:
            for product_id, quantity in station.demands.items():
                key = (station.id, product_id)
                unmet_demands[key] = quantity
        
        print(f"\n🔍 Debug - Demandes initiales:")
        for key, qty in unmet_demands.items():
            print(f"  Station {key[0]}, Produit {key[1]}: {qty}")
        
        # Pour chaque véhicule, construire des mini-routes jusqu'à ce que tout soit livré
        vehicle_idx = 0
        iteration = 0
        max_iterations = 1000
        
        while any(qty > 0 for qty in unmet_demands.values()) and iteration < max_iterations:
            iteration += 1
            
            # Sélectionner un véhicule (rotation circulaire)
            vehicle = self.instance.vehicles[vehicle_idx % len(self.instance.vehicles)]
            vehicle_idx += 1
            
            # Déterminer quel produit livrer
            # Priorité au produit actuel du véhicule s'il y a des demandes
            current_product = vehicle.initial_product
            if len(solution.routes[vehicle.id]) > 0:
                current_product = solution.routes[vehicle.id][-1]['product_id']
            
            # Trouver les demandes non satisfaites
            products_needed = set()
            for (station_id, product_id), qty in unmet_demands.items():
                if qty > 0:
                    products_needed.add(product_id)
            
            if not products_needed:
                break
            
            # Choisir le produit (préférence au produit actuel pour éviter changements)
            chosen_product = current_product if current_product in products_needed else min(products_needed)
            
            # Trouver un dépôt qui a ce produit
            depot_id = None
            for depot in self.instance.depots:
                if chosen_product in depot.products:
                    depot_id = depot.id
                    break
            
            if depot_id is None:
                print(f" Aucun dépôt ne propose le produit {chosen_product}")
                break
            
            # Construire une mini-route pour ce produit
            mini_route_visits = []
            remaining_capacity = vehicle.capacity
            
            # Trouver toutes les stations qui ont besoin de ce produit
            stations_needing_product = [
                (station_id, qty)
                for (station_id, product_id), qty in unmet_demands.items()
                if product_id == chosen_product and qty > 0
            ]
            
            # Partir du dépôt et faire du plus proche voisin
            current_loc = ('D', depot_id)
            
            while remaining_capacity > 0 and stations_needing_product:
                # Trouver la station la plus proche
                best_station = None
                best_distance = float('inf')
                
                for station_id, qty in stations_needing_product:
                    dist = self.instance.get_distance(
                        current_loc[0], current_loc[1],
                        'S', station_id
                    )
                    
                    if dist < best_distance:
                        best_distance = dist
                        best_station = (station_id, qty)
                
                if best_station is None:
                    break
                
                station_id, qty = best_station
                
                # Livrer autant que possible
                delivered = min(qty, remaining_capacity)
                mini_route_visits.append((station_id, delivered))
                remaining_capacity -= delivered
                
                # Mettre à jour les demandes non satisfaites
                key = (station_id, chosen_product)
                unmet_demands[key] -= delivered
                
                # Mettre à jour la liste des stations ayant besoin
                stations_needing_product = [
                    (s, q - (delivered if s == station_id else 0))
                    for s, q in stations_needing_product
                ]
                stations_needing_product = [(s, q) for s, q in stations_needing_product if q > 0]
                
                current_loc = ('S', station_id)
            
            # Ajouter la mini-route si elle contient des livraisons
            if mini_route_visits:
                solution.add_mini_route(vehicle.id, depot_id, chosen_product, mini_route_visits)
                print(f"  ✓ Véhicule {vehicle.id}: mini-route avec produit {chosen_product}, "
                      f"{len(mini_route_visits)} livraisons")
        
        if iteration >= max_iterations:
            print(f" Limite d'itérations atteinte ({max_iterations})")
        
        # Calculer le coût
        solution.calculate_cost()
        return solution
    
    def improve_solution(self, solution, max_iterations=100):
        """Amélioration locale par 2-opt sur les mini-routes"""
        improved = True
        iteration = 0
        
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            for vehicle_id, mini_routes in solution.routes.items():
                for mr_idx, mini_route in enumerate(mini_routes):
                    visits = mini_route['visits']
                    
                    if len(visits) <= 2:
                        continue
                    
                    # Essayer toutes les permutations 2-opt
                    for i in range(len(visits) - 1):
                        for j in range(i + 2, len(visits)):
                            # Inverser le segment [i+1, j]
                            new_visits = visits[:i+1] + visits[i+1:j+1][::-1] + visits[j+1:]
                            
                            # Calculer le gain
                            old_cost = self.calculate_mini_route_distance(
                                mini_route['depot_id'], visits
                            )
                            new_cost = self.calculate_mini_route_distance(
                                mini_route['depot_id'], new_visits
                            )
                            
                            if new_cost < old_cost:
                                mini_route['visits'] = new_visits
                                improved = True
        
        solution.calculate_cost()
        return solution
    
    def calculate_mini_route_distance(self, depot_id, visits):
        """Calcule la distance d'une mini-route"""
        if not visits:
            return 0
        
        distance = 0
        current = ('D', depot_id)
        
        for station_id, qty in visits:
            distance += self.instance.get_distance(
                current[0], current[1], 'S', station_id
            )
            current = ('S', station_id)
        
        return distance

def write_solution(solution, output_filename):
    """Écrit la solution au format requis"""
    with open(output_filename, 'w') as f:
        for vehicle in solution.instance.vehicles:
            v_id = vehicle.id
            mini_routes = solution.routes[v_id]
            
            if not mini_routes:
                # Véhicule non utilisé: juste garage -> garage
                f.write(f"{vehicle.garage_id} - {vehicle.garage_id}\n")
                f.write(f"{vehicle.initial_product}(0.0) - {vehicle.initial_product}(0.0)\n\n")
                continue
            
            # Construire la ligne de route
            route_line = [str(vehicle.garage_id)]
            product_line = [f"{vehicle.initial_product}(0.0)"]
            
            current_product = vehicle.initial_product
            
            for mini_route in mini_routes:
                depot_id = mini_route['depot_id']
                product_id = mini_route['product_id']
                visits = mini_route['visits']
                
                # Aller au dépôt
                route_line.append(str(depot_id))
                
                # Coût de changement si nécessaire
                if current_product != product_id:
                    changeover = solution.instance.transition_costs[current_product - 1][product_id - 1]
                    product_line.append(f"{product_id}({changeover})")
                else:
                    product_line.append(f"{product_id}(0.0)")
                
                current_product = product_id
                
                # Livraisons
                for station_id, qty in visits:
                    route_line.append(f"{station_id} ( {qty} )")
                    product_line.append(f"{product_id}(0.0)")
            
            # Retour au garage
            route_line.append(str(vehicle.garage_id))
            product_line.append(f"{current_product}(0.0)")
            
            f.write(" - ".join(route_line) + "\n")
            f.write(" - ".join(product_line) + "\n\n")
        
        # Résumé
        f.write(f"Distance totale parcourue : {solution.total_distance:.1f}\n")
        f.write(f"Coût total de changement de produit : {solution.total_changeover_cost:.1f}\n")
        f.write(f"Coût total : {solution.total_cost:.1f}\n")

def validate_solution(instance, solution):
    """Vérifie que la solution respecte toutes les contraintes"""
    errors = []
    
    # Vérifier que toutes les demandes sont satisfaites
    delivered = {}
    for station in instance.stations:
        for product_id in station.demands.keys():
            delivered[(station.id, product_id)] = 0
    
    for vehicle_id, mini_routes in solution.routes.items():
        for mini_route in mini_routes:
            for station_id, qty in mini_route['visits']:
                product_id = mini_route['product_id']
                key = (station_id, product_id)
                delivered[key] = delivered.get(key, 0) + qty
    
    for station in instance.stations:
        for product_id, demand in station.demands.items():
            key = (station.id, product_id)
            if delivered.get(key, 0) != demand:
                errors.append(f"Station {station.id}, Produit {product_id}: "
                            f"livré {delivered.get(key, 0)}, demandé {demand}")
    
    # Vérifier la capacité des véhicules
    for vehicle in instance.vehicles:
        for mini_route in solution.routes[vehicle.id]:
            total_loaded = sum(qty for _, qty in mini_route['visits'])
            if total_loaded > vehicle.capacity:
                errors.append(f"Véhicule {vehicle.id}: capacité dépassée "
                            f"({total_loaded} > {vehicle.capacity})")
    
    if errors:
        print(" ERREURS DE VALIDATION:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print(" Solution valide!")
        return True

# ============== PROGRAMME PRINCIPAL ==============

if __name__ == "__main__":
    # Charger l'instance
    instance_file = "MPVRP_S_047_s6_d1_p2.dat"
    print(f" Chargement de l'instance: {instance_file}")
    
    instance = MPVRPInstance(instance_file)
    
    print(f"\n Informations sur l'instance:")
    print(f"  - Produits: {instance.nb_products}")
    print(f"  - Véhicules: {instance.nb_vehicles}")
    print(f"  - Garages: {instance.nb_garages}")
    print(f"  - Dépôts: {instance.nb_depots}")
    print(f"  - Stations: {instance.nb_stations}")
    
    print(f"\n Détails des véhicules:")
    for v in instance.vehicles:
        print(f"  - Véhicule {v.id}: capacité={v.capacity}, produit initial={v.initial_product}, garage={v.garage_id}")
    
    # Résoudre
    print(f"\n Résolution en cours...")
    solver = MPVRPSolver(instance)
    solution = solver.solve()
    
    print(f"\n Solution initiale:")
    print(f"  - Distance totale: {solution.total_distance:.2f}")
    print(f"  - Coût de changement: {solution.total_changeover_cost:.2f}")
    print(f"  - Coût total: {solution.total_cost:.2f}")
    
    # Amélioration
    print(f"\n Amélioration de la solution...")
    solution = solver.improve_solution(solution)
    
    print(f"\n Solution améliorée:")
    print(f"  - Distance totale: {solution.total_distance:.2f}")
    print(f"  - Coût de changement: {solution.total_changeover_cost:.2f}")
    print(f"  - Coût total: {solution.total_cost:.2f}")
    
    # Validation
    print(f"\n Validation de la solution:")
    validate_solution(instance, solution)
    
    # Écrire la solution
    output_file = "Sol_" + instance_file
    write_solution(solution, output_file)
    print(f"\n Solution enregistrée dans: {output_file}")