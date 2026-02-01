"""
================================================================================
SOLVEUR MPVRP-CC - VERSION FINALE
Multi-Product Vehicle Routing Problem with Changeover Cost

Basé sur la modélisation officielle (MVRP-CC_2.pdf)
Génère des solutions au format officiel (solution_description fr.pdf)
================================================================================

Installation : pip install ortools

Auteurs : Groupe 15
Date : 31 Janvier 2026
"""

from ortools.linear_solver import pywraplp
import math
import time as time_module
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ==============================================================================
# STRUCTURES DE DONNÉES
# ==============================================================================

@dataclass
class Vehicle:
    id: int
    capacity: int
    initial_product: int
    garage_id: int

@dataclass
class Depot:
    id: int
    x: float
    y: float
    stocks: List[int]

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
    demands: List[int]


# ==============================================================================
# CLASSE INSTANCE
# ==============================================================================

class MPVRPInstance:
    """Gère le chargement des données"""
    
    def __init__(self, filename: str):
        self.filename = filename
        self.nP = 0
        self.nK = 0
        self.nG = 0
        self.nD = 0
        self.nS = 0
        
        self.vehicles: List[Vehicle] = []
        self.depots: List[Depot] = []
        self.garages: List[Garage] = []
        self.stations: List[Station] = []
        
        self.C: Dict[Tuple[int, int], float] = {}  # Distances Cij
        self.changeover_costs: List[List[float]] = []  # Cp1p2
        
        self.load_instance(filename)
    
    def load_instance(self, filename: str):
        """Charge l'instance"""
        print(f" Chargement: {filename}")
        
        with open(filename, 'r') as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
        
        idx = 0
        
        # Paramètres
        params = list(map(int, lines[idx].split()))
        self.nP, self.nG, self.nD, self.nS, self.nK = params
        idx += 1
        
        print(f"  Produits: {self.nP} | Camions: {self.nK}")
        print(f"  Garages: {self.nG} | Dépôts: {self.nD} | Stations: {self.nS}")
        
        # Coûts de changement Cp1p2
        for i in range(self.nP):
            row = list(map(float, lines[idx].split()))
            self.changeover_costs.append(row)
            idx += 1
        
        # Véhicules
        for _ in range(self.nK):
            parts = list(map(int, lines[idx].split()))
            self.vehicles.append(Vehicle(*parts))
            idx += 1
        
        # Dépôts
        for _ in range(self.nD):
            parts = lines[idx].split()
            depot_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            stocks = list(map(int, parts[3:]))
            self.depots.append(Depot(depot_id, x, y, stocks))
            idx += 1
        
        # Garages
        for _ in range(self.nG):
            parts = lines[idx].split()
            garage_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            self.garages.append(Garage(garage_id, x, y))
            idx += 1
        
        # Stations
        for _ in range(self.nS):
            parts = lines[idx].split()
            station_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            demands = list(map(int, parts[3:]))
            self.stations.append(Station(station_id, x, y, demands))
            idx += 1
        
        # Debug: afficher les IDs
        print(f"\n   Debug:")
        print(f"     Garages IDs: {[g.id for g in self.garages]}")
        print(f"     Véhicules garage_id: {[v.garage_id for v in self.vehicles]}")
        
        # Calcul distances Cij
        self._compute_distances()
        print(f"   Instance chargée")
    
    def _compute_distances(self):
        """Calcule la matrice Cij (distances euclidiennes)"""
        all_nodes = {}
        
        for g in self.garages:
            all_nodes[('G', g.id)] = (g.x, g.y)
        for d in self.depots:
            all_nodes[('D', d.id)] = (d.x, d.y)
        for s in self.stations:
            all_nodes[('S', s.id)] = (s.x, s.y)
        
        for (t1, id1), (x1, y1) in all_nodes.items():
            for (t2, id2), (x2, y2) in all_nodes.items():
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                self.C[(t1, id1, t2, id2)] = dist
    
    def get_distance(self, type1: str, id1: int, type2: str, id2: int) -> float:
        """Retourne Cij"""
        return self.C.get((type1, id1, type2, id2), 0.0)


# ==============================================================================
# SOLVEUR MILP
# ==============================================================================

class MPVRPSolver:
    """Solveur basé sur la modélisation officielle"""
    
    def __init__(self, instance: MPVRPInstance, time_limit: int = 300):
        self.instance = instance
        self.time_limit = time_limit
        self.start_time = time_module.time()
        
        print("\n Initialisation SCIP...")
        self.solver = pywraplp.Solver.CreateSolver('SCIP')
        if not self.solver:
            raise RuntimeError(" Solver indisponible")
        
        # Variables selon la modélisation
        self.X = {}      # Xijk : routage
        self.t = {}      # tp1p2 : changement
        self.f = {}      # f^p_ijk : flux
    
    def build_model(self):
        """Construit le modèle selon la modélisation officielle"""
        print("\n" + "="*70)
        print("CONSTRUCTION DU MODÈLE")
        print("="*70)
        
        self._create_variables()
        self._set_objective()
        self._add_constraints()
        
        print(f"\n Modèle construit")
        print(f"  Variables   : {self.solver.NumVariables()}")
        print(f"  Contraintes : {self.solver.NumConstraints()}")
        print("="*70)
    
    def _create_variables(self):
        """Crée les variables"""
        print("\n Création des variables...")
        inst = self.instance
        
        # Variables Xijk
        print("   Xijk...", end='', flush=True)
        count = 0
        
        for k in inst.vehicles:
            # Vérifier que le garage existe
            garage_exists = any(g.id == k.garage_id for g in inst.garages)
            if not garage_exists:
                continue
            
            # Garage → Dépôt
            for g in inst.garages:
                if g.id == k.garage_id:
                    for d in inst.depots:
                        var_name = f'X_G{g.id}_D{d.id}_{k.id}'
                        self.X[('G', g.id, 'D', d.id, k.id)] = self.solver.BoolVar(var_name)
                        count += 1
            
            # Dépôt → Station
            for d in inst.depots:
                for s in inst.stations:
                    var_name = f'X_D{d.id}_S{s.id}_{k.id}'
                    self.X[('D', d.id, 'S', s.id, k.id)] = self.solver.BoolVar(var_name)
                    count += 1
            
            # Station → Station
            for s1 in inst.stations:
                for s2 in inst.stations:
                    if s1.id != s2.id:
                        var_name = f'X_S{s1.id}_S{s2.id}_{k.id}'
                        self.X[('S', s1.id, 'S', s2.id, k.id)] = self.solver.BoolVar(var_name)
                        count += 1
            
            # Station → Garage
            for s in inst.stations:
                for g in inst.garages:
                    if g.id == k.garage_id:
                        var_name = f'X_S{s.id}_G{g.id}_{k.id}'
                        self.X[('S', s.id, 'G', g.id, k.id)] = self.solver.BoolVar(var_name)
                        count += 1
        
        print(f" {count}")
        
        # Variables tp1p2
        print("   tp1p2...", end='', flush=True)
        count = 0
        for p1 in range(inst.nP):
            for p2 in range(inst.nP):
                if p1 != p2:
                    self.t[(p1, p2)] = self.solver.BoolVar(f't_{p1}_{p2}')
                    count += 1
        print(f" {count}")
        
        # Variables f^p_ijk
        print("   f^p_ijk...", end='', flush=True)
        count = 0
        
        for k in inst.vehicles:
            # Vérifier que le garage existe
            garage_exists = any(g.id == k.garage_id for g in inst.garages)
            if not garage_exists:
                continue
            
            for p in range(inst.nP):
                # Dépôt → Station
                for d in inst.depots:
                    for s in inst.stations:
                        var_name = f'f{p}_D{d.id}_S{s.id}_{k.id}'
                        self.f[(p, 'D', d.id, 'S', s.id, k.id)] = \
                            self.solver.NumVar(0, k.capacity, var_name)
                        count += 1
                
                # Station → Station
                for s1 in inst.stations:
                    for s2 in inst.stations:
                        if s1.id != s2.id:
                            var_name = f'f{p}_S{s1.id}_S{s2.id}_{k.id}'
                            self.f[(p, 'S', s1.id, 'S', s2.id, k.id)] = \
                                self.solver.NumVar(0, k.capacity, var_name)
                            count += 1
        
        print(f" {count}")
    
    def _set_objective(self):
        """Fonction objectif"""
        print("\n Fonction objectif...")
        inst = self.instance
        objective = self.solver.Objective()
        
        # Coût transport
        for key, var in self.X.items():
            t1, id1, t2, id2, k_id = key
            dist = inst.get_distance(t1, id1, t2, id2)
            objective.SetCoefficient(var, dist)
        
        # Coût changement
        for key, var in self.t.items():
            p1, p2 = key
            cost = inst.changeover_costs[p1][p2]
            objective.SetCoefficient(var, cost)
        
        objective.SetMinimization()
        print("  ✓ min(Transport + Changement)")
    
    def _add_constraints(self):
        """Ajoute les contraintes"""
        print("\n⚖️  Contraintes...")
        
        self._constraint_continuity()
        self._constraint_mandatory_delivery()
        self._constraint_demand_satisfaction()
        self._constraint_flow()
        self._constraint_capacity()
        self._constraint_depot_capacity()
        self._constraint_no_subtours()  # NOUVELLE contrainte
    
    def _constraint_no_subtours(self):
        """Empêche les sous-tours avec contraintes fortes"""
        print("  7  Élimination sous-tours...", end='', flush=True)
        inst = self.instance
        count = 0
        
        # CONTRAINTE FORTE : Pour chaque véhicule
        # La somme des arcs S→S ne peut PAS être >= nb_stations
        # (sinon c'est un cycle complet sans garage/dépôt)
        for k in inst.vehicles:
            garage = None
            for g in inst.garages:
                if g.id == k.garage_id:
                    garage = g
                    break
            
            if garage is None:
                continue
            
            # Contrainte 1 : Si on a des arcs S→S, on DOIT avoir G→D
            c = self.solver.Constraint(-self.solver.infinity(), 0)
            
            # Compter les arcs S→S
            for s1 in inst.stations:
                for s2 in inst.stations:
                    if s1.id != s2.id:
                        key = ('S', s1.id, 'S', s2.id, k.id)
                        if key in self.X:
                            c.SetCoefficient(self.X[key], 1)
            
            # On DOIT avoir au moins un départ du garage
            for d in inst.depots:
                key = ('G', garage.id, 'D', d.id, k.id)
                if key in self.X:
                    c.SetCoefficient(self.X[key], -100)  # Très grand pour forcer
            
            count += 1
            
            # Contrainte 2 : Pour chaque station visitée, il DOIT y avoir
            # un arc D→Station quelque part
            for s in inst.stations:
                c = self.solver.Constraint(-self.solver.infinity(), 0)
                
                # Si cette station est visitée (arcs sortants)
                for s2 in inst.stations:
                    if s2.id != s.id:
                        key = ('S', s.id, 'S', s2.id, k.id)
                        if key in self.X:
                            c.SetCoefficient(self.X[key], 1)
                
                for g in inst.garages:
                    key = ('S', s.id, 'G', g.id, k.id)
                    if key in self.X:
                        c.SetCoefficient(self.X[key], 1)
                
                # Il DOIT y avoir un arc D→Station ou S→Station entrant
                for d in inst.depots:
                    key = ('D', d.id, 'S', s.id, k.id)
                    if key in self.X:
                        c.SetCoefficient(self.X[key], -1)
                
                for s2 in inst.stations:
                    if s2.id != s.id:
                        key = ('S', s2.id, 'S', s.id, k.id)
                        if key in self.X:
                            c.SetCoefficient(self.X[key], -1)
                
                count += 1
        
        print(f" {count}")
    
    def _constraint_continuity(self):
        """Continuité du trafic"""
        print("  1  Continuité...", end='', flush=True)
        inst = self.instance
        count = 0
        
        for k in inst.vehicles:
            # Pour chaque station
            for s in inst.stations:
                c = self.solver.Constraint(0, 0)
                
                # Entrant
                for d in inst.depots:
                    key = ('D', d.id, 'S', s.id, k.id)
                    if key in self.X:
                        c.SetCoefficient(self.X[key], 1)
                
                for s2 in inst.stations:
                    if s2.id != s.id:
                        key = ('S', s2.id, 'S', s.id, k.id)
                        if key in self.X:
                            c.SetCoefficient(self.X[key], 1)
                
                # Sortant
                for s2 in inst.stations:
                    if s2.id != s.id:
                        key = ('S', s.id, 'S', s2.id, k.id)
                        if key in self.X:
                            c.SetCoefficient(self.X[key], -1)
                
                for g in inst.garages:
                    key = ('S', s.id, 'G', g.id, k.id)
                    if key in self.X:
                        c.SetCoefficient(self.X[key], -1)
                
                count += 1
            
            # Pour chaque dépôt
            for d in inst.depots:
                c = self.solver.Constraint(0, 0)
                
                # Entrant
                for g in inst.garages:
                    key = ('G', g.id, 'D', d.id, k.id)
                    if key in self.X:
                        c.SetCoefficient(self.X[key], 1)
                
                # Sortant
                for s in inst.stations:
                    key = ('D', d.id, 'S', s.id, k.id)
                    if key in self.X:
                        c.SetCoefficient(self.X[key], -1)
                
                count += 1
        
        print(f" {count}")
    
    def _constraint_mandatory_delivery(self):
        """Livraisons obligatoires"""
        print("  2  Livraisons...", end='', flush=True)
        inst = self.instance
        count = 0
        
        # Sortantes
        for s in inst.stations:
            c = self.solver.Constraint(1, self.solver.infinity())
            for k in inst.vehicles:
                for s2 in inst.stations:
                    if s2.id != s.id:
                        key = ('S', s.id, 'S', s2.id, k.id)
                        if key in self.X:
                            c.SetCoefficient(self.X[key], 1)
                for g in inst.garages:
                    key = ('S', s.id, 'G', g.id, k.id)
                    if key in self.X:
                        c.SetCoefficient(self.X[key], 1)
            count += 1
        
        # Entrantes
        for s in inst.stations:
            c = self.solver.Constraint(1, self.solver.infinity())
            for k in inst.vehicles:
                for d in inst.depots:
                    key = ('D', d.id, 'S', s.id, k.id)
                    if key in self.X:
                        c.SetCoefficient(self.X[key], 1)
                for s2 in inst.stations:
                    if s2.id != s.id:
                        key = ('S', s2.id, 'S', s.id, k.id)
                        if key in self.X:
                            c.SetCoefficient(self.X[key], 1)
            count += 1
        
        print(f" {count}")
    
    def _constraint_demand_satisfaction(self):
        """Satisfaction de la demande : Bps"""
        print("  3  Demande...", end='', flush=True)
        inst = self.instance
        count = 0
        
        for s in inst.stations:
            for p in range(inst.nP):
                if s.demands[p] == 0:
                    continue
                
                c = self.solver.Constraint(s.demands[p], s.demands[p])
                
                for k in inst.vehicles:
                    # Flux entrant
                    for d in inst.depots:
                        key_f = (p, 'D', d.id, 'S', s.id, k.id)
                        if key_f in self.f:
                            c.SetCoefficient(self.f[key_f], 1)
                    
                    for s2 in inst.stations:
                        if s2.id != s.id:
                            key_f = (p, 'S', s2.id, 'S', s.id, k.id)
                            if key_f in self.f:
                                c.SetCoefficient(self.f[key_f], 1)
                    
                    # Flux sortant
                    for s2 in inst.stations:
                        if s2.id != s.id:
                            key_f = (p, 'S', s.id, 'S', s2.id, k.id)
                            if key_f in self.f:
                                c.SetCoefficient(self.f[key_f], -1)
                
                count += 1
        
        print(f" {count}")
    
    def _constraint_flow(self):
        """Flux des camions (garage d'origine)"""
        print("  4 Flux...", end='', flush=True)
        inst = self.instance
        count = 0
        
        for k in inst.vehicles:
            garage = None
            for g in inst.garages:
                if g.id == k.garage_id:
                    garage = g
                    break
            
            if garage is None:
                continue
            
            # Contrainte : sortie du garage = entrée au garage
            c = self.solver.Constraint(0, 0)
            
            for d in inst.depots:
                key = ('G', garage.id, 'D', d.id, k.id)
                if key in self.X:
                    c.SetCoefficient(self.X[key], 1)
            
            for s in inst.stations:
                key = ('S', s.id, 'G', garage.id, k.id)
                if key in self.X:
                    c.SetCoefficient(self.X[key], -1)
            
            count += 1
        
        print(f" {count}")
    
    def _constraint_capacity(self):
        """Capacité : 0 ≤ f^p_ijk ≤ Qk.Xijk"""
        print("  5  Capacité...", end='', flush=True)
        inst = self.instance
        count = 0
        
        for key_f, var_f in self.f.items():
            p, t1, id1, t2, id2, k_id = key_f
            key_x = (t1, id1, t2, id2, k_id)
            
            if key_x in self.X:
                vehicle = next(v for v in inst.vehicles if v.id == k_id)
                
                c = self.solver.Constraint(-self.solver.infinity(), 0)
                c.SetCoefficient(var_f, 1)
                c.SetCoefficient(self.X[key_x], -vehicle.capacity)
                count += 1
        
        print(f" {count}")
    
    def _constraint_depot_capacity(self):
        """Capacité dépôts"""
        print("  6  Dépôts...", end='', flush=True)
        inst = self.instance
        count = 0
        
        for d in inst.depots:
            for k in inst.vehicles:
                for p in range(inst.nP):
                    c = self.solver.Constraint(0, self.solver.infinity())
                    
                    # Sortie
                    for s in inst.stations:
                        key = (p, 'D', d.id, 'S', s.id, k.id)
                        if key in self.f:
                            c.SetCoefficient(self.f[key], 1)
                    
                    count += 1
        
        print(f" {count}")
    
    def solve(self):
        """Résout"""
        print("\n" + "="*70)
        print("RÉSOLUTION")
        print("="*70)
        
        self.solver.SetTimeLimit(self.time_limit * 1000)
        status = self.solver.Solve()
        
        elapsed = time_module.time() - self.start_time
        
        print()
        if status == pywraplp.Solver.OPTIMAL:
            print(" SOLUTION OPTIMALE")
        elif status == pywraplp.Solver.FEASIBLE:
            print(" SOLUTION RÉALISABLE")
        else:
            print(" AUCUNE SOLUTION")
            return None
        
        print(f" Coût : {self.solver.Objective().Value():.2f}")
        print(f"⏱  Temps : {elapsed:.3f}s")
        print("="*70)
        
        return self.extract_solution(elapsed)
    
    def extract_solution(self, elapsed_time):
        """Extrait la solution"""
        inst = self.instance
        
        solution = {
            'vehicles_routes': {},
            'total_distance': 0,
            'total_changeover': 0,
            'num_vehicles_used': 0,
            'num_changeovers': 0,
            'elapsed_time': elapsed_time
        }
        
        # DEBUG : Afficher tous les arcs utilisés
        print("\n DEBUG - Arcs utilisés dans la solution:")
        arcs_count = 0
        for key, var in self.X.items():
            if var.solution_value() > 0.5:
                t1, id1, t2, id2, k_id = key
                dist = inst.get_distance(t1, id1, t2, id2)
                print(f"  Véhicule {k_id}: {t1}{id1} → {t2}{id2} (dist: {dist:.2f})")
                solution['total_distance'] += dist
                arcs_count += 1
        
        print(f"  Total arcs: {arcs_count}")
        
        # Changements
        for key, var in self.t.items():
            if var.solution_value() > 0.5:
                p1, p2 = key
                solution['total_changeover'] += inst.changeover_costs[p1][p2]
                solution['num_changeovers'] += 1
                print(f"  Changement: P{p1} → P{p2}")
        
        # Extraire routes
        print("\n DEBUG - Extraction des routes:")
        for k in inst.vehicles:
            # Vérifier si ce véhicule est utilisé
            is_used = False
            for key, var in self.X.items():
                if key[4] == k.id and var.solution_value() > 0.5:
                    is_used = True
                    break
            
            print(f"  Véhicule {k.id}: {'UTILISÉ' if is_used else 'NON UTILISÉ'}")
            
            if is_used:
                route = self._build_route_improved(k)
                if route and len(route) > 0:
                    solution['vehicles_routes'][k.id] = route
                    solution['num_vehicles_used'] += 1
                    print(f"    → Route extraite avec {len(route)} étapes")
                else:
                    print(f"    → ERREUR: Route vide ou None")
        
        return solution
    
    def _build_route_improved(self, vehicle):
        """Construit la route d'un véhicule - VERSION AMÉLIORÉE"""
        inst = self.instance
        
        # Collecter tous les arcs utilisés par ce véhicule
        arcs_used = []
        for key, var in self.X.items():
            if var.solution_value() > 0.5:
                t1, id1, t2, id2, k_id = key
                if k_id == vehicle.id:
                    arcs_used.append((t1, id1, t2, id2))
        
        if not arcs_used:
            return None
        
        # Reconstruire la séquence à partir du garage
        route = []
        current = None
        
        # Trouver le départ du garage
        for arc in arcs_used:
            if arc[0] == 'G' and arc[1] == vehicle.garage_id:
                current = arc
                break
        
        if current is None:
            return None
        
        visited = set()
        
        while current is not None:
            t1, id1, t2, id2 = current
            visited.add(current)
            
            # Calculer les quantités pour cet arc
            quantities = {}
            for p in range(inst.nP):
                key_f = (p, t1, id1, t2, id2, vehicle.id)
                if key_f in self.f:
                    qty = self.f[key_f].solution_value()
                    if qty > 0.1:
                        quantities[p] = qty
            
            route.append({
                'from': (t1, id1),
                'to': (t2, id2),
                'quantities': quantities
            })
            
            # Si on est de retour au garage, on s'arrête
            if t2 == 'G':
                break
            
            # Trouver l'arc suivant
            next_arc = None
            for arc in arcs_used:
                if arc not in visited and arc[0] == t2 and arc[1] == id2:
                    next_arc = arc
                    break
            
            current = next_arc
        
        return route if route else None


# ==============================================================================
# GÉNÉRATION FICHIER SOLUTION
# ==============================================================================

def write_solution_file(solution: dict, instance: MPVRPInstance, filename: str):
    """Génère le fichier solution au format officiel"""
    
    with open(filename, 'w') as f:
        # Routes des véhicules
        for k_id in sorted(solution['vehicles_routes'].keys()):
            route = solution['vehicles_routes'][k_id]
            vehicle = next(v for v in instance.vehicles if v.id == k_id)
            
            # Ligne 1 : séquence de visites
            visit_seq = [str(vehicle.garage_id)]
            
            for step in route:
                t_from, id_from = step['from']
                t_to, id_to = step['to']
                
                # Dépôt : [Qty]
                if t_to == 'D':
                    total_qty = int(sum(step['quantities'].values())) if step['quantities'] else 0
                    visit_seq.append(f"{id_to} [{total_qty}]")
                # Station : (Qty)
                elif t_to == 'S':
                    total_qty = int(sum(step['quantities'].values())) if step['quantities'] else 0
                    visit_seq.append(f"{id_to} ({total_qty})")
                # Garage
                elif t_to == 'G':
                    visit_seq.append(str(id_to))
            
            f.write(f"{k_id} " + " - ".join(visit_seq) + "\n")
            
            # Ligne 2 : produits et coûts
            prod_seq = []
            current_prod = None
            
            # Départ du garage
            if route and route[0]['quantities']:
                first_prod = list(route[0]['quantities'].keys())[0]
                prod_seq.append(f"{first_prod}(0.0)")
                current_prod = first_prod
            else:
                prod_seq.append("0(0.0)")
                current_prod = 0
            
            for step in route:
                if step['quantities']:
                    prod = list(step['quantities'].keys())[0]
                    
                    if prod != current_prod and current_prod is not None:
                        cost = instance.changeover_costs[current_prod][prod]
                        prod_seq.append(f"{prod}({cost:.1f})")
                        current_prod = prod
                    else:
                        prod_seq.append(f"{prod}(0.0)")
                        current_prod = prod
                else:
                    # Pas de quantité, garder le produit actuel
                    if current_prod is not None:
                        prod_seq.append(f"{current_prod}(0.0)")
            
            f.write(" - ".join(prod_seq) + "\n\n")
        
        # Métriques
        f.write(f"{solution['num_vehicles_used']}\n")
        f.write(f"{solution['num_changeovers']}\n")
        f.write(f"{solution['total_changeover']:.2f}\n")
        f.write(f"{solution['total_distance']:.2f}\n")
        f.write("Intel Core i7-10700K\n")
        f.write(f"{solution['elapsed_time']:.3f}\n")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SOLVEUR MPVRP-CC - VERSION FINALE")
    print("="*70)
    
    instance_file = "MPVRP_S_047_s6_d1_p2.dat"
    
    try:
        # Charger
        instance = MPVRPInstance(instance_file)
        
        # Résoudre
        solver = MPVRPSolver(instance, time_limit=300)
        solver.build_model()
        solution = solver.solve()
        
        if solution:
            # Générer fichier solution
            import os
            basename = os.path.basename(instance_file)
            output_file = f"Sol_{basename}"
            
            write_solution_file(solution, instance, output_file)
            
            print(f"\n Solution : {output_file}")
            print(f"   Véhicules : {solution['num_vehicles_used']}")
            print(f"   Distance  : {solution['total_distance']:.2f}")
            print(f"   Changeover: {solution['total_changeover']:.2f}")
            print(f"   Total     : {solution['total_distance'] + solution['total_changeover']:.2f}")
    
    except Exception as e:
        print(f"\n ERREUR : {e}")
        import traceback
        traceback.print_exc()
    
    print("\n FIN")