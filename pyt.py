# Libraries

import pandas as pd
import random
import math
import numpy as np

# Gene & Chromosome Class

class Gene:
    def __init__(self, airport, aircraft):
        self.airport = airport
        self.aircraft = aircraft

class Chromosome:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = -1

    def compute_fitness(self, fitness_function, weather_data, aircraft_data, airport_data, travel_date):
        self.fitness = fitness_function.calculate(self, weather_data, aircraft_data, airport_data, travel_date)

    def __str__(self):
        route_description = ' -> '.join([f"{gene.airport}({gene.aircraft})" for gene in self.genes])
        return f"Route: {route_description}\n"

# Population Class

class Population:
    def __init__(self, size, generate_chromosome):
        self.size = size
        self.generate_chromosome = generate_chromosome
        self.chromosomes = [self.generate_chromosome() for _ in range(size)]

    def __str__(self):
        population_str = ""
        for i, chromosome in enumerate(self.chromosomes):
            population_str += f"Chromosome {i + 1}:\n"
            population_str += str(chromosome)  # Use the __str__ method of Chromosome
            population_str += "\n"
        return population_str

    def tournament_selection(self, tournament_size=3):
      selected_parents = []
      for _ in range(self.size):
          tournament = random.sample(self.chromosomes, tournament_size)
          winner = max(tournament, key=lambda chromo: chromo.fitness)
          selected_parents.append(winner)
      return selected_parents


    def ordered_crossover(self, parent1, parent2):
        child1_genes = []
        child2_genes = []

        start, end = sorted(random.sample(range(len(parent1.genes)), 2))

        middle1 = parent1.genes[start:end]
        middle2 = parent2.genes[start:end]

        remainder1 = [gene for gene in parent2.genes if gene not in middle1]
        remainder2 = [gene for gene in parent1.genes if gene not in middle2]

        child1_genes.extend(remainder1[:start])
        child1_genes.extend(middle1)
        child1_genes.extend(remainder1[start:])

        child2_genes.extend(remainder2[:start])
        child2_genes.extend(middle2)
        child2_genes.extend(remainder2[start:])

        return Chromosome(child1_genes), Chromosome(child2_genes)

    def mutate_swap(self, chromosome, mutation_rate=0.01):
        for _ in range(len(chromosome.genes)):
            if random.random() < mutation_rate:
                # Select two different indices to swap
                idx1, idx2 = random.sample(range(len(chromosome.genes)), 2)
                # Swap the genes at these indices
                chromosome.genes[idx1], chromosome.genes[idx2] = chromosome.genes[idx2], chromosome.genes[idx1]

    def evolve(self, fitness_function, weather_data, aircraft_data, airport_data, travel_date):
        new_population = []
        while len(new_population) < self.size:
            parents = self.tournament_selection(tournament_size=3)
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = self.ordered_crossover(parent1, parent2)
            self.mutate_swap(child1, mutation_rate = 0.01)
            self.mutate_swap(child2, mutation_rate = 0.01)
            child1.compute_fitness(fitness_function, weather_data, aircraft_data, airport_data, travel_date)
            child2.compute_fitness(fitness_function, weather_data, aircraft_data, airport_data, travel_date)
            new_population.extend([child1, child2])
        self.chromosomes = sorted(new_population, key=lambda chromo: chromo.fitness, reverse=True)[:self.size]

# Fitness Function Class

class FitnessFunction:
    def __init__(self, alpha, beta, gamma):
        self.alpha = alpha  # weight for distance cost
        self.beta = beta    # weight for fuel efficiency cost
        self.gamma = gamma  # weight for weather impact cost

    def calculate(self, chromosome, weather_data, aircraft_data, airport_data, travel_date):

        distance_cost = self.calculate_distance_cost(chromosome, airport_data)
        fuel_efficiency_cost = self.calculate_fuel_efficiency_cost(chromosome, aircraft_data, airport_data)
        weather_impact_cost = self.calculate_weather_impact_cost(chromosome, weather_data, airport_data, travel_date)

        combined_fitness = (self.alpha * distance_cost + self.beta * fuel_efficiency_cost + self.gamma * weather_impact_cost)

        return combined_fitness

    # def calculate_distance_cost(self, chromosome, airport_data):
    #     total_distance = 0

    #     unique_cities = airport_data['City'].unique()

    #     # Create a dictionary to map cities to their corresponding ICAO codes
    #     city_to_ICAO_mapping = {city[:4].upper(): city for city in unique_cities}

    #     for i in range(len(chromosome.genes) - 1):
    #         airport1 = chromosome.genes[i].airport
    #         airport2 = chromosome.genes[i+1].airport

    #         # print('Airport1:', airport1)
    #         # print('Airport2:', airport2)

    #         # Map airport1 and airport2 to ICAO codes if represented by city names
    #         city1 = city_to_ICAO_mapping.get(airport1[:4].upper(), None)
    #         city2 = city_to_ICAO_mapping.get(airport2[:4].upper(), None)

    #         # print('city1',city1)
    #         # print('city2',city2)

    #         # Find the row in the airport dataset that matches airport1's ICAO code and city2
    #         row1 = airport_data[(airport_data['ICAO Code'] == airport1) & (airport_data['City'] == city2)]
    #         if not row1.empty:
    #           lat1, lon1 = row1[['Latitude', 'Longitude']].values[0]

    #         # Find the row in the airport dataset that matches airport2's ICAO code and city1
    #         row2 = airport_data[(airport_data['ICAO Code'] == airport2) & (airport_data['City'] == city1)]
    #         if not row2.empty:
    #           lat2, lon2 = row2[['Latitude', 'Longitude']].values[0]

    #         # print(f'dummy value of {airport1} & {city2} are:',lat1, lon1)
    #         # print(f'dummy value of {airport2} & {city1} are:',lat2, lon2)

    #         total_distance += haversine_distance(lat1, lon1, lat2, lon2)

    #         # print('Total_distance: ',total_distance)

    #         # print("-----------------------------------------\n")

    #     return total_distance

    def calculate_distance_cost(self, chromosome, airport_data):
    total_distance = 0
    unique_cities = airport_data['City'].unique()

    city_to_ICAO_mapping = {city[:4].upper(): city for city in unique_cities}

    for i in range(len(chromosome.genes) - 1):
        airport1 = chromosome.genes[i].airport
        airport2 = chromosome.genes[i + 1].airport

        city1 = city_to_ICAO_mapping.get(airport1[:4].upper(), None)
        city2 = city_to_ICAO_mapping.get(airport2[:4].upper(), None)

        lat1, lon1, lat2, lon2 = None, None, None, None  # Initialize latitudes and longitudes with default values

        # Retrieve latitude and longitude values for airport1 and airport2 from airport_data
        if city1 and city2:
            row1 = airport_data[(airport_data['ICAO Code'] == airport1) & (airport_data['City'] == city2)]
            row2 = airport_data[(airport_data['ICAO Code'] == airport2) & (airport_data['City'] == city1)]

            if not row1.empty and not row2.empty:
                lat1, lon1 = row1[['Latitude', 'Longitude']].values[0]
                lat2, lon2 = row2[['Latitude', 'Longitude']].values[0]

        if lat1 is not None and lon1 is not None and lat2 is not None and lon2 is not None:
            total_distance += haversine_distance(lat1, lon1, lat2, lon2)
        else:
            # Handle cases where data is missing or invalid
            pass

    return total_distance


    def calculate_fuel_efficiency_cost(self, chromosome, aircraft_data, airport_data):
    fuel_cost = 0
    unique_cities = airport_data['City'].unique()

    city_to_ICAO_mapping = {city[:4].upper(): city for city in unique_cities}


        for i in range(len(chromosome.genes) - 1):
        aircraft_type = chromosome.genes[i].aircraft
        airport1 = chromosome.genes[i].airport
        airport2 = chromosome.genes[i+1].airport

            # print('aircraft type:', aircraft_type)
            # print('airport1:',airport1)
            # print('airport2:',airport2)

            # Map airport1 and airport2 to ICAO codes if represented by city names
            city1 = city_to_ICAO_mapping.get(airport1[:4].upper(), None)
            city2 = city_to_ICAO_mapping.get(airport2[:4].upper(), None)

            # print('city1',city1)
            # print('city2',city2)

            # Find the row in the airport dataset that matches airport1's ICAO code and city2
            lat1, lon1 = None, None
            row1 = airport_data[(airport_data['ICAO Code'] == airport1) & (airport_data['City'] == city2)]
            if not row1.empty:
            lat1, lon1 = row1[['Latitude', 'Longitude']].values[0]  # <--- lat1 used here
            lat2, lon2 = None, None
            # Find the row in the airport dataset that matches airport2's ICAO code and city1
            row2 = airport_data[(airport_data['ICAO Code'] == airport2) & (airport_data['City'] == city1)]
            if not row2.empty:
                lat2, lon2 = row2[['Latitude', 'Longitude']].values[0]

            # print(f'dummy value of {airport1} & {city2} are:',lat1, lon1)
            # print(f'dummy value of {airport2} & {city1} are:',lat2, lon2)


            # Calculate the distance using the haversine formula
            distance = haversine_distance(lat1, lon1, lat2, lon2)

            # print('Haversine Distance: ',distance)

            # Get the fuel consumption per km for the aircraft type
            fuel_consumption_per_km = aircraft_data[aircraft_data['Aircraft Type'] == aircraft_type]['Fuel Consumption at Cruise'].values[0]

            # nan values avoidance
            fuel_consumption_per_km = fuel_consumption_per_km if not np.isnan(fuel_consumption_per_km) else 0

            # print('Fuel/km: ',fuel_consumption_per_km)

            # Calculate the fuel cost for this segment
            fuel_cost += fuel_consumption_per_km * distance

        return fuel_cost


    def calculate_weather_impact_cost(self, chromosome, weather_data, airport_data, travel_date):
        weather_cost = 0

        # Extract unique city names from the airport_data DataFrame
        unique_cities = airport_data['City'].unique()

        # Create a dictionary to map cities to their corresponding ICAO codes
        city_to_ICAO_mapping = {city[:4].upper(): city for city in unique_cities}

        # Assuming 'travel_date' is the date you want to search for
        max_wind_speed = weather_data.loc[weather_data['Date'] == travel_date, 'Wind Speed'].max()
        max_wind_direction = weather_data.loc[weather_data['Date'] == travel_date, 'Wind Direction'].max()

        # Replace NaN values with 0
        max_wind_speed = max_wind_speed if not np.isnan(max_wind_speed) else 0
        max_wind_direction = max_wind_direction if not np.isnan(max_wind_direction) else 0

        for i in range(len(chromosome.genes) - 1):
            airport1 = chromosome.genes[i].airport
            airport2 = chromosome.genes[i + 1].airport

            # Extract the city names corresponding to the airports
            city1 = city_to_ICAO_mapping.get(airport1[:4].upper(), None)
            city2 = city_to_ICAO_mapping.get(airport2[:4].upper(), None)

            if city1 is not None and city2 is not None:
                # Assuming that weather_data is a DataFrame
                matching_rows = weather_data[(weather_data['Date'] == travel_date) & (weather_data['City'].isin([city1, city2]))]

                if not matching_rows.empty:
                    # Retrieve wind speed and wind direction for the corresponding cities
                    wind_speed1 = matching_rows[matching_rows['City'] == city1]['Wind Speed'].values[0]
                    wind_speed2 = matching_rows[matching_rows['City'] == city2]['Wind Speed'].values[0]
                    wind_direction1 = matching_rows[matching_rows['City'] == city1]['Wind Direction'].values[0]
                    wind_direction2 = matching_rows[matching_rows['City'] == city2]['Wind Direction'].values[0]

                    # Replace NaN values with 0
                    wind_speed1 = wind_speed1 if not np.isnan(wind_speed1) else 0
                    wind_speed2 = wind_speed2 if not np.isnan(wind_speed2) else 0
                    wind_direction1 = wind_direction1 if not np.isnan(wind_direction1) else 0
                    wind_direction2 = wind_direction2 if not np.isnan(wind_direction2) else 0

                    # Calculate the impact score based on wind speed and direction
                    speed_impact1 = wind_speed1 / max_wind_speed
                    speed_impact2 = wind_speed2 / max_wind_speed
                    direction_impact1 = abs(wind_direction1 - max_wind_direction) / 360.0
                    direction_impact2 = abs(wind_direction2 - max_wind_direction) / 360.0

                    # Average the impact scores for the two airports, considering both speed and direction
                    avg_impact_score = (speed_impact1 + speed_impact2 + direction_impact1 + direction_impact2) / 4

                    # Add the average impact score to the weather cost
                    weather_cost += avg_impact_score

        return weather_cost

# Genetic Algorithm Class

class GeneticAlgorithm:
    def __init__(self, population, fitness_function, max_generations):
        self.population = population
        self.fitness_function = fitness_function
        self.max_generations = max_generations

    def run(self, weather_data, aircraft_data, airport_data, travel_date):
        for generation in range(self.max_generations):
            print(f"Generation {generation + 1}/{self.max_generations}")

            # Evaluate the fitness of each chromosome in the population
            for chromosome in self.population.chromosomes:
                chromosome.compute_fitness(self.fitness_function, weather_data, aircraft_data, airport_data, travel_date)

            # Evolve the population to the next generation
            self.population.evolve(self.fitness_function, weather_data, aircraft_data, airport_data, travel_date)

            # Optionally, print the best fitness in the population after each generation
            best_fitness = max(self.population.chromosomes, key=lambda chromo: chromo.fitness).fitness
            print(f"Best fitness in current generation: {best_fitness}")

        # After the final generation, return the best chromosome as the optimal solution
        best_chromosome = max(self.population.chromosomes, key=lambda chromo: chromo.fitness)
        return best_chromosome

## Generating Chromosomes

def generate_chromosome(source_airport, destination_airport, aircraft_data, airport_data):
    genes = []

    # Function to get a random aircraft type available at a given airport
    def get_random_aircraft(airport):
        available_aircraft = aircraft_data[aircraft_data['ICAO CODES'] == airport]['Aircraft Type']
        return random.choice(available_aircraft.tolist()) if not available_aircraft.empty else None

    # Add the source airport with a random available aircraft
    source_aircraft = get_random_aircraft(source_airport)

    if source_aircraft:
        genes.append(Gene(source_airport, source_aircraft))
    else:
        # No available aircraft at source airport
        print("No available aircraft at source airport.")
        return Chromosome([])

    # Generate a list of possible stopover airports excluding source and destination
    stopover_airports = list(set(airport_data['ICAO Code']) - {source_airport, destination_airport})
    random.shuffle(stopover_airports)
    stopover_airports = stopover_airports[:10]  # Limit to 10 to keep total genes to 12

    # Create genes for each stopover with a randomly selected available aircraft
    for airport in stopover_airports:
        aircraft = get_random_aircraft(airport)
        if aircraft:
            genes.append(Gene(airport, aircraft))

    # Add the destination airport with a random available aircraft
    destination_aircraft = get_random_aircraft(destination_airport)

    if destination_aircraft:
        genes.append(Gene(destination_airport, destination_aircraft))
    else:
        # Incomplete chromosome if no aircraft available at destination
        print("No available aircraft at destination airport.")
        return Chromosome(genes)

    # Ensure 12 unique genes in the chromosome
    if len(genes) != 12:
        print("Not enough genes in the chromosome.")
        return Chromosome([])

    return Chromosome(genes)

## Haversine Distance Function

# Assuming latitude and longitude are in degrees and converting them to radians in the distance formula
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c
def main():
    # Load datasets into pandas DataFrames
    weather_data = pd.read_csv('/data/weatherDataset.csv')
    aircraft_data = pd.read_csv('/data/aircraftDataset.csv')
    airport_data = pd.read_csv('/data/airportDataset.csv')

    # Parameters for the genetic algorithm
    population_size = 50
    max_generations = 10
    alpha, beta, gamma = 0.5, 1.0, 0.75  # Example weights for fitness function

    # Define source and destination airports and travel date
    source_airport = ''
    destination_airport = ''
    travel_date = ''

    # Initialize the fitness function
    fitness_function = FitnessFunction(alpha, beta, gamma)

    # Initialize the initial population
    initial_population = Population(population_size,
        lambda: generate_chromosome(source_airport, destination_airport, aircraft_data, airport_data))

    # Initialize the genetic algorithm
    ga = GeneticAlgorithm(initial_population, fitness_function, max_generations)

    # Run the genetic algorithm
    best_chromosome = ga.run(weather_data, aircraft_data, airport_data, travel_date)
    res=[(gene.airport, gene.aircraft) for gene in best_chromosome.genes]
    ress=best_chromosome.fitness

    # Output the best route found
    print(f"Best route found: {[(gene.airport, gene.aircraft) for gene in best_chromosome.genes]}")
    print(f"With fitness: {best_chromosome.fitness}")

if __name__ == "__main__":
    main()

#LOG TABLE

