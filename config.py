import spotpy

class OptimizationConfig:
    TEST_MODE = True
    DREAM_CHAINS = 20 if TEST_MODE else 40
    REPETITIONS = 1000 if TEST_MODE else 200
    OBS_PATH = 'Output1_Input2/obs_values.csv'
    OBS_CSV_PATH = 'Output1_Input2/obs.csv'
    DB_NAME = 'DREAM_GWM_test' if TEST_MODE else 'DREAM_GWM_run'
    DB_FORMAT = 'csv'
    
    @classmethod
    def get_parameter_set(cls):
        if cls.TEST_MODE:
            return [
                spotpy.parameter.Uniform(name='hk1', low=100, high=10000),
                spotpy.parameter.Uniform(name='hk2', low=100, high=10000),
                spotpy.parameter.Uniform(name='hk3', low=100, high=10000),
                spotpy.parameter.Uniform(name='hk4', low=100, high=10000),
                spotpy.parameter.Uniform(name='hk5', low=100, high=10000),
                spotpy.parameter.Uniform(name='Kriv_Isar', low=10, high=1000),
            ]
        else:
            return [
                # Hydraulic Conductivities (m/d)
                spotpy.parameter.Uniform(name='hk1', low=100, high=10000),
                spotpy.parameter.Uniform(name='hk2', low=100, high=10000),
                spotpy.parameter.Uniform(name='hk3', low=100, high=10000),
                spotpy.parameter.Uniform(name='hk4', low=100, high=10000),
                spotpy.parameter.Uniform(name='hk5', low=100, high=10000),
                # Specific Yield (-)
                spotpy.parameter.Uniform(name='sy1', low=0.05, high=0.35),
                spotpy.parameter.Uniform(name='sy2', low=0.05, high=0.35),
                spotpy.parameter.Uniform(name='sy3', low=0.05, high=0.35),
                spotpy.parameter.Uniform(name='sy4', low=0.05, high=0.35),
                spotpy.parameter.Uniform(name='sy5', low=0.05, high=0.35),
                # River Stage and Conductance
                spotpy.parameter.Uniform(name='D_Isar', low=-0.5, high=0.5),
                spotpy.parameter.Uniform(name='Kriv_Isar', low=10, high=1000),
                spotpy.parameter.Uniform(name='Kriv_Muhlbach', low=10, high=1000),
                spotpy.parameter.Uniform(name='Kriv_Giessen', low=10, high=1000),
                spotpy.parameter.Uniform(name='Kriv_Griesbach', low=10, high=1000),
                spotpy.parameter.Uniform(name='Kriv_Schwabinger_Bach', low=10, high=1000),
                spotpy.parameter.Uniform(name='Kriv_Wiesackerbach', low=10, high=1000),
                # Recharge Multipliers (-)
                spotpy.parameter.Uniform(name='D_rch1', low=0, high=3),
                spotpy.parameter.Uniform(name='D_rch2', low=0, high=1)
            ]
    
    @classmethod
    def get_default_parameters(cls):
        return {
            # Default hydraulic conductivities (m/d)
            'hk1': 5000,
            'hk2': 3000, 
            'hk3': 4000,
            'hk4': 2000,
            'hk5': 1500,
            # Default specific yields (-)
            'sy1': 0.2,
            'sy2': 0.15,
            'sy3': 0.25,
            'sy4': 0.18,
            'sy5': 0.12,
            # Default river parameters
            'D_Isar': 0.0,
            'Kriv_Isar': 500,
            'Kriv_Muhlbach': 200,
            'Kriv_Giessen': 150,
            'Kriv_Griesbach': 100,
            'Kriv_Schwabinger_Bach': 80,
            'Kriv_Wiesackerbach': 60,
            # Default recharge multipliers (-)
            'D_rch1': 1.5,
            'D_rch2': 0.5
        }
    
    @classmethod
    def print_config(cls):
        mode = "TEST MODE" if cls.TEST_MODE else "FULL MODE"
        n_params = len(cls.get_parameter_set())
        
        print(f"=== OPTIMIZATION CONFIGURATION ({mode}) ===")
        print(f"Number of parameters to optimize: {n_params}")
        print(f"DREAM chains: {cls.DREAM_CHAINS}")
        print(f"Repetitions per chain: {cls.REPETITIONS}")
        print(f"Total model runs: {cls.DREAM_CHAINS * cls.REPETITIONS}")
        print(f"Database name: {cls.DB_NAME}")
        
        if cls.TEST_MODE:
            print("\nParameters being optimized (TEST MODE):")
            for param in cls.get_parameter_set():
                print(f"  - {param.name}")
        print("=" * 50)
