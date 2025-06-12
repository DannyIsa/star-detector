import pytest
import random
import itertools
from algorithms import *
from helper_functions import *
import numpy as np
from math import radians, cos, sin

class TestAlgorithm1:
    

    # Test case 1: Empty input - no detected stars
    def test_empty_input(self):
        detected_stars = []
        star_catalog = [
            {'RA': 100.0, 'Dec': 20.0, 'HR': 1, 'N': 'Star1'},
            {'RA': 110.0, 'Dec': 25.0, 'HR': 2, 'N': 'Star2'},
            {'RA': 120.0, 'Dec': 30.0, 'HR': 3, 'N': 'Star3'}
        ]
        
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        assert result is None, "Function should return None for empty input"
    
    # Test case 2: Insufficient stars (less than 3)
    def test_insufficient_stars(self):
        detected_stars = [
            {'x': 100, 'y': 200},
            {'x': 150, 'y': 250}
        ]
        star_catalog = [
            {'RA': 100.0, 'Dec': 20.0, 'HR': 1, 'N': 'Star1'},
            {'RA': 110.0, 'Dec': 25.0, 'HR': 2, 'N': 'Star2'},
            {'RA': 120.0, 'Dec': 30.0, 'HR': 3, 'N': 'Star3'}
        ]
        
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        assert result is None, "Function should return None for insufficient stars"
    
    # Test case 3: Perfect match - known stars
    def test_perfect_match(self):
        detected_stars = detect_stars("test_image.png") 
        
        
        star_catalog = [
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah', 'B': 'η', 'C': 'Vir', 'F': '15', 'K': '9500', 'V': '3.89'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima', 'B': 'γ', 'C': 'Vir', 'F': '29', 'K': '7500', 'V': '3.65'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva', 'B': 'δ', 'C': 'Vir', 'F': '43', 'K': '3050', 'V': '3.38'}
        ]
        
        expected_result = tuple(star_catalog)
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        
        assert result == expected_result, f"Expected {expected_result}, but got {result}"
    
    # Test case 4: No matching stars in catalog
    def test_no_matching_stars(self):
        detected_stars = [
            {'x': 100, 'y': 100},
            {'x': 200, 'y': 200},
            {'x': 300, 'y': 300}
        ]
        
        # Create a star catalog with very different angular distances
        star_catalog = [
            {'RA': 10.0, 'Dec': 10.0, 'HR': 1, 'N': 'Star1'},
            {'RA': 20.0, 'Dec': 20.0, 'HR': 2, 'N': 'Star2'},
            {'RA': 30.0, 'Dec': 30.0, 'HR': 3, 'N': 'Star3'}
        ]
        
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        
        # The function should still return a triplet even with high RMS error
        assert result is not None, "Function should return a triplet even with high RMS error"
        assert len(result) == 3, "Function should return exactly 3 stars"
    
    # Test case 5: Large and random catalog test
    def test_large_catalog(self):
        detected_stars = detect_stars("test_image.png")
        
        # Create a large catalog of 100 stars (a lot of outliers), randomly generated
        star_catalog = [
            {'RA': random.uniform(0, 360), 'Dec': random.uniform(-90, 90), 
             'HR': i, 'N': f'Star{i}'} for i in range(1, 100)
        ]
        
        # Add three stars that should match better than others
        star_catalog[0] = {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'}
        star_catalog[1] = {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'}
        star_catalog[2] = {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        
        result = stars_identification_bf(detected_stars, star_catalog, 1/16.30)
        
        # The test should verify that a triplet is returned
        assert result is not None, "Function should return a triplet"
        assert len(result) == 3, "Function should return exactly 3 stars"



class TestAlgorithm2:
    
    # Test case 1: Empty input - no detected stars
    def test_empty_input(self):
        detected_stars = []
        spht = {}
        al_parameter = 1
        
        result = stars_identification(detected_stars, spht, al_parameter, 16.30)
        assert result == [], "Function should return empty list for empty input"
    
    # Test case 2: Insufficient stars (less than 3)
    def test_insufficient_stars(self):
        detected_stars = [
            {'x': 100, 'y': 200},
            {'x': 150, 'y': 250}
        ]
        spht = {}
        al_parameter = 1
        
        result = stars_identification(detected_stars, spht, al_parameter, 16.30)
        assert result == [], "Function should return empty list for insufficient stars"
    
    # Test case 3: Perfect match with 3 stars, no outliers
    def test_perfect_match(self):
        # Three detected stars in the frame
        detected_stars = detect_stars("test_image.png")
        
        # Stars in the catalog
        catalog_stars = [
            {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
            {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
            {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        ]
        
        bsc = get_star_catalog()
        subset_bsc = []

        # Find and from the original bsc by HR value (as string or int)
        hr_values = set(str(star["HR"]) for star in catalog_stars)
        for star in bsc:
            if str(star.get("HR")) in hr_values and star not in subset_bsc:
                subset_bsc.append(star)
        
        al_parameter = 1
        camera_scaling_factor = 16.30 # We can assume this is the scaling factor for the camera

        # Build the SPHT (Star Pattern Hash Table) for all possible triplets in subset_bsc (14 stars only)
        spht = {}
        for triplet in itertools.combinations(subset_bsc, 3):
            key = create_spht_key_offline(triplet, al_parameter)
            if key not in spht:
                spht[key] = []
            # Store the HR values (or another unique identifier) for the triplet
            spht[key].append(tuple(star.get("HR") for star in triplet))
            
        
        # Expected result: Match between detected stars and catalog stars
        expected_result = [
            {'coords': (46, 48), 'spht_value': 4689, 'confidence': 1}, 
            {'coords': (205, 32), 'spht_value': 4825, 'confidence': 1}, 
            {'coords': (135, 88), 'spht_value': 4910, 'confidence': 1}
            ]
        
        result = stars_identification(detected_stars, spht , al_parameter, camera_scaling_factor)
        assert result == expected_result, f"Expected {expected_result}, but got {result}"
    
    
    # Test case 4: Real life scenario, testing matching between offline generated keys and online generated keys created by the algorithm in real time.
    # This test is not trivial, even though the catalog only contains the 14 stars, the algorithm has to identify them by matching the pixel distances to angular distances.
    def test_multiple_matches(self):
        # Detect stars in the image
        detected_stars = detect_stars("ursa-major-reduced.png")
        
        # Stars in Ursa Major constellation
        ursa_major_bsc = [
            { "B": "μ", "N": "Tania Australis", "C": "UMa", "Dec": "+41° 29′ 58″", "F": "34", "HR": "4069", "K": "3500", "RA": "10h 22m 19.7s", "V": "3.05" },
            { "B": "λ", "N": "Tania Borealis", "C": "UMa", "Dec": "+42° 54′ 52″", "F": "33", "HR": "4033", "K": "9500", "RA": "10h 17m 05.8s", "V": "3.45" },
            { "B": "θ", "N": "Sarir", "C": "UMa", "Dec": "+51° 40′ 38″", "F": "25", "HR": "3775", "K": "6600", "RA": "09h 32m 51.4s", "V": "3.17" },
            { "B": "β", "N": "Merak", "C": "UMa", "Dec": "+56° 22′ 57″", "F": "48", "HR": "4295", "K": "9750", "RA": "11h 01m 50.5s", "V": "2.37" },
            { "B": "ψ", "C": "UMa", "Dec": "+44° 29′ 55″", "F": "52", "HR": "4335", "K": "4850", "RA": "11h 09m 39.8s", "V": "3.01" },
            { "B": "χ", "N": "Al Kaphrah", "C": "UMa", "Dec": "+47° 46′ 46″", "F": "63", "HR": "4518", "K": "5000", "RA": "11h 46m 03.0s", "V": "3.71" },
            { "B": "γ", "N": "Phecda", "C": "UMa", "Dec": "+53° 41′ 41″", "F": "64", "HR": "4554", "K": "10000", "RA": "11h 53m 49.8s", "V": "2.44" },
            { "B": "δ", "N": "Megrez", "C": "UMa", "Dec": "+57° 01′ 57″", "F": "69", "HR": "4660", "K": "9250", "RA": "12h 15m 25.6s", "V": "3.31" },
            { "B": "ε", "N": "Alioth", "C": "UMa", "Dec": "+55° 57′ 35″", "F": "77", "HR": "4905", "K": "10000", "RA": "12h 54m 01.7s", "V": "1.77" },
            { "B": "α", "N": "Dubhe", "C": "UMa", "Dec": "+61° 45′ 03″", "F": "50", "HR": "4301", "K": "5000", "RA": "11h 03m 43.7s", "V": "1.79" },
            { "B": "υ", "C": "UMa", "Dec": "+59° 02′ 19″", "F": "29", "HR": "3888", "K": "7200", "RA": "09h 50m 59.4s", "V": "3.80" },
            { "C": "UMa", "Dec": "+63° 03′ 43″", "F": "23", "HR": "3757", "K": "7500", "RA": "09h 31m 31.7s", "V": "3.67" },
            { "B": "ο", "N": "Muscida", "C": "UMa", "Dec": "+60° 43′ 05″", "F": "1", "HR": "3323", "K": "5500", "RA": "08h 30m 15.9s", "V": "3.36" },
            { "B": "η", "N": "Alkaid", "C": "UMa", "Dec": "+49° 18′ 48″", "F": "85", "HR": "5191", "K": "24000", "RA": "13h 47m 32.4s", "V": "1.86" },
        ]
        
        bsc = get_star_catalog()
        subset_bsc = []

        # Find and add the Ursa Major stars from the original bsc by HR value (as string or int)
        hr_values = set(str(star["HR"]) for star in ursa_major_bsc)
        for star in bsc:
            if str(star.get("HR")) in hr_values and star not in subset_bsc:
                subset_bsc.append(star)

        # Parameters
        camera_scaling_factor = 18.18 # We can assume this is the scaling factor for the camera
        al_parameter = 1
        
        # Build the SPHT (Star Pattern Hash Table) for all possible triplets in subset_bsc (14 stars only)
        spht = {}
        for triplet in itertools.combinations(subset_bsc, 3):
            key = create_spht_key_offline(triplet, al_parameter)
            if key not in spht:
                spht[key] = []
            # Store the HR values (or another unique identifier) for the triplet
            spht[key].append(tuple(star.get("HR") for star in triplet))
            
        # Execute online algorithm on ursa-major-reduced.png
        result = stars_identification(detected_stars, spht , al_parameter, camera_scaling_factor)
        print(result)
    
    
    # Test case 5: real life scenario, 14 stars visible, 100 random stars (outliers) in the catalog.
    def real_life_scenario(self):
        # Detect stars in the image
        detected_stars = detect_stars("ursa-major-reduced.png")
        
        # Stars in Ursa Major constellation
        ursa_major_bsc = [
            { "B": "μ", "N": "Tania Australis", "C": "UMa", "Dec": "+41° 29′ 58″", "F": "34", "HR": "4069", "K": "3500", "RA": "10h 22m 19.7s", "V": "3.05" },
            { "B": "λ", "N": "Tania Borealis", "C": "UMa", "Dec": "+42° 54′ 52″", "F": "33", "HR": "4033", "K": "9500", "RA": "10h 17m 05.8s", "V": "3.45" },
            { "B": "θ", "N": "Sarir", "C": "UMa", "Dec": "+51° 40′ 38″", "F": "25", "HR": "3775", "K": "6600", "RA": "09h 32m 51.4s", "V": "3.17" },
            { "B": "β", "N": "Merak", "C": "UMa", "Dec": "+56° 22′ 57″", "F": "48", "HR": "4295", "K": "9750", "RA": "11h 01m 50.5s", "V": "2.37" },
            { "B": "ψ", "C": "UMa", "Dec": "+44° 29′ 55″", "F": "52", "HR": "4335", "K": "4850", "RA": "11h 09m 39.8s", "V": "3.01" },
            { "B": "χ", "N": "Al Kaphrah", "C": "UMa", "Dec": "+47° 46′ 46″", "F": "63", "HR": "4518", "K": "5000", "RA": "11h 46m 03.0s", "V": "3.71" },
            { "B": "γ", "N": "Phecda", "C": "UMa", "Dec": "+53° 41′ 41″", "F": "64", "HR": "4554", "K": "10000", "RA": "11h 53m 49.8s", "V": "2.44" },
            { "B": "δ", "N": "Megrez", "C": "UMa", "Dec": "+57° 01′ 57″", "F": "69", "HR": "4660", "K": "9250", "RA": "12h 15m 25.6s", "V": "3.31" },
            { "B": "ε", "N": "Alioth", "C": "UMa", "Dec": "+55° 57′ 35″", "F": "77", "HR": "4905", "K": "10000", "RA": "12h 54m 01.7s", "V": "1.77" },
            { "B": "α", "N": "Dubhe", "C": "UMa", "Dec": "+61° 45′ 03″", "F": "50", "HR": "4301", "K": "5000", "RA": "11h 03m 43.7s", "V": "1.79" },
            { "B": "υ", "C": "UMa", "Dec": "+59° 02′ 19″", "F": "29", "HR": "3888", "K": "7200", "RA": "09h 50m 59.4s", "V": "3.80" },
            { "C": "UMa", "Dec": "+63° 03′ 43″", "F": "23", "HR": "3757", "K": "7500", "RA": "09h 31m 31.7s", "V": "3.67" },
            { "B": "ο", "N": "Muscida", "C": "UMa", "Dec": "+60° 43′ 05″", "F": "1", "HR": "3323", "K": "5500", "RA": "08h 30m 15.9s", "V": "3.36" },
            { "B": "η", "N": "Alkaid", "C": "UMa", "Dec": "+49° 18′ 48″", "F": "85", "HR": "5191", "K": "24000", "RA": "13h 47m 32.4s", "V": "1.86" },
        ]
        
        # Add outliers to the catalog
        random.seed(42)
        bsc = get_star_catalog()
        subset_bsc = random.sample(bsc, 100)  # Take a random subset of 100

        # Find and add the Ursa Major stars from the original bsc by HR value (as string or int)
        hr_values = set(str(star["HR"]) for star in ursa_major_bsc)
        for star in bsc:
            if str(star.get("HR")) in hr_values and star not in subset_bsc:
                subset_bsc.append(star)

        # Parameters
        camera_scaling_factor = 18.18 # We can assume this is the scaling factor for the camera
        al_parameter = 1
        
        # Build the SPHT (Star Pattern Hash Table) for all possible triplets in subset_bsc (14 stars + 100 outliers)
        spht = {}
        for triplet in itertools.combinations(subset_bsc, 3):
            key = create_spht_key_offline(triplet, al_parameter,camera_scaling_factor)
            if key not in spht:
                spht[key] = []
            # Store the HR values (or another unique identifier) for the triplet
            spht[key].append(tuple(star.get("HR") for star in triplet))
            
        # Execute online algorithm on ursa-major-reduced.png
        result = stars_identification(detected_stars, spht , al_parameter, camera_scaling_factor)
        print(result)
                


# helper functions for algo 3
def make_unit_vector(ra_deg, dec_deg):
    ra = radians(ra_deg)
    dec = radians(dec_deg)
    return np.array([
        cos(dec) * cos(ra),
        cos(dec) * sin(ra),
        sin(dec)
    ])


class MockBSC:
    def __init__(self, stars):
        self.catalog = stars
        self.RaDec = np.array(
            [make_unit_vector(s['RA'], s['Dec']) for s in stars])


class TestValidationAlgorithmOrientation:

    def test_no_detected_stars(self):
        detected_stars = []
        orientation_matrix = np.eye(3)
        bsc_catalog = [{'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
                       {'RA': 190.415, 'Dec': -1.449444,
                           'HR': 4825, 'N': 'Porrima'},
                       {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}]
        raDecCalculations = [make_unit_vector(star['RA'], star['Dec']) for star in bsc_catalog]
        result = validation_algorithm_orientation(
            detected_stars, orientation_matrix, bsc_catalog, "test_image_black.jpg", raDecCalculation=raDecCalculations, confidence=[])
        assert result == None

    def test_insufficient_matches(self):
        detected_stars = [{'x': 200, 'y': 300}]
        bsc_catalog = [{'RA': 184.976667, 'Dec': -
                        0.666944, 'HR': 4689, 'N': 'Zaniah'}]
        raDecCalculations = [make_unit_vector(star['RA'], star['Dec']) for star in bsc_catalog]
        orientation_matrix = calculate_orientation_matrix(
            detected_stars, bsc_catalog, raDecCalculations)
        spht = {}
        confidence = stars_identification(detected_stars, spht, 1.5, 16.16)
        result = validation_algorithm_orientation(
            detected_stars, orientation_matrix, bsc_catalog, "test_image_2.png", raDecCalculations, confidence)
        assert result == float('inf')

    def test_large_random_bsc(self):
        detected_stars = [{'x': random.uniform(
            0, 1024), 'y': random.uniform(0, 1024)} for _ in range(10)]
        detected_stars += [{'x': 205, 'y': 30},
                           {'x': 135, 'y': 88}, {'x': 46, 'y': 48}]
        # orientation matrix is just an identity matrix, wrong matrix in this case.
        orientation_matrix = np.eye(3)
        bsc_catalog = [
            {
                'RA': random.uniform(0, 360),
                'Dec': random.uniform(-90, 90),
                'HR': i,
                'N': f'Star{i}'
            } for i in range(100)
        ]
        raDecCalculations = [make_unit_vector(star['RA'], star['Dec']) for star in bsc_catalog]
        confidence = stars_identification(detected_stars, spht={}, al_parameter=0.1, camera_scaling_factor=16.16)
        result = validation_algorithm_orientation(
            detected_stars, orientation_matrix, bsc_catalog, "ursa-major-original.jpg", raDecCalculations, confidence)
        assert isinstance(result, float)
        assert result >= 5  # bad result

    def test_upside_down_orientation(self):
        detected_stars = [{'x': 0, 'y': 0}]
        # Rotation that flips Z axis
        orientation_matrix = np.diag([1, 1, -1])  # 180° flip

        catalog_star = {'id': 0, 'RA': 0.0, 'Dec': 90.0}
        bsc = MockBSC([catalog_star])
        raDecCalculations = bsc.RaDec
        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        confidence = stars_identification(detected_stars, spht={}, al_parameter=0.4, camera_scaling_factor=18.18)
        cv2.imwrite("test_img.jpg", image)

        error = validation_algorithm_orientation(
            detected_stars, orientation_matrix, bsc, "test_img.jpg", raDecCalculations, confidence=confidence)

        assert error > 150.0  # Near-opposite direction

    def test_no_valid_match(self):
        detected_stars = [{'x': 0, 'y': 0}]
        orientation_matrix = np.identity(3)

        # Star far away from the direction Z points to
        catalog_star = {'id': 0, 'RA': 180.0, 'Dec': -90.0}
        bsc = MockBSC([catalog_star])

        image = np.zeros((1000, 1000, 3), dtype=np.uint8)
        cv2.imwrite("test_img.jpg", image)
        raDecCalculations = bsc.RaDec
        confidence = stars_identification(detected_stars, spht={}, al_parameter=0.4, camera_scaling_factor=18.18)
        error = validation_algorithm_orientation(
            detected_stars, orientation_matrix, bsc, "test_img.jpg", raDecCalculations, confidence)

        assert error == float('inf') or error > 90.0



class TestAlgorithm4:

    def test_empty_inputs(self):
        sm_table_for_individual_pixels_by_index = {} # no stars detected/matched
        result = setConfidence(sm_table_for_individual_pixels_by_index)
        assert result == {} or result is None, "Function should return empty dict or None for empty input"

    def test_single_single_match(self):
        sm = {
            0: [5524],
            1: [3757],
            2: [3888],
        }
        result = setConfidence(sm)
        expected_result = {0: (5524, 1), 1: (3757, 1), 2: (3888, 1)}
        assert result == expected_result, f'Expected a non-empty result, but got {result}'

    def test_stalemate_match(self):
        sm = {
            0: [3775],
            1: [3775],
            2: [3775],
            3: [3775],
            4: [3775],
        }
        result = setConfidence(sm)
        expected_result = {0: (3775, 1), 1: (3775, 1), 2: (3775, 1), 3: (3775, 1), 4: (3775, 1)}
        assert result == expected_result, f'Got {result} but expected {expected_result}'
       
    def test_large_random_data(self):
        sm = {
            0: [3757, 1589, 4022, 3821, 3824, 3323, 1829, 4022, 1684, 3158, 4384, 3323, 3323, 3323, 4022, 4022, 1540, 5891, 4022, 3536, 4562, 3666, 3620, 2346, 4562, 3666, 3536, 2672, 3666, 2623, 6211, 4022, 3666, 3824, 3888, 4562, 4022, 2672, 2051, 1659, 4022, 3666, 8950, 4020, 3821, 2292, 4562, 3821, 1312, 3824, 3824, 2555, 1144, 4516, 3824, 8099, 715, 5830, 4751, 4516, 2346, 3464, 3666, 3620, 2346, 3265, 1296, 8293, 4022, 3821, 3775, 3620, 4562, 3620, 5645, 3824, 5524, 5830, 3824, 3323, 4562, 8950, 8950, 3620, 7372, 3536, 3323, 3323, 3323, 3620, 5830, 3591, 5524],
            1: [3775, 920, 3821, 4384, 2293, 3757, 1684, 4384, 1659, 3775, 2685, 3757, 3757, 3757, 5891, 5645, 3821, 4384, 3536, 4384, 3757, 3775, 3620, 3743, 3536, 3757, 4069, 3536, 3757, 4033, 3620, 3620, 3757, 3620, 4562, 3757, 3620, 3757, 4518, 3666, 3757, 3757, 1829, 3620, 3757, 4069, 3757, 3757, 3743, 3757, 3757, 1684, 4022, 4562, 3824, 4020, 3757, 3757, 4562, 3757, 438, 4022, 3821, 1540, 3666, 3775, 3775, 4516, 4069, 4022, 3591, 3743, 8950, 2051, 1589, 3620, 1296, 3743, 1589, 4516, 3743, 3757, 3440, 3757, 3757, 3757, 2623, 3757, 3757, 4022, 3440, 3757, 1589, 3757, 3757, 3757, 3757, 4516, 3743, 3757, 3591, 3591, 3265, 3591, 3821, 3440, 3757, 3757, 4022, 3536, 3757, 3757, 4022, 4562, 3824, 4020, 3757, 3757, 4562, 3323, 1829, 3824, 3757, 3757, 4562, 3757],
            2: [3888, 3743, 4295, 3757, 3757, 3743, 1589, 2672, 4384, 3743, 2672, 5935, 4020, 2346, 4301, 4295, 3757, 3888, 4295, 3743, 3757, 3440, 3888, 4554, 3440, 3888, 4554, 3464, 3464, 3888, 3464, 4554, 3888, 3464, 3888, 4660, 4554, 3888, 3888, 3775, 3265, 3265, 3888, 4033, 3888, 3888, 3888, 3620, 2555, 2292, 3888, 3888, 8950, 8950, 3775, 3775, 3775, 8943, 3775, 1684, 1589, 4751, 3824, 3775, 3775, 3775, 4295, 1144, 3440, 3666, 4567, 3666, 8950, 3666, 4567, 3666, 4567, 3666, 8950, 3536, 3888, 3888, 3888, 3620, 4562, 4020, 4022, 3666, 3824, 3888, 3888, 6924, 3888, 3888, 3888, 3888, 4033, 3888, 3888, 3888, 9059, 4516, 3888, 2555, 3888, 3888, 3888, 3888, 3888, 3620, 2555, 2292, 3757, 5830, 4069, 3620, 4020, 3666, 3743, 3775, 4562, 3775, 3888, 4069, 4069, 3888, 3888],
        }
        result = setConfidence(sm)
        expected_result = {0: (4022, 9), 1: (3757, 41), 2: (3888, 36)}
        assert result == expected_result, f'Got {result} but expected {expected_result}'
