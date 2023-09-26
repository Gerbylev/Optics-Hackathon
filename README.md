# Optics-Hackathon

![image](https://github.com/Gerbylev/Optics-Hackathon/assets/33491221/431aa76d-455b-4727-9200-fdfb599129b2)

## Goal
Develop an algorithm to find the optimal optical scheme for the specified parameters

## Parameters, Materials and Limitations

### Optical parameters
- Wavelength: 470-650 nm
- Field of view (sensor): 2,0 mm
- Symmetry: axis
- Number of lenses: >=2

### Allowed Materials
- Plastic_1: n_1 = 1.54, Abbe_num_1 = 75.0
- Plastic_2: n_2 = 1.67, Abbe_num_2 = 39.0
- Plastic_mix, k=[0..1]: n_1 * k + n_2 * (1-k), Abbe_num_1 * k + Abbe_num_2 * (1-k)

### Surfaces Types
- Spherical convex and concave surfaces
- Aspheric surfaces
- Any other surfaces, the main requirement is that they have a minimum thickness and work correctly in ray-optics package

### Design Limitations
- Focal distance: 5.0 mm
- F/#: ≤2.1
- Total length: ≤7mm
- Lens thickness: ≥100µ
- Air thickness: ≥0

## Loss Function
### Parameters
- Spot RMS: the smaller the better
- Encircled energy in D20µ: ≥80%
Spot Root Mean Square shows how well parallel incoming rays are focused at the destination, and is calculated for 5 different points of view

![image](https://github.com/Gerbylev/Optics-Hackathon/assets/33491221/38fd6ca6-52d4-46d0-beb7-5d8984d52f38)

pip install -r requirements.txt 