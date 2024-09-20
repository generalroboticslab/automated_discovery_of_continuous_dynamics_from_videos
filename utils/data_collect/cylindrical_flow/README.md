# Cylindrical Flow Data Collection

We generated the cylindrical flow dataset using direct numerical simulations on a domain of $−15 \leq x \leq 35,\, −15 \leq y \leq 15$ with a unit-diameter cylinder centered at $x = 0,\, y = 0$. The initial velocity was set as $\mathbf{u}^0 = (u^0_x, 0)$ where $u^0_x$ was sampled from the range $[0, 1]$. The dataset contains videos for flows with Reynold numbers Re ranging from 0 to 100.

Data generation was performed using the open-source CFD solver [Nek5000](https://github.com/Nek5000/Nek5000) and the `neksuite.py` script from the [PySINDy](https://github.com/dynamicslab/pysindy/tree/master) package.

## Steps to Reproduce

1. Download Nek5000 and set up the environment as instructed in the [Nek5000 Quickstart Guide](https://nek5000.github.io/NekDoc/quickstart.html).
   
2. Navigate to the `run` directory in Nek5000 and run:
    ```bash
    cp -r ../examples/ext_cyl .
    ```

3. Copy `simulate.py`, `make_data.py`, and `neksuite.py` here into the `run/ext_cyl` directory in Nek5000.

4. Run the simulation and data generation scripts under the `run/ext_cyl` directory in Nek5000:
    ```bash
    python simulate.py
    python make_data.py
    ```