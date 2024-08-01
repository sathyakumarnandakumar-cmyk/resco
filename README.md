# TABLE OF CONTENTS
1. **[PROJECT REQUIREMENTS](#setup)**
    - 1.1. **[REQUIRED TOOLS / PACKAGES](#requirements)**
    - 1.2. **[HOW TO GET Neptune.ai TOKEN](#neptune_token)**
2. **[HOW TO RUN AN EXPERIMENT](#run_experiment)**
    - 2.1. **[EXAMPLE: HOW TO RUN AN EXPERIMENT](#example_run_experiment)**
3. **[HOW TO VISUALIZE AN EXPERIMENT](#visualization)**
4. **[AGENT](#describe)**
<br></br>

## **1. PROJECT REQUIREMENTS <a name="setup"></a>**

### **1.1. REQUIRED TOOLS / PACKAGES <a name="requirements"></a>**

To run or visualize an experiment you must install:
- Python 3.10,
- **SUMO**, which must be [installed separately](https://sumo.dlr.de/docs/Installing/index.html). SUMO_HOME environment variable must be set, this is done automatically on the install of Sumo on Windows and Ubuntu. **SUMO 1.19.0 has been tested**,
- required packages from the requirements.txt (`pip install -r requirements.txt`)


### **1.2. HOW TO GET Neptune.ai TOKEN <a name="neptune_token"></a>**

* Log in to [Neptune.ai](https://ui.neptune.ai/). In the bottom-left corner, click the arrow (as shown in the screenshot below).

![press_on_the_arrow](./images/press_on_the_arrow.png)
* Click `Get your API token`.

![get_your_api_token](./images/get_your_api_token.png)
* Click the button highlighted in the image below.

![copy_the_api_token](./images/copy_the_api_token.png)
<br></br>



## **2. HOW TO RUN AN EXPERIMENT <a name="run_experiment"></a>**

Get your Neptune.ai token (**see: [HOW TO GET Neptune.ai TOKEN](#neptune_token)**), open `main.py` file and replace `"your_api_token"` in the `neptune.init_run()` method with your token.
```python
run = neptune.init_run(
            api_token="your_api_token",
            project="pgora/Malaysia2",
            name=f"{args.agent}-sumo-v0",
            description=f"Apply {args.agent} algorithm to the sumo-v0 environment",
            tags=[
                "sumo-v0",
                f"{args.agent}",
                f"Net: {args.net}",
                f"Activation: {args.activation}",
                "stable-baselines3",
                # "10 episodes - train, 1 episode - validation on new own generated file",
                "4 phases for PBB_Junc and SIRIM_Junc, 3 phases for INFMain_Junc - Full",
                "no new vehicles after 1 hour",
                f"Reward: {agent_configs[args.agent]['reward'].__name__}",
                # "5e5 steps",
                "7-8 am",
                # "aggregating data from lanes on the same road",
            ],
        )
```
Open the console, navigate to the `resco_benchmark` directory (`cd resco_benchmark`) and enter the following command:

`python main.py --map BB5B`<br></br>
Note that there are also other parameters to set.

<table><tbody>
<tr>
  <th> 
    
  **COMMAND NAME** </th>
  <th> 
    
  **AVAILABLE VALUES** </th>
  <th> 
    
  **DESCRIPTION** </th>
<tr>
  <td>
  --agent
  </td>
  <td>

  * IDQN
  * IPPO
  * STOCHASTIC
  </td>
  <td>
  RL algorithm
  </td>
<tr>
  <td>
    --map
  </td>
  <td> 
  BB5B
  </td>
  <td>
  Available scenario. Currently BB5B is the only available.
  </td>
<tr>
  <td>
    --eps_val
  </td>
  <td>
  Any integer greater than 1
  </td>
  <td>
    
  Expected number of validation episodes. By default every 11th episode will be the validation episode. You can change that by overwriting the `VALIDATION_INTERVAL` variable in `main.py`. 
  </td>
<tr>
  <td>
    --seed
  </td>
  <td>
  Any positive integer
  </td>
  <td>
    Allows to reproduce the results of the experiment. Experiment must be run on the same machine. Otherwise, the results might be different.
  </td>
<tr>
  <td>
    --net
  </td>
  <td>
    
  * _**default**_
  * _**double_conv**_
  </td>
  <td>
    Different PyTorch neural networks.
  </td>
<tr>
  <td>
    --activation
  </td>
  <td>
    
  * _**relu**_
  * _**leaky_relu**_
  * _**tanh**_
  * _**swish**_
  </td>
  <td>
    Different activation functions.
  </td>
<tr>
  <td>
    --gui
  </td>
  <td>
    
  _boolean_
  </td>
  <td>
    Allows to run simulation with visualization in SUMO. False by default.
  </td>
<tr>
  <td>
  log_dir
  </td>
  <td>
  Any directory (string)
  </td>
  <td>
  Location where experiment metrics and simulation data shared from SUMO are saved.
  </td>
<tr>
  <td>
  procs
  </td>
  <td>
  Any positive integer
  </td>
  <td>
  
  Runs the simulation on the provided number of processes. If `procs` is set to 2, it increases simulation performance.
  </td>
<tr>
  <td>
    --load
  </td>
  <td>
    
  _boolean_
  </td>
  <td>

  Loads provided models from `models/models_for_visualization` directory. Used for visualization, but also we can continue training from a checkpoint. To resume training, we need to enable the training mode for the model (for example `agents/pfrl_dqn.py` -> comment out line 34 `self.agents[key].agent.training = False`).
  </td>
<tbody></table>


### **2.1 EXAMPLE: HOW TO RUN AN EXPERIMENT <a name="example_run_experiment"></a>**

To run the experiment, open the console, navigate to the **resco_benchmark** (`cd resco_benchmark`) folder, and enter the following command: <br></br>
`python main.py --agent IDQN --eps_val 10 --map BB5B --seed 42 --net default --activation leaky_relu`
<br></br>



## **3. HOW TO VISUALIZE AN EXPERIMENT <a name="visualization"></a>**

To visualize the model, copy the token from the Neptune.ai (**see: [HOW TO GET Neptune.ai TOKEN](#neptune_token)**), open `main-testagent.py` file and replace `"your_api_token"` in the `neptune.init_run()` method with your token.

```python
run = neptune.init_run(
        api_token="your_api_token",
        project="pgora/Malaysia2",
        with_id=args.experiment_name,
        mode="read-only"
    )
```

After setting the token, open console, navigate to `resco_benchmark` (`cd resco_benchmark`), and enter the command:

`python main-testagent.py --experiment_name MAL2-623`

where `MAL2-623` is the name of the experiment you want to visualize. Training hyperparameters will be passed automatically. Keep in mind that the experiment must have a `models` directory on Neptune.ai. Otherwise, a FileNotFoundError will be returned.

## **4. AGENT <a name="describe"></a>**

- **State**: this is a dictionary where the keys are the names of intersections and their values are numpy arrays that 
  contain the current phase of the intersection with traffic lights, normalized approach, normalized waiting time, 
  normalized the total number of halting vehicles for the last time step on the incoming lanes (a speed of less than 
  0.1 m/s is considered a halt) and normalized vehicle speed of all vehicles on incoming lanes.
  You can find more information in [resco_benchmark/states.py ](https://gitlab.com/trafficsimulationframework/rl/resco-for-malaysia/-/blob/main/resco_benchmark/states.py)
- **Action**: a single action should be understood as changing the current green phase on intersection for next (1)
  or not (0). For *INFMain_Junc* we have 3 possible green phases and for *PBB_Junc* and *SIRIM_Junc* there are
  4 possible green phases. So the action is a vector containing information about 3 choosing green phases for
  intersections, eg. [0, 1, 0] means that we won't change the current phase on *PBB_Junc* and *SIRIM_Junc'* and
  change for next green phase on *INFMain_Junc*. It should be remembered that if the current green phase at the
  intersection is 1, the next green phase is 2 (not 3 or not 0). It is assumed that you cannot randomly choose phases.
  Note that changing phases then a mandatory yellow and red phase is enforced for a predefined time duration.
- **Reward**: several types available:
  - *wait*: the total waiting time of all vehicles in the incoming lanes between actions, where the waiting time of a 
    car is the count of seconds spent with speed below 0.1 m/s since the spawn in an incoming lane before junctions
    with traffic light;
  - *wait_norm*: normalized waiting time on incoming lanes;
  - *pressure*: queue length, i.e. the number of vehicles on the incoming lanes.
  You can find more information in [resco_benchmark/rewards.py](https://gitlab.com/trafficsimulationframework/rl/resco-for-malaysia/-/blob/main/resco_benchmark/rewards.py)

## **5. Metrics <a name="describe"></a>**
- **Average_time_of_journey**: average time of journey for all vehicles that have completed drive
- **Count_of_vehicles_completing_journey**: number of vehicles that have completed their journey
- **Total_average_delays_of_all_vehicles_from_all_routes**: average of all delays of all vehicles that completed their drive
- **Total_average_delays_real_times_by_ideal_times**: the sum of all real times divided by the sum of ideal times
- **Total_sum_delays_of_all_vehicles_from_all_routes**: the total sum of delays of all vehicles that have completed their drive
- **Total_time_of_journey**: the total sum of travel times for all vehicles
- **Total_waiting_time_all_vehicles_in_simulation_in_episode**: total waiting time of all vehicles in a simulation (all, not only those that have completed their drive)
- **Total_waiting_time_on_the_incoming_lanes_in_episode**: total waiting time of all vehicles on the incoming lanes (all, not only those that have completed their drive)
