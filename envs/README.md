When a new object is created, the default arguments will be number of human players 1, number of bots 1, infinite laps, 6 frames per second, and render_mode None:

    Number of players and number of bots can be any int, minimum of 1

    render_mode can be "human" or "rgb_array":

        "human" render mode will render the race in a visual way
        "rgb_array" will run the game only mathematically



The reset method of the object will set the players to the starting possition, and it returns observations and additional info.

    The observations is a 2d array:
        first row contaling a list of tuples indicating each player location,
        second row containg a list of tuples indicating the target checkpoint for each player.
        each with number of columns equal to the number of players.

        For example, a 3 player start would return:
        [[(12000, 1990), (12000, 1990), (12000, 1990)], 
         [(10680, 4990), (10680, 4990), (10680, 4990)]]
        

    The additional info is array of tuples containig the location of all players and all bots.


The step function takes an action in the shape (x, y, thrust) and it returns observation, reward, terminated, False and info

    The action must be in range (0 to WIDTH, 0 to HEIGHT, 0 to 101)

    The observations is a 2d array:
        first row contaling a list of tuples indicating each player location,
        second row containg a list of tuples indicating the target checkpoint for each player.
        each with number of columns equal to the number of players.


    The reward will be a value between -1 and 10

        to calculate the reward:
            
            a value -1 will be given if the distance did not decrease since last action, 
            a value between 0 and 1 will be calculated depending on the percentage of the distance that was traveled this action
            a value of 10 will be given if the agent gets within 800 units of the target, and a new target will be given


    If one of the palyers completes the number of laps (reaches all 4 targets) determined on itilization, the game will be terminated.


    
To create the environemnt

        from pod_racing import RaceTrackEnv

        env = RaceTrackEnv()

    Then the reset and step functions can be called

        observations, info = env.reset()

        observations, reward, terminated, info = env.step(action)

