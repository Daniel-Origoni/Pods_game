DISCOUNT_FACTOR = 0.5
STATE_DIM = 5
ACTION_DIM = 3
HIDDEN_DIM = 7
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.0001

CRITIC_NETWORK = ["Dense", "Tanh", "Dense", "Tanh", "Dense", "Tanh"]
ACTOR_NETWORK = ["Dense", "Tanh", "Dense", "Tanh", "Dense", "Tanh"]