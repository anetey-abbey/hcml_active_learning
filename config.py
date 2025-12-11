SEED = 42
NUM_RUNS = 5
MODEL_NAME = "prajjwal1/bert-small"
DATASET = "gametox_merged"
DATASET_LABELS = {"gametox": 6, "gametox_merged": 4}
NUM_LABELS = DATASET_LABELS[DATASET]  # don't change this
LEARNING_RATE = 3e-5
EPOCHS = 2
BATCH_SIZE = 16

# active learning variables
POOL_SIZE = 5000  # num of random unlabeled samples from which can be selected during active learning (rule of thumb is use 10-20x labeling buget (=initial + (num iterations*samples per iteration)))
INITIAL_LABELED = 250  # num of labeled samples available at start of run
NUM_ITERATIONS = 50  # num of rounds of active learning sampling. an iteration trains+evals and selects additional top samples for labeled set
SAMPLES_PER_ITERATION = 10 # num of samples to add to labeled set each iteration
ANNOTATOR_NOISE = 0.1  # the value 0.0 means human labeler makes 0 mistakes 1.0 means all was labeled wrong
UNCERTAINTY_STRATEGY = "entropy"  # choose from these sampling strategies: "margin" or "entropy"
