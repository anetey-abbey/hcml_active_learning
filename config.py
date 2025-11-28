SEED = 42
NUM_RUNS = 2
MODEL_NAME = "prajjwal1/bert-mini"
DATASET = "ag_news"
DATASET_LABELS = {"aegis": 14, "ag_news": 4, "hate_speech_offensive": 3} # change only when adding a new dataset
NUM_LABELS = DATASET_LABELS[DATASET] # don't change this
LEARNING_RATE = 3e-5
EPOCHS = 2
BATCH_SIZE = 16

# active learning variables
POOL_SIZE = 2500 # num of random unlabeled samples from which can be selected during active learning (rule of thumb is use 10-20x labeling buget (=initial + (num iterations*samples per iteration)))
INITIAL_LABELED = 100 # num of labeled samples available at start of run
NUM_ITERATIONS = 40 # num of rounds of active learning sampling. an iteration trains+evals and selects additional top samples for labeled set
SAMPLES_PER_ITERATION = 10
ANNOTATOR_NOISE = 0.0 # the value 0.0 means human labeler makes 0 mistakes 1.0 means all was labeled wrong