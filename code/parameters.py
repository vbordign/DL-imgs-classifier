N_SAMPLES = 1000
N_VAL = 100
SEED = 0

img_size = 14
num_rounds = 10
batch_size = 32
num_epochs = 30
learning_rate = 0.001
num_classes = 2
num_dig_classes = 10

rho = [1, 1, 1]

arch_setup = [[1,16],
              [1,32],
              [1, 16, 32],
              [1, 32, 64],
              [1, 16, 32, 64],
              [1, 32, 64, 128]]

kernel_setup =[3, 5]

