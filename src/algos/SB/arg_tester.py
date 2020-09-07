import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Pass in parameters. ')
    parser.add_argument('--n_steps', type=int, required=False, default=100, help='Number of training steps .')
    parser.add_argument('--terrain_type', type=str, default="flat", help='Type of terrain for training .')
    parser.add_argument('--lr', type=int, default=0.001, help='Learning rate .')
    parser.add_argument('--batchsize', type=int, default=32, help='Batchsize .')

    args = parser.parse_args()
    return args.__dict__

args = parse_args()
print(args)