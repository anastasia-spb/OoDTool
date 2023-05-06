import argparse

from tool.core.density_based_sampler.sampler import DensityBasedSampler


def get_ood():
    parser = argparse.ArgumentParser(description='Calculate Distances based on embeddings')
    parser.add_argument('-emb', '--embeddings', default='', required=True)
    parser.add_argument('-ood', '--ood_file', default='', required=True)
    parser.add_argument('-n', '--n_samples', type=int, default=300, required=True)
    args = parser.parse_args()

    n_samples = args.n_samples
    assert n_samples > 0

    sampler = DensityBasedSampler(args.embeddings, args.ood_file)
    result = sampler.fit(n_samples=n_samples)
    print(result)


if __name__ == "__main__":
    get_ood()
