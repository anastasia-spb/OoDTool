from tool.core.data_projectors.data_projector import DataProjector


def test(embeddings_file):
    projector = DataProjector("tsne")
    output_file = projector.project(metadata_folder='./',
                                    embeddings_file=embeddings_file)
    print(output_file)


if __name__ == "__main__":
    test(embeddings_file='../../../../example_data/tool_working_dir/BalloonsBubbles/TimmResnetWrapper_BalloonsBubbles_1024_230430_001343.emb.pkl')
