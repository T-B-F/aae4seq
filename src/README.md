# Script descriptions

## Main application

The main application is made of following scripts :

    aae_model.py
    utils.py
    io.py

## Utilities

### Sequence generation and visualization :

Apply a trained model to new sequences :

    python src/encoded2dim.py -i <new sequences> -o <encoded sequences> -e <model>

Visualize the first two dimenssions of the encoded space:

    python src/visualize_space.py -i <encoded sequences> -o <plot>

Perform spherical interpolation between points and run the decoder on the interpolated data :
  
    python src/interpolate_space.py -i <fasta file> -c <encoded sequences> -d <decoder model> -q <query index in fasta and coordinate files> -t <target index, see query> -o <output file> -s <number of step>

### Encoded sequence clustering

Apply hdbscan clustering on encoded sequences:

    python src/hdbscan_labels.py -i <encoded sequences> -o <output file> --min_cluster_size <hdbscan parameter, default=30> --min_samples <hdbscan parameter, default=-1> --metric <hdbscan parameter, default=euclidean>

Compute a dendrogram of the encoded sequences using the euclidean distance:

    python src/build_tree_encoded.py -i <encoded seuqneces> -o <output file>

Fit gaussian mixture models on encoded seqences:

    python src/mixture_models.py -i <encoded seuqneces> -o <output file>
  
