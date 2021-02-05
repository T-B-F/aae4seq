# aae4seq
Set of tools to encode protein sequences using an Adversarial AutoEncoder and explore the corresponding latent space


## installation

Create a conda environment and activate it

```bash
conda create --name aae python=3.6
conda activate aae
```
(for analyses, install ete3)
```bash
conda install -c etetoolkit ete3 ete_toolchain 
```

Install required packages

```bash
pip install -r requirements.txt
```

export src path to `PYTHONPATH`

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```


## Train a model


Train the exemple model
```bash
python src/aae_model.py -i data/Sulfatases/fasta/hmmsearch_gc90_gs25.fasta -o log/ 
```


## Interpolation

Perform interpolation between sulfatase protein with pretrained model
```bash
python src/interpolate_space.py -i data/fasta/hmmsearch_gc90_gs25.fasta.gz -c data/fasta/hmmsearch_gc90_gs25_encoded.npz -d data/sulfatase_decoder_weights.h5 -q 0 -t 10000 -o interpolated_seq1_seq10000.fasta -s 50
```
