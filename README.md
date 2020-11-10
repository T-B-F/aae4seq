# seq2aae
Set of tools to encode protein sequences using an Adversarial AutoEncoder and explore the corresponding latent space


## installation

Create a conda environment and activate it

```
conda create --name aae python=3.6
conda activate aae
```
(for analyses, install ete3)
```
conda install -c etetoolkit ete3 ete_toolchain 
```

Install required packages

```
pip install -r requirements.txt
```

export src path to `PYTHONPATH`

```
export PYTHONPATH=$PWD:$PYTHONPATH
```


## Train a model


Train the exemple model
```
python src/aae_model.py -i data/Sulfatases/fasta/hmmsearch_gc90_gs25.fasta -o log/ 
```

