# Start

Make sure you have the `embeddings.csv` file in the `./data` directory

### Install

```
pip install -r ./requirements.txt
```

### Before running

Make sure `OPENAI_API_KEY` is exported to en env.

### Generate embeddings

Will use openai to generate embeddings from a csv file. File MUST have `id`, `title`, and `overview` columns.

```
python ./embeddings.py --movies_data='./file.csv'
```

### Run

```
streamlit run ./scorer.py
```

### TODO

- Structure the features according to what is important for us (perhaps x genre) to qualify as original idea.
- Data annotation / standardization. Its a challenge to get a great db for synopsis. We need some human labeling / reviewing
- Customized model for these feature extractions
