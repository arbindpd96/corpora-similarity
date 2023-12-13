EMBEDDINGS_MODEL = "text-embedding-ada-002"
EMBEDDINGS_BATCH_SIZE = 1024  # up to 2048

# Template prompt
prompt_extract = """
Extract the following features from the movie synopsis in about 850 characters:

- Themes
- Plot points
- Character archetypes
- Other notable features

Make sure to address character motivations, conflicts, and relationships. \
Don't use any proper nouns or character names.

Movie title: {}
Movie synopsis: {}
"""
