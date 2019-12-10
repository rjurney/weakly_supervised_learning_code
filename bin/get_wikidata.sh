#!/bin/bash

# Get all entities in wikidata
curl -Lko data/wikidata/entities-latest-all.json.bz2 https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2

# Get all wikipedia pages
curl -Lko data/wikidata/enwiki-latest-pages-articles-multistream.xml.bz2 https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
