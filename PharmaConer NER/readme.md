## PharmaCoNER: Pharmacological Substances, Compounds and proteins and Named Entity Recognition

http://temu.bsc.es/pharmaconer/

### Task Specifications
For this task we have a manually classified collection of clinical case sections derived from Open access Spanish medical publications, named the Spanish Clinical Case Corpus (SPACCC). The corpus contains a total of 1000 clinical cases / 396,988 words. It is noteworthy to say that this kind of narrative shows properties of both, the biomedical and medical literature as well as clinical records. Participants are asked to classify objects in four granularity levels:

* NER offset and entity classification

### Dataset

The annotation of the entire set of entity mentions was carried out by medicinal chemistry experts and it includes the following four entity types:

<b> NORMALIZABLES </b>: Mentions of chemicals that can be manually normalized to a unique concept identifier (primarily SNOMED-CT).

<b> NO_NORMALIZABLES </b>: Mentions of chemicals that could not be normalized manually to a unique concept identifier.

<b> PROTEINAS </b>: Mentions of proteins and genes following an adaptation of the BioCreative GPRO track annotation guidelines. This class includes also peptides, peptide hormones and antibodies.

<b> UNCLEAR </b>: Cases of general substance class mentions of clinical and biomedical relevance, including certain pharmaceutical formulations, general treatments, chemotherapy programs, vaccines and a predefined set of general substances (e.g.: Estragón, Silimarina, Bromelaína, Melanina, Vaselina, Lanolina, Alcohol, Tabaco, Marihuana, Cannabis, Opio and Gluten). Mentions of this class will not be part of the entities evaluated by this track, but serve as additional annotations of medical relevance.

### Scoring

Average macro F1
  
### Solution overview

1. Run notebooks/data_prep.ipyb
2. Use notebooks/Bi-LSTM notebooks

### Final score

Final macro F1 score achieved is 0.67.

### Other links

Workshop proceedings - https://www.aclweb.org/anthology/D19-5701.pdf

ACL proceedings (all workshop papers) - https://www.aclweb.org/anthology/D19-57.pdf



