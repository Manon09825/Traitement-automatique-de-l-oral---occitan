# Traitement automatique de l'oral - Occitan
Ce projet est mené dans le cadre du projet TARTOAS, mené conjointement par les laboratoires LATTICE, LPP et ERTIM.
L'objectif de ce projet est la conception d'un modèle de transcription de l'occitan oral vers l'alphabet phonétique international. Nous commençons par tester un modèle déjà existant: wav2vec2phoneme.

## Pipeline
La pipeline établie est la suivante:

Récupération des données (audios, transcriptions et référence)

|

 Pré-traitements

| 

Test du modèle de transcription phonétique

| 

Évaluation

## Données
Les données utilisée pour le premier test du modèle sont des fichiers mp3 issus du corpus occitan de CommonVoice (Mozilla), accessibles à [cette adresse](https://mozilladatacollective.com/datasets/cmn1qeqny00xno107p0rrn9lq). Les audios durent chacun entre 1 et 8 secondes et consistent en une phrase ou expression prononcée par un locuteur de l'occitan. Le corpus audio est accompagné d'un fichier tsv qui recense les audios et donne, pour chaque audio, sa transcription graphique.

J'ai également récupéré des données du corpus [OcOr](https://zenodo.org/records/145175), qui est un corpus de contes racontés par des locuteurs natifs, et j'ai commencé à constituer un corpus de chants à partir de CD, que je ne peux pas rendre disponible car les données ne sont pas dans le domaine public.

Afin d'avoir des données de référence auxquelles comparer les performances du modèle, nous avons passé la transcription graphique de chaque audio dans [le phonétiseur de Lo Congrès](http://fonetizar.locongres.com/). Cela nous a permis d'obtenir une transcription phonétique pour chaque audio (y_train et y_test).

Le phonétiseur de Lo Congrès renvoie la transcription dans les 6 variantes d'occitan (auvergnat, gascon, limousin, languedocien, provençal et vivaro-alpin). Comme la variété d'occitan du locuteur n'est pas renseignée dans le corpus de CommonVoice, et afin de ne pas pénaliser inutilement le modèle lors de l'évaluation, nous avons procédé à une étape de choix de la variante la plus proche. Pour chaque audio, nous avons calculé la distance entre la transcription du modèle et chacune des 6 transcriptions de Lo Congrès grâce au module `edit_distance` de nltk. Pour chaque audio, nous avons ensuite retenu, parmi les 6 transcriptions de Lo Congrès, celle qui s'approchait le plus de la transcription du modèle. Dans la majorité des cas, il s'agissait du gascon. Nous avons donc fini par déduire que les locuteurs de ce corpus étaient en majorité gascons.

## Le modèle
Le modèle de transcription que j'utilise est [Wav2Vec2Phoneme](https://doi.org/10.48550/arXiv.2109.11680), développé en 2021 par Xu et al. Il s'agit d'un modèle de transcription phonétique qui n'a pas été entraîné sur l'occitan. Le but ici est de voir où se situent ses performances sur l'occitan, et à quel point elles sont exploitables. Mon but n'est pas encore de le fine-tuner, mais de rendre compte de ses performances pour l'occitan.
Le modèle n'a pas été entraîné sur l'occitan mais il a été entraîné sur des langues proches, notamment le catalan, ce qui peut être intéressant, notamment si l'on s'intéresse à l'efficacité du transfert interlingue pour l'ASR.
