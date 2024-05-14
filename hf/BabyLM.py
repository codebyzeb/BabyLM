import datasets

from typing import List

_DESCRIPTION = """\
Dataset for the BabyLM Round2: French, German, Chinese & Japanese Small-Scale LMs
The goal is to train a language model from scratch on this data which represents
roughly the amount of text and speech data a young child observes.
Author– Suchir Salhan
"""

_HOMEPAGE = "https://babylm.github.io"

filenames = [
    "aochildes.txt",
    "aochinese.txt",
    "aochinese_dev.txt",
    "aochinese_test.txt",
    "aofrench.txt",
    "aofrench_dev.txt",
    "aofrench_test.txt",
    "aogerman.txt",
    "aogerman_dev.txt",
    "aogerman_test.txt",
    "aojapanese.txt",
    "aojapanese_dev.txt",
    "aojapanese_test.txt",
    "bnc_spoken.txt",
    "cbt.txt",
    "children_stories.txt",
    "gutenberg.txt",
    "open_subtitles.txt",
    "qed.txt", 
    "simple_wikipedia.txt",
    "switchboard.txt",
    "wikipedia.txt"
]


en_filenames = [
    "aochildes.txt",
    "bnc_spoken.txt",
    "cbt.txt",
    "children_stories.txt",
    "gutenberg.txt",
    "open_subtitles.txt",
    "qed.txt", 
    "simple_wikipedia.txt",
    "switchboard.txt",
    "wikipedia.txt"
]

#Suchir Salhan– addition of French, German, Japanese and Chinese dataset BUILDER_CONFIGS

class BabyLM(datasets.GeneratorBasedBuilder):
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="original_strict_small",
            description="Original dataset, 10M words, no POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="strict_small",
            description="Cleaned version of the dataset, 10M words, no POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="original_strict",
            description="Original dataset, 100M words, no POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="strict",
            description="Cleaned version of the dataset, 100M words, unsupervised POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="original_strict_small_gold",
            description="Original dataset, 10M words, gold POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="strict_small_gold",
            description="Cleaned version of the dataset, 10M words, gold POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="original_strict_gold",
            description="Original dataset, 100M words, gold POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="strict_gold",
            description="Cleaned version of the dataset, 100M words, gold POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="fr_lang_small",  #FRENCH
            description="FRENCH Cleaned version of the dataset, 10M words, unsupervised POS tags",
            version="1.0.0",
        ),
         datasets.BuilderConfig(
            name="ja_lang_small",
            description="GERMAN Cleaned version of the dataset, 10M words, unsupervised POS tags",
            version="1.0.0",
        ),
         datasets.BuilderConfig(
            name="zh_lang_small",
            description="JAPANESE Cleaned version of the dataset, 10M words, unsupervised POS tags",
            version="1.0.0",
        ),
         datasets.BuilderConfig(
            name="de_lang_small",
            description="GERMAN Cleaned version of the dataset, 10M words, unsupervised POS tags",
            version="1.0.0",
        ),
        
        datasets.BuilderConfig(
            name="fr_lang_gold",
            description="FRENCH Cleaned version of the dataset, 100M words, gold POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="ja_lang_gold",
            description="JAPANESE Cleaned version of the dataset, 100M words, gold POS tags",
            version="1.0.0",
        ),
                datasets.BuilderConfig(
            name="de_lang_gold",
            description="GERMAN Cleaned version of the dataset, 100M words, gold POS tags",
            version="1.0.0",
        ),
        datasets.BuilderConfig(
            name="zh_lang_gold",
            description="CHINESE Cleaned version of the dataset, 100M words, gold POS tags",
            version="1.0.0",
        ),
    ]

    DEFAULT_CONFIG_NAME = "strict_small"


    def _info(self):
            features = datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "tagged_text": datasets.Value("string"),
                    "filename": datasets.Value("string"),
                }
            )
            return datasets.DatasetInfo(
                # This is the description that will appear on the datasets page.
                description=_DESCRIPTION,
                features=features,  # Here we define them above because they are different between the two configurations
                homepage=_HOMEPAGE,
            )


#Suchir Salhan– addition of French, German, Japanese and Chinese datasets


    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        """ 
        Returns data for different splits 
        """


        if "fr_lang_small" in self.config.name:
            train_data_dir = "FR"
        elif "de_lang_small" in self.config.name:
            train_data_dir = "DE"
        elif "zh_lang_small" in self.config.name:
            train_data_dir = "ZH"
        elif "ja_lang_small" in self.config.name:
            train_data_dir = "JA"
        elif "strict_small" in self.config.name: #default settings – English
            train_data_dir = "10M"
        else: 
            train_data_dir = "100M"


        folder = 'original_tagged' if 'original' in self.config.name else 'clean_tagged' #
        folder = folder + '_gold' if 'gold' in self.config.name else folder #gold tags for french, german, japanese and english


        #modified urls to download

        urls_to_download = {
            "train": [],
            "dev": [],
            "test": []
            }

        if 'fr_lang_small' in self.config.name:
            urls_to_download["train"].append(f"{folder}/FR/aofrench.txt")
            urls_to_download["dev"].append(f"{folder}/dev/aofrench_dev.txt")
            urls_to_download["test"].append(f"{folder}/test/aofrench_test.txt")
        elif 'de_lang_small' in self.config.name:
            urls_to_download["train"].append(f"{folder}/DE/aogerman.txt")
            urls_to_download["dev"].append(f"{folder}/dev/aogerman_dev.txt")
            urls_to_download["test"].append(f"{folder}/test/aogerman_test.txt")
        elif 'zh_lang_small' in self.config.name:
            urls_to_download["train"].append(f"{folder}/ZH/aochinese.txt")
            urls_to_download["dev"].append(f"{folder}/dev/aochinese_dev.txt")
            urls_to_download["test"].append(f"{folder}/test/aochinese_test.txt")
        elif 'ja_lang_small' in self.config.name:
            urls_to_download["train"].append(f"{folder}/JA/aojapanese.txt")
            urls_to_download["dev"].append(f"{folder}/dev/aojapanese_dev.txt")
            urls_to_download["test"].append(f"{folder}/test/aojapanese_test.txt")
        else:
            urls_to_download["train"] = [f"{folder}/10M/{fn}" for fn in en_filenames]
            urls_to_download["dev"] = [f"{folder}/dev/{fn}" for fn in en_filenames]
            urls_to_download["test"] = [f"{folder}/test/{fn}" for fn in en_filenames]

        
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "filepaths": downloaded_files["train"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "dev",
                    "filepaths": downloaded_files["dev"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "filepaths": downloaded_files["test"]
                }
            ),
        ]


     # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, split, filepaths):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        # the filepaths should be a list of filepaths 
        if isinstance(filepaths, str):
            filepaths = [filepaths]
        
        global_idx = 0 

        for filepath in filepaths:
            with open(filepath, encoding="utf-8") as f:
                is_tags = False
                text = ""
                filename = ""
                # Every other row contains POS tags. First row is the filename (we can't use filepath since the file path changes upon caching)
                for row in f:
                    if filename == "":
                        filename = row.strip()
                        continue
                    if is_tags:
                        yield global_idx, {"text": text.strip(), "tagged_text": row.strip(), "filename": filename}
                        global_idx += 1 
                        is_tags = False
                    else:
                        text = row
                        is_tags = True

