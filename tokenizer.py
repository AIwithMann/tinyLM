from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, processors

tokenizer = Tokenizer(models.BPE())

tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["<pad>","<unk>", "<bos>", "<eos>"]
)
tokenizer.train(["text_corpus.txt"], trainer)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
tokenizer.save("tokenizer.json")