from tokenizers import (
    Tokenizer,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
    processors,
    decoders
)

tokenizer = Tokenizer(
    models.BPE(unk_token="<unk>")
)

tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

trainer = trainers.BpeTrainer(
    vocab_size=8000,
    special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]
)

tokenizer.train(["text_corpus.txt"], trainer)

tokenizer.post_processor = processors.TemplateProcessing(
    single="<bos> $A <eos>",
    pair="<bos> $A <sep> $B <eos>",
    special_tokens=[
        ("<bos>", tokenizer.token_to_id("<bos>")),
        ("<eos>", tokenizer.token_to_id("<eos>")),
        ("<sep>", tokenizer.token_to_id("<sep>")),
    ],
)

tokenizer.decoder = decoders.ByteLevel()

tokenizer.save("tokenizer.json")
