import nlpaug.augmenter.word.context_word_embs as naw

test_sentence = 'long-term methylphenidate therapy in children with comorbid attention-deficit hyperactivity disorder and chronic multiple tic disorder.'
TOPK =1  # default=100
ACT ="substitute"

aug_bert = naw.ContextualWordEmbsAug(
    model_path='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
    device='cuda',
    action=ACT, top_k=TOPK)
print("Original:")
print(test_sentence)
print("Augmented Text:")
for ii in range(5):
    augmented_text = aug_bert.augment(test_sentence)
    print(augmented_text)