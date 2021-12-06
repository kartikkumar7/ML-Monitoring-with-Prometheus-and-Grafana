import spacy

nlp = spacy.load("en_core_web_md")


def ner_spacy(text):
    doc1 = nlp(text)
    for ent in doc1.ents:
        yield ent.text, ent.label_
