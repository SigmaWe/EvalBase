import typing
from env import nlp_spacy


# EvalPieces = typing.Dict[str, typing.List[str]]  # { raw: segments }


class EvalPieces:
    raw_list: typing.List[str]
    segments_list: typing.List[typing.List[str]]

    def __init__(self, raw_list: typing.List[str]):
        self.raw_list = raw_list
        self.segments_list = [EvalPieces.segmentation(raw) for raw in raw_list]

    def segmentation(piece: str) -> typing.List[str]:
        doc = nlp_spacy(piece)
        doc_sents = [sent.text for sent in doc.sents]
        return doc_sents

# def pieces_gen(raw_pieces: typing.List[str]) -> EvalPieces:
#     pieces = OrderedDict()
#     [pieces.update({piece: segmentation(piece)}) for piece in raw_pieces]
#     return pieces


# def pieces_raw_list(pieces: EvalPieces):
#     return list(pieces.keys())


# def pieces_segments_list(pieces: EvalPieces):
#     return list(pieces.values())
