"""
Trika eval data — held-out Śiva Sūtra + Spanda Kārikā verses for benchmark evaluation.

Neither text was seen during training (VBT / Itihāsa), so these provide an
uncontaminated test set for cross-lingual retrieval, STS correlation, and
triplet discrimination.
"""

# ---------------------------------------------------------------------------
# Śiva Sūtra (शिवसूत्र) — Vasugupta, 77 sūtras in 3 sections
# ---------------------------------------------------------------------------
SIVA_SUTRA = {
    "SS_1.1":  "चैतन्यमात्मा",
    "SS_1.2":  "ज्ञानं बन्धः",
    "SS_1.5":  "उद्यमो भैरवः",
    "SS_1.6":  "शक्तिचक्रसन्धाने विश्वसंहारः",
    "SS_1.7":  "जाग्रत्स्वप्नसुषुप्तभेदे तुर्याभोगसम्भवः",
    "SS_1.12": "विस्मयो योगभूमिकाः",
    "SS_1.13": "इच्छाशक्तिरुमा कुमारी",
    "SS_1.17": "वितर्क आत्मज्ञानम्",
    "SS_1.18": "लोकानन्दः समाधिसुखम्",
    "SS_1.22": "महाह्रदानुसन्धानान्मन्त्रवीर्यानुभवः",
    "SS_2.1":  "चित्तं मन्त्रः",
    "SS_2.5":  "विद्यासमुत्थाने स्वाभाविके खेचरी शिवावस्था",
    "SS_2.6":  "गुरुरुपायः",
    "SS_3.9":  "नर्तक आत्मा",
    "SS_3.12": "धीवशात् सत्त्वसिद्धिः",
    "SS_3.21": "त्रिषु चतुर्थं तैलवदासेच्यम्",
    "SS_3.26": "शिवतुल्यो जायते",
    "SS_3.33": "तत्प्रवृत्तावप्यनिरासः संवेत्तृभावात्",
    "SS_3.43": "भूतकञ्चुकी तदा विमुक्तो भूयः पतिसमः परः",
    "SS_3.45": "नासिकान्तर्मध्यसंयमात् किमत्र सव्यापसव्यसौषुम्नेषु",
}

SIVA_SUTRA_EN = {
    "SS_1.1":  "Consciousness is the Self.",
    "SS_1.2":  "Limited knowledge is bondage.",
    "SS_1.5":  "The upsurge of consciousness is Bhairava.",
    "SS_1.6":  "By union with the circle of powers, the universe is withdrawn.",
    "SS_1.7":  "Even during the differentiation of waking, dream, and deep sleep, the Fourth state expands.",
    "SS_1.12": "The stages of yoga are a wonder.",
    "SS_1.13": "The power of will is the playful maiden Uma.",
    "SS_1.17": "Wordless discernment is the knowledge of the Self.",
    "SS_1.18": "The bliss of the world is the joy of samadhi.",
    "SS_1.22": "By merging with the great lake of consciousness, the power of mantra is experienced.",
    "SS_2.1":  "The mind is mantra.",
    "SS_2.5":  "When knowledge of one's Self arises naturally, one moves in the sky of consciousness — the state of Siva.",
    "SS_2.6":  "The guru is the means.",
    "SS_3.9":  "The Self is the dancer.",
    "SS_3.12": "The pure state is achieved by the power of the intellect.",
    "SS_3.21": "The fourth state should be poured like oil into the other three.",
    "SS_3.26": "One becomes equal to Siva.",
    "SS_3.33": "Even during activity there is no break in awareness, because of the state of being the knower.",
    "SS_3.43": "Although cloaked in the elements, one is then free, supreme, like the Lord.",
    "SS_3.45": "Concentrating on the centre within the nose — what use then are the left, right, and central channels?",
}

# ---------------------------------------------------------------------------
# Spanda Kārikā (स्पन्दकारिका) — Vasugupta / Kallaṭa, 52 verses
# ---------------------------------------------------------------------------
SPANDA_KARIKA = {
    "SK_1":  "यस्योन्मेषनिमेषाभ्यां जगतः प्रलयोदयौ / तं शक्तिचक्रविभवप्रभवं शङ्करं स्तुमः",
    "SK_2":  "यत्र स्थितमिदं सर्वं कार्यं यस्माच्च निर्गतम् / तस्यानावृतरूपत्वान्न निरोधोऽस्ति कुत्रचित्",
    "SK_3":  "जाग्रदादिविभेदेऽपि तदभिन्ने प्रसर्पति / निवर्तते निजान्नैव स्वभावादुपलब्धृतः",
    "SK_5":  "न दुःखं न सुखं यत्र न ग्राह्यं ग्राहकं न च / न चास्ति मूढभावोऽपि तदस्ति परमार्थतः",
    "SK_9":  "निजाशुद्ध्यासमर्थस्य कर्तव्येष्वभिलाषिणः / यदा क्षोभः प्रलीयेत तदा स्यात्परमं पदम्",
    "SK_11": "तमधिष्ठातृभावेन स्वभावमवलोकयन् / स्मयमान इवास्ते यस्तस्येयं कुसृतिः कुतः",
    "SK_17": "तस्योपलब्धिः सततं त्रिपदाव्यभिचारिणी / नित्यं स्यात्सुप्रबुद्धस्य तदाद्यन्ते परस्य तु",
    "SK_21": "अतः सततमुद्युक्तः स्पन्दतत्त्वविविक्तये / जाग्रदेव निजं भावमचिरेणाधिगच्छति",
    "SK_22": "अतिक्रुद्धः प्रहृष्टो वा किं करोमीति वा मृशन् / धावन्वा यत्पदं गच्छेत्तत्र स्पन्दः प्रतिष्ठितः",
    "SK_30": "इति वा यस्य संवित्तिः क्रीडात्वेनाखिलं जगत् / स पश्यन्सततं युक्तो जीवन्मुक्तो न संशयः",
    "SK_44": "प्रबुद्धः सर्वदा तिष्ठेज्ज्ञानेनालोक्य गोचरम् / एकत्रारोपयेत्सर्वं ततोऽन्येन न पीड्यते",
    "SK_48": "सैषा क्रियात्मिका शक्तिः शिवस्य पशुवर्तिनी / बन्धयित्री स्वमार्गस्था ज्ञाता सिद्ध्युपपादिका",
}

SPANDA_KARIKA_EN = {
    "SK_1":  "We praise Sankara, the source of the power of the wheel of energies, by whose opening and closing of the eyes the world dissolves and arises.",
    "SK_2":  "Because His nature is unobstructed, in whom all this rests and from whom all has come forth, there is no obstruction anywhere.",
    "SK_3":  "The Spanda principle continues to flow undivided even in the differentiation of waking and other states, and never departs from its own essential nature as the Perceiver.",
    "SK_5":  "That in which there is neither pain nor pleasure, neither object nor subject, and not even insentiency — that alone exists in the highest sense.",
    "SK_9":  "When the agitation of one who is incapacitated by impurity and who desires to perform actions dissolves, then the supreme state arises.",
    "SK_11": "How could the miserable path of transmigration belong to him who, filled with wonder, gazes upon his own nature as the presiding reality?",
    "SK_17": "For the fully awakened, the perception of the Self as Spanda is constant and unfailing in all three states. For others, it is present only at the beginning and end of each state.",
    "SK_21": "Therefore one who is constantly engaged in discerning the Spanda principle attains his own essential nature quickly, even in the waking state.",
    "SK_22": "Spanda is firmly established in that state which one reaches when extremely angry, intensely joyful, wondering 'what shall I do?', or running for one's life.",
    "SK_30": "He who has the realization that the entire world is divine play, seeing thus, perpetually united — he is liberated while living, without doubt.",
    "SK_44": "One should always remain awakened, observing all phenomena through knowledge, and should deposit everything in one place. Then one is not afflicted by anything else.",
    "SK_48": "This very power of action of Siva, which abides in the bound soul as a binding force — when situated in one's own path and recognized, it brings about spiritual perfection.",
}

# ---------------------------------------------------------------------------
# STS benchmark — 21 pairs with human-annotated 0–5 similarity scores
# ---------------------------------------------------------------------------
# 5 = paraphrase/identical meaning, 4 = very similar, 3 = related,
# 2 = loosely related, 1 = vaguely related, 0 = unrelated
STS_PAIRS = [
    # --- Score 5: paraphrase / same meaning across texts ---
    ("SS_1.7", "SK_3", 5.0,
     "SS 1.7 <> SK 3: both about turiya pervading three states"),
    ("SS_1.7", "SK_17", 5.0,
     "SS 1.7 <> SK 17: turiya in three states / spanda in three states"),
    ("SS_3.26", "SK_30", 4.5,
     "SS 3.26 <> SK 30: becoming Siva / jivanmukti"),

    # --- Score 4: closely related theme ---
    ("SS_1.1", "SK_5", 4.0,
     "SS 1.1 <> SK 5: consciousness as Self / paramartha beyond duality"),
    ("SS_1.12", "SK_11", 4.0,
     "SS 1.12 <> SK 11: wonder (vismaya) / gazing with wonder (smayamana)"),
    ("SS_3.43", "SK_9", 4.0,
     "SS 3.43 <> SK 9: freedom despite elements / supreme state when agitation dissolves"),
    ("SS_1.5", "SK_22", 3.5,
     "SS 1.5 <> SK 22: upsurge of awareness / spanda in extreme states"),

    # --- Score 3: related but different aspect ---
    ("SS_1.22", "SS_2.1", 3.0,
     "SS 1.22 <> SS 2.1: mantra power / mind-as-mantra"),
    ("SS_1.5", "SS_3.9", 3.0,
     "SS 1.5 <> SS 3.9: dynamic upsurge / cosmic dancer"),
    ("SK_21", "SK_44", 3.0,
     "SK 21 <> SK 44: discerning spanda in waking / remaining awakened through knowledge"),
    ("SS_1.17", "SS_1.1", 3.0,
     "SS 1.17 <> SS 1.1: atma-jnana / caitanya as atma"),

    # --- Score 2: loosely related ---
    ("SS_2.6", "SK_48", 2.0,
     "SS 2.6 <> SK 48: guru as means / sakti as path to siddhi"),
    ("SS_1.6", "SK_2", 2.0,
     "SS 1.6 <> SK 2: sakticakra -> dissolution / unobstructed source of all"),
    ("SS_1.13", "SK_1", 2.0,
     "SS 1.13 <> SK 1: iccha-sakti as Uma / sakticakra and Sankara"),

    # --- Score 1: vaguely related (both spiritual but different topic) ---
    ("SS_1.18", "SK_22", 1.0,
     "SS 1.18 <> SK 22: bliss of samadhi / anger-joy revealing spanda"),
    ("SS_3.21", "SK_48", 1.0,
     "SS 3.21 <> SK 48: pouring turiya into states / sakti as binding force"),
    ("SS_2.5", "SK_9", 1.5,
     "SS 2.5 <> SK 9: natural khecari Siva-state / supreme state when agitation dissolves"),

    # --- Score 0: unrelated themes ---
    ("SS_1.1", "SS_3.45", 0.0,
     "SS 1.1 <> SS 3.45: consciousness-as-Self / nadi pranayama"),
    ("SS_1.18", "SK_48", 0.5,
     "SS 1.18 <> SK 48: world-bliss / binding power of sakti"),
    ("SS_3.45", "SK_30", 0.0,
     "SS 3.45 <> SK 30: nasal pranayama / world as divine play"),
    ("SS_2.6", "SS_3.45", 0.0,
     "SS 2.6 <> SS 3.45: guru as means / nasal pranayama"),
]

# ---------------------------------------------------------------------------
# Triplet discrimination — 16 triplets in 3 categories
# ---------------------------------------------------------------------------
# (label, anchor, positive, negative, category)
TRIPLETS = [
    # --- Monolingual Sa: same-theme positive, different-theme negative ---
    ("SS 1.7 <> SK 3 vs SS 3.45",
     "SS_1.7", "SK_3", "SS_3.45", "mono_sa"),
    ("SS 1.12 <> SK 11 vs SK 48",
     "SS_1.12", "SK_11", "SK_48", "mono_sa"),
    ("SS 3.26 <> SK 30 vs SS 3.45",
     "SS_3.26", "SK_30", "SS_3.45", "mono_sa"),
    ("SK 21 <> SK 44 vs SS 1.1",
     "SK_21", "SK_44", "SS_1.1", "mono_sa"),
    ("SS 1.5 <> SK 22 vs SS 2.6",
     "SS_1.5", "SK_22", "SS_2.6", "mono_sa"),
    ("SS 1.22 <> SS 2.1 vs SK 9",
     "SS_1.22", "SS_2.1", "SK_9", "mono_sa"),
    ("SK 1 <> SK 2 vs SS 3.12",
     "SK_1", "SK_2", "SS_3.12", "mono_sa"),
    ("SS 3.43 <> SK 9 vs SS 2.1",
     "SS_3.43", "SK_9", "SS_2.1", "mono_sa"),

    # --- Cross-lingual: Sa anchor, En positive (own translation), En negative (other) ---
    ("SS 1.1 Sa <> SS 1.1 En vs SS 3.45 En",
     "SS_1.1", "SS_1.1_EN", "SS_3.45_EN", "cross_lingual"),
    ("SK 22 Sa <> SK 22 En vs SK 2 En",
     "SK_22", "SK_22_EN", "SK_2_EN", "cross_lingual"),
    ("SK 30 Sa <> SK 30 En vs SS 1.2 En",
     "SK_30", "SK_30_EN", "SS_1.2_EN", "cross_lingual"),
    ("SS 3.9 Sa <> SS 3.9 En vs SK 48 En",
     "SS_3.9", "SS_3.9_EN", "SK_48_EN", "cross_lingual"),
    ("SK 5 Sa <> SK 5 En vs SS 1.18 En",
     "SK_5", "SK_5_EN", "SS_1.18_EN", "cross_lingual"),
    ("SS 1.7 Sa <> SS 1.7 En vs SK 48 En",
     "SS_1.7", "SS_1.7_EN", "SK_48_EN", "cross_lingual"),

    # --- Hard triplets: positive and negative are both somewhat related ---
    ("SS 1.1 <> SS 1.17 vs SS 3.9 (all atma)",
     "SS_1.1", "SS_1.17", "SS_3.9", "hard"),
    ("SK 3 <> SK 17 vs SK 21 (all spanda/states)",
     "SK_3", "SK_17", "SK_21", "hard"),
]


def _lookup_text(verse_id: str) -> str:
    """Resolve a verse ID (e.g. 'SS_1.1', 'SK_22_EN') to its text."""
    if verse_id.endswith("_EN"):
        base_id = verse_id[:-3]
        if base_id in SIVA_SUTRA_EN:
            return SIVA_SUTRA_EN[base_id]
        return SPANDA_KARIKA_EN[base_id]
    if verse_id in SIVA_SUTRA:
        return SIVA_SUTRA[verse_id]
    return SPANDA_KARIKA[verse_id]


def get_all_verses():
    """Return (sa_ids, sa_texts, en_ids, en_texts) for cross-lingual retrieval.

    Combines Śiva Sūtra (20) + Spanda Kārikā (12) = 32 verse pairs.
    sa_ids[i] and en_ids[i] are parallel (same verse, different language).
    """
    sa_ids = list(SIVA_SUTRA.keys()) + list(SPANDA_KARIKA.keys())
    sa_texts = [SIVA_SUTRA[k] for k in SIVA_SUTRA] + [SPANDA_KARIKA[k] for k in SPANDA_KARIKA]
    en_ids = list(SIVA_SUTRA_EN.keys()) + list(SPANDA_KARIKA_EN.keys())
    en_texts = [SIVA_SUTRA_EN[k] for k in SIVA_SUTRA_EN] + [SPANDA_KARIKA_EN[k] for k in SPANDA_KARIKA_EN]
    return sa_ids, sa_texts, en_ids, en_texts


def get_sts_pairs():
    """Return list of (text_a, text_b, human_score, label) with texts resolved."""
    resolved = []
    for id_a, id_b, score, label in STS_PAIRS:
        text_a = _lookup_text(id_a)
        text_b = _lookup_text(id_b)
        resolved.append((text_a, text_b, score, label))
    return resolved


def get_triplets():
    """Return list of (label, anchor_text, positive_text, negative_text, category)."""
    resolved = []
    for label, anchor_id, pos_id, neg_id, category in TRIPLETS:
        anchor = _lookup_text(anchor_id)
        positive = _lookup_text(pos_id)
        negative = _lookup_text(neg_id)
        resolved.append((label, anchor, positive, negative, category))
    return resolved
