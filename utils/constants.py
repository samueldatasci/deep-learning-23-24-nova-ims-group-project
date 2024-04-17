
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)  # although not ideal this size greatly reduces the training time
INPUT_SHAPE = IMAGE_SIZE + (3,)  # plus 3 because images are rgb

CLASS_INT_TO_STR = {
    0: 'acanthosis nigricans',
    1: 'acne',
    2: 'acne vulgaris',
    3: 'acquired autoimmune bullous diseaseherpes gestationis',
    4: 'acrodermatitis enteropathica',
    5: 'actinic keratosis',
    6: 'allergic contact dermatitis',
    7: 'aplasia cutis',
    8: 'basal cell carcinoma',
    9: 'basal cell carcinoma morpheiform',
    10: 'becker nevus',
    11: 'behcets disease',
    12: 'calcinosis cutis',
    13: 'cheilitis',
    14: 'congenital nevus',
    15: 'dariers disease',
    16: 'dermatofibroma',
    17: 'dermatomyositis',
    18: 'disseminated actinic porokeratosis',
    19: 'drug eruption',
    20: 'drug induced pigmentary changes',
    21: 'dyshidrotic eczema',
    22: 'eczema',
    23: 'ehlers danlos syndrome',
    24: 'epidermal nevus',
    25: 'epidermolysis bullosa',
    26: 'erythema annulare centrifigum',
    27: 'erythema elevatum diutinum',
    28: 'erythema multiforme',
    29: 'erythema nodosum',
    30: 'factitial dermatitis',
    31: 'fixed eruptions',
    32: 'folliculitis',
    33: 'fordyce spots',
    34: 'granuloma annulare',
    35: 'granuloma pyogenic',
    36: 'hailey hailey disease',
    37: 'halo nevus',
    38: 'hidradenitis',
    39: 'ichthyosis vulgaris',
    40: 'incontinentia pigmenti',
    41: 'juvenile xanthogranuloma',
    42: 'kaposi sarcoma',
    43: 'keloid',
    44: 'keratosis pilaris',
    45: 'langerhans cell histiocytosis',
    46: 'lentigo maligna',
    47: 'lichen amyloidosis',
    48: 'lichen planus',
    49: 'lichen simplex',
    50: 'livedo reticularis',
    51: 'lupus erythematosus',
    52: 'lupus subacute',
    53: 'lyme disease',
    54: 'lymphangioma',
    55: 'malignant melanoma',
    56: 'melanoma',
    57: 'milia',
    58: 'mucinosis',
    59: 'mucous cyst',
    60: 'mycosis fungoides',
    61: 'myiasis',
    62: 'naevus comedonicus',
    63: 'necrobiosis lipoidica',
    64: 'nematode infection',
    65: 'neurodermatitis',
    66: 'neurofibromatosis',
    67: 'neurotic excoriations',
    68: 'neutrophilic dermatoses',
    69: 'nevocytic nevus',
    70: 'nevus sebaceous of jadassohn',
    71: 'papilomatosis confluentes and reticulate',
    72: 'paronychia',
    73: 'pediculosis lids',
    74: 'perioral dermatitis',
    75: 'photodermatoses',
    76: 'pilar cyst',
    77: 'pilomatricoma',
    78: 'pityriasis lichenoides chronica',
    79: 'pityriasis rosea',
    80: 'pityriasis rubra pilaris',
    81: 'porokeratosis actinic',
    82: 'porokeratosis of mibelli',
    83: 'porphyria',
    84: 'port wine stain',
    85: 'prurigo nodularis',
    86: 'psoriasis',
    87: 'pustular psoriasis',
    88: 'pyogenic granuloma',
    89: 'rhinophyma',
    90: 'rosacea',
    91: 'sarcoidosis',
    92: 'scabies',
    93: 'scleroderma',
    94: 'scleromyxedema',
    95: 'seborrheic dermatitis',
    96: 'seborrheic keratosis',
    97: 'solid cystic basal cell carcinoma',
    98: 'squamous cell carcinoma',
    99: 'stasis edema',
    100: 'stevens johnson syndrome',
    101: 'striae',
    102: 'sun damaged skin',
    103: 'superficial spreading melanoma ssm',
    104: 'syringoma',
    105: 'telangiectases',
    106: 'tick bite',
    107: 'tuberous sclerosis',
    108: 'tungiasis',
    109: 'urticaria',
    110: 'urticaria pigmentosa',
    111: 'vitiligo',
    112: 'xanthomas',
    113: 'xeroderma pigmentosum'
}

# FYI: non-neoplastic == A non-cancerous, non-malignant, or benign disease or lesion
# therefore, if malign => 1, if benign or non-neoplastic => 0
MALIGN_DISEASES = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 0,
    7: 0,
    8: 1,
    9: 1,
    10: 0,
    11: 0,
    12: 0,
    13: 0,
    14: 0,
    15: 0,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
    21: 0,
    22: 0,
    23: 0,
    24: 0,
    25: 0,
    26: 0,
    27: 0,
    28: 0,
    29: 0,
    30: 0,
    31: 0,
    32: 0,
    33: 0,
    34: 0,
    35: 0,
    36: 0,
    37: 0,
    38: 0,
    39: 0,
    40: 0,
    41: 0,
    42: 1,
    43: 0,
    44: 0,
    45: 0,
    46: 1,
    47: 0,
    48: 0,
    49: 0,
    50: 0,
    51: 0,
    52: 0,
    53: 0,
    54: 0,
    55: 1,
    56: 1,
    57: 0,
    58: 0,
    59: 0,
    60: 1,
    61: 0,
    62: 0,
    63: 0,
    64: 0,
    65: 0,
    66: 0,
    67: 0,
    68: 0,
    69: 0,
    70: 0,
    71: 0,
    72: 0,
    73: 0,
    74: 0,
    75: 0,
    76: 0,
    77: 0,
    78: 0,
    79: 0,
    80: 0,
    81: 0,
    82: 0,
    83: 0,
    84: 0,
    85: 0,
    86: 0,
    87: 0,
    88: 0,
    89: 0,
    90: 0,
    91: 0,
    92: 0,
    93: 0,
    94: 0,
    95: 0,
    96: 0,
    97: 1,
    98: 1,
    99: 0,
    100: 0,
    101: 0,
    102: 0,
    103: 1,
    104: 0,
    105: 0,
    106: 0,
    107: 0,
    108: 0,
    109: 0,
    110: 0,
    111: 0,
    112: 0,
    113: 0
}