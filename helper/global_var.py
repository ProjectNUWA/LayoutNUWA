# this is global vars

KEY_MULT_DICT = {
    "x-y-w-h": {"x":0, "y": 1, "w": 2, "h": 3},
    "xywh": {},
}


MULTI_CHOICE = {
    "kmeans": "x-y-w-h",
    "linear": "xywh",
    "int2str": "",
    "float2str": "",
}

SPECIAL_TOKENS = ["<MASK>", "<PAD>", "<EOS>", "<BOS>", "<SEP>"]

IGNORE_INDEX = -100

TEMPLATE_FORMAT = {
    "html_format": " <body> <svg width=\"{W}\" height=\"{H}\"> {content} </svg> </body>",
    "bbox_format": "<rect data-category=\"{c}\", x=\"{x}\", y=\"{y}\", width=\"{w}\", height=\"{h}\"/>",
}

TASK_INSTRUCTION = {
    "rico25": "I want to generate layout in the mobile app design format. ",
    "publaynet": "I want to generate layout in the document design format. ",
    "magazine": "I want to generate layout in the magazine design format. ",
}

INSTRUCTION = {
    "cond_cate_to_size_pos": "please generate the layout html according to the categories I provide (in html format):\n###bbox html: {bbox_html}",
    "cond_cate_size_to_pos": "please generate the layout html according to the categories and size I provide (in html format):\n###bbox html: {bbox_html}",
    "cond_random_mask": "please recover the layout html according to the bbox, categories and size I provide (in html format):\n###bbox html: {bbox_html}"
}

INFILLING_INSTRUCTION = {
    "cond_cate_to_size_pos": "please fulfilling the layout html according to the categories I provide (in html format):\n###bbox html: {bbox_html}",
    "cond_cate_size_to_pos": "please fulfilling the layout html according to the categories and size I provide (in html format):\n###bbox html: {bbox_html}",
    "cond_random_mask": "please recover the layout html according to the bbox, categories and size I provide (in html format):\n###bbox html: {bbox_html}"
}

SEP_SEQ = [
    "{instruct}\n\n##Here is the result:\n\n```{result}```",
    "{instruct}\n\n##Here is the result:",
    "{instruct} <MID> {result}",
    "{instruct} <MID>",
]

DATASET_META = {
    "magazine": {
        0: 'text',
        1: 'image',
        2: 'headline',
        3: 'text-over-image',
        4: 'headline-over-image',
    },
    "publaynet": {
        0: 'text',
        1: 'title',
        2: 'list',
        3: 'table',
        4: 'figure',
    },
    "rico25": {
        0: "Text",
        1: "Image",
        2: "Icon",
        3: "Text Button",
        4: "List Item",
        5: "Input",
        6: "Background Image",
        7: "Card",
        8: "Web View",
        9: "Radio Button",
        10: "Drawer",
        11: "Checkbox",
        12: "Advertisement",
        13: "Modal",
        14: "Pager Indicator",
        15: "Slider",
        16: "On/Off Switch",
        17: "Button Bar",
        18: "Toolbar",
        19: "Number Stepper",
        20: "Multi-Tab",
        21: "Date Picker",
        22: "Map View",
        23: "Video",
        24: "Bottom Navigation",
    },
}

# verborsed number
VERBALIZED_NUM = {  
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',  
    10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',  
    17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty',  
    21: 'twenty-one', 22: 'twenty-two', 23: 'twenty-three', 24: 'twenty-four', 25: 'twenty-five',  
    26: 'twenty-six', 27: 'twenty-seven', 28: 'twenty-eight', 29: 'twenty-nine', 30: 'thirty',  
    31: 'thirty-one', 32: 'thirty-two'  
}  
  
 
