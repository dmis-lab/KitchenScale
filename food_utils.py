_decimals = [
    0.0,
     0.0625,
     0.125,
     0.25,
     0.333,
     0.375,
     0.5,
     0.625,
     0.666,
     0.75,
     0.875,
    1.0,
]


__additional_tokens__= """
[SEP2]
[NUM]
[NUM_MASK]
"""


__unit_keys__ = """
[None]
cup
teaspoon
tablespoon
can
lb
pinch
package
g
ounce
dash
bunch
l
ml
jar
clove
quart
box
inch
bag
pint
stalk
container
bottle
sprig
loaf
kg
slice
packet
envelope
piece
carton
pound
drop
gallon
stick
sheet
scoop
"""

unit_convert_dict = {
    'count': {
        '[None]': 1.0,  # default is None
        None: 1.0,  # default is None
    },
    'weight': {
        'g': 1.,
        'kg': 1000.0,  # default is kg
        'pound': 453.59237,  # https://en.wikipedia.org/wiki/Pound_(mass)
        # https://en.wikipedia.org/wiki/Pound_(mass) , lb == pound
        'lb': 453.59237,
        'ounce': 28.3 ,  # https://en.wikipedia.org/wiki/Ounce
    },
    'amount': {
        # https://en.wikipedia.org/wiki/United_States_customary_units
        # https://en.wikipedia.org/wiki/Cooking_weights_and_measures
        'ml': 1.0,  # default is ml
        'tablespoon': 14.79,
        'cup': 236.59,
        'teaspoon': 4.93,
        'pinch': 0.231043,
        'pint': 473.176,
        'quart': 946.35,
        'dash': 0.462086,
        'drop': 0.0513429,
        'gallon': 3785.41,
        'l': 1000.,
    },
    'others': {
        'can': 1.0,  # ? can .. noise? https://www.thespruceeats.com/can-sizes-for-recipes-4077057
        'slice': 1.0,
        'jar': 1.0,
        'package': 1.0,
        'inch': 1.0,
        'bag': 1.0,
        'clove': 1.0,  # error case? https://en.wikipedia.org/wiki/Clove
        'sprig': 1.0,  # ?
        # vary size.. https://en.wikipedia.org/wiki/Scoop_(utensil)
        'scoop': 1.0,
        'loaf': 1.0,  # bread?
        'container': 1.0,
        'stalk': 1.0,  # ?
        'sheet': 1.0,  # ?
        'piece': 1.0,
        'packet': 1.0,
        'box': 1.0,
        'bunch': 1.0,
        'bottle': 1.0,
        'carton': 1.0,  # same as box?
        'envelope': 1.0,
        'stick': 1.0,
    }
}

unit_2_unit_cat_dict = {}
unit_2_normalize_factor_dict = {}
unit_2_normalize_factor_except_others_dict = {}

for k, v in unit_convert_dict.items():
    for ik, iv in v.items():
        unit_2_unit_cat_dict[ik] = k
        unit_2_normalize_factor_dict[ik] = iv
        if k != 'others':
            unit_2_normalize_factor_except_others_dict[ik] = iv 


def get_units(is_include_none=False, is_include_others=False):
    unit_keys = __unit_keys__.split('\n')
    units = [] 
    for unit in unit_keys:
        if not is_include_none and ( unit == '[None]' or unit is None):
            continue
        if not is_include_others and unit in unit_convert_dict['others']:
            continue

        ## TODO : Don't know why? 1?
        if len(unit) > 0:
            units.append(unit)
    return units

def get_unit_dict(is_include_none=False, is_include_others=False):
    units = get_units(is_include_none, is_include_others)
    res = { u:i  for i, u in enumerate(units) }
    if is_include_none:
        ni = res['[None]']
        res[None] = ni
    return res

def get_unit_cat_dict(is_include_none=False, is_include_others=False):
    units = get_units(is_include_none, is_include_others)
    cat_set = set()
    for u in units:
        cat_set.add(unit_2_unit_cat_dict[u])
    cat_list = list(cat_set)

    cat_num_dict = { c:i  for i, c in enumerate(cat_list) }
    unit_cat_dict = unit_2_unit_cat_dict
    unit_num_dict = {u: cat_num_dict[unit_cat_dict[u]] for u in units}

    return unit_num_dict, cat_num_dict, unit_cat_dict, cat_list 

def get_additonal_tokens():
    additional_keys = __additional_tokens__.split('\n')
    tokens = [] 
    for add in additional_keys:
        if len(add) > 1:
            tokens.append(add)

    tokens.extend(get_units())
    return tokens

_special_tokens = get_additonal_tokens()



tags = [
    'easy',
    'cuisine',
    'occasion',
    'north-american',
    'fruit',
    'american',
    'dinner-party',
    'seasonal',
    'taste-mood',
    'sweet',
    '60-minutes-or-less',
    'beef',
    'meat',
    'tropical-fruit',
    'appetizers',
    'romantic',
    'mango',
    'spring',
    'steaks',
    'californian',
    '30-minutes-or-less',
    'vegetables',
    'citrus',
    'number-of-servings',
    'for-1-or-2',
    'lunch',
    'greens',
    'diabetic',
    'south-west-pacific',
    'australian',
    'lime',
    'lettuces',
    '4-hours-or-less',
    'seafood',
    'shellfish',
    'asian',
    'chinese',
    'shrimp',
    'refrigerator',
    'beginner-cook',
    'potluck',
    'to-go',
    'presentation',
    'served-cold',
    '15-minutes-or-less',
    'dips',
    'mexican',
    'spicy',
    'for-large-groups',
    'weeknight',
    'main-dish',
    'holiday-event',
    'sandwiches',
    'superbowl',
    'european',
    'italian',
    'eggplant',
    'soups-stews',
    'potatoes',
    'stove-top',
    'eggs-dairy',
    'bisques-cream-soups',
    'winter',
    'kid-friendly',
    'ground-beef',
    'oven',
    'casseroles',
    'english',
    'breads',
    'quick-breads',
    'breakfast',
    'nuts',
    'muffins',
    'canadian',
    'british-columbian',
    'no-cook',
    'picnic',
    'inexpensive',
    'comfort-food',
    'vegetarian',
    'broccoli',
    'spreads',
    'cheese',
    'condiments-etc',
    'southern-united-states',
    'southwestern-united-states',
    'st-patricks-day',
    'valentines-day',
    'oamc-freezer-make-ahead',
    'easter',
    'christmas',
    'independence-day',
    'food-processor-blender',
    'small-appliance',
    'ramadan',
    '5-ingredients-or-less',
    'vegan',
    'grains',
    'pasta-rice-and-grains',
    'savory',
    'sauces',
    'savory-sauces',
    'south-american',
    'colombian',
    'toddler-friendly',
    'free-of-something',
    'high-calcium',
    'high-in-something',
    'egg-free',
    'brown-bag',
    'rolls-biscuits',
    'new-zealand',
    'carrots',
    'fall',
    'one-dish-meal',
    'kosher',
    'onions',
    'gifts',
    'brunch',
    'low-protein',
    'healthy',
    'beverages',
    'cocktails',
    'low-fat',
    'low-cholesterol',
    'low-saturated-fat',
    'low-sodium',
    'african',
    'south-african',
    'no-shell-fish',
    'squid',
    'caribbean',
    'central-american',
    'finger-food',
    'deep-fry',
    'marinades-and-rubs',
    'new-years',
    'kwanzaa',
    'poultry',
    'spanish',
    'served-hot',
    'chicken',
    'chicken-thighs-legs',
    'chicken-breasts',
    'eggs',
    'barbecue',
    'grilling',
    'thanksgiving',
    'pacific-northwest',
    '3-steps-or-less',
    'salads',
    'side-dishes',
    'low-carb',
    'low-calorie',
    'northeastern-united-states',
    'gluten-free',
    'mussels',
    'steam',
    'french',
    'peppers',
    'from-scratch',
    'pasta',
    'middle-eastern',
    'somalian',
    'hawaiian',
    'beans',
    'turkey',
    'microwave',
    'healthy-2',
    'desserts',
    'puddings-and-mousses',
    'very-low-carbs',
    'high-protein',
    'omelets-and-frittatas',
    'fish',
    'saltwater-fish',
    'scandinavian',
    'burgers',
    'spinach',
    'german',
    'lamb-sheep',
    'flat-shapes',
    'lasagna',
    'crock-pot-slow-cooker',
    'greek',
    'heirloom-historical',
    'cookies-and-brownies',
    'swedish',
    'indian',
    'summer',
    'papaya',
    'rice',
    'short-grain-rice',
    'moroccan',
    'pies-and-tarts',
    'pies',
    'midwestern',
    'apples',
    'baking',
    'pork',
    'oranges',
    'jewish-ashkenazi',
    'passover',
    'cheesecake',
    'russian',
    'steak',
    'black-beans',
    'hand-formed-cookies',
    'egyptian',
    'stir-fry',
    'pork-sausage',
    'polish',
    'clear-soups',
    'beef-sausage',
    'chowders',
    'mushrooms',
    'tex-mex',
    'novelty',
    'pet-food',
    'bizarre',
    'freezer',
    'yams-sweet-potatoes',
    'beef-organ-meats',
    'beef-liver',
    'freshwater-fish',
    'tilapia',
    'whitefish',
    'bar-cookies',
    'stews',
    'pressure-cooker',
    'tomatoes',
    'green-yellow-beans',
    'salsas',
    'veggie-burgers',
    'squash',
    'camping',
    'chick-peas-garbanzos',
    'mardi-gras-carnival',
    'baja',
    'lentils',
    'white-rice',
    'duck',
    'duck-breasts',
    'japanese',
    'melons',
    'bass',
    'spaghetti',
    'chocolate',
    'ecuadorean',
    'non-alcoholic',
    'curries',
    'lemon',
    'brownies',
    'long-grain-rice',
    'indonesian',
    'snacks',
    'puerto-rican',
    'cuban',
    'leftovers',
    'thai',
    'frozen-desserts',
    'pitted-fruit',
    'copycat',
    'chili',
    'ontario',
    'hungarian',
    'filipino',
    'pineapple',
    'smoothies',
    'chilean',
    'savory-pies',
    'swiss',
    'yeast',
    'clams',
    'simply-potatoes',
    'irish',
    'pasta-shells',
    'sweet-sauces',
    'cauliflower',
    'plums',
    'birthday',
    '1-day-or-more',
    'finnish',
    'corn',
    'pakistani',
    'whole-chicken',
    'oaxacan',
    'cakes',
    'crusts-pastry-dough-2',
    'asparagus',
    'ravioli-tortellini',
    'coconut',
    'brazilian',
    'pork-loins',
    'stocks',
    'celebrity',
    'hanukkah',
    'jewish-sephardi',
    'roast-beef',
    'pears',
    'bacon',
    'scottish',
    'lactose',
    'tuna',
    'tarts',
    'wild-game',
    'deer',
    'nut-free',
    'penne',
    'berries',
    'cherries',
    'scones',
    'biscotti',
    'manicotti',
    'canning',
    'water-bath',
    'cod',
    'halibut',
    'wings',
    'roast',
    'laotian',
    'candy',
    'brown-rice',
    'broil',
    'cupcakes',
    'drop-cookies',
    'zucchini',
    'vietnamese',
    'portuguese',
    'pancakes-and-waffles',
    'bananas',
    'brewing',
    'mixer',
    'saudi-arabian',
    'strawberries',
    'blueberries',
    'raspberries',
    'soy-tofu',
    'turkey-breasts',
    'turkey-burgers',
    'wedding',
    'icelandic',
    'korean',
    'belgian',
    'creole',
    'fudge',
    'welsh',
    'bok-choys',
    'pork-chops',
    'salad-dressings',
    'pork-ribs',
    'cake-fillings-and-frostings',
    'danish',
    'cooking-mixes',
    'turkish',
    'crab',
    'punch',
    'salmon',
    'guatemalan',
    'chutneys',
    'peaches',
    'elbow-macaroni',
    'chicken-livers',
    'scallops',
    'pizza',
    'granola-and-porridge',
    'iranian-persian',
    'lebanese',
    'szechuan',
    'norwegian',
    'palestinian',
    'herb-and-spice-mixes',
    'cambodian',
    'ham',
    'cinco-de-mayo',
    'cobblers-and-crisps',
    'veal',
    'iraqi',
    'sudanese',
    'austrian',
    'hunan',
    'ragu-recipe-contest',
    'jams-and-preserves',
    'cajun',
    'rolled-cookies',
    'bread-machine',
    'kiwifruit',
    'cantonese',
    'dutch',
    'rosh-hashanah',
    'high-fiber',
    'peruvian',
    'czech',
    'orange-roughy',
    'halloween',
    'coffee-cakes',
    'stuffings-dressings',
    'chard',
    'native-american',
    'dairy-free',
    'garnishes',
    'peanut-butter',
    'malaysian',
    'infant-baby-friendly',
    'avocado',
    'artichoke',
    'shakes',
    'ethiopian',
    'quebec',
    'lobster',
    'meatballs',
    'angolan',
    'medium-grain-rice',
    'collard-greens',
    'trout',
    'sole-and-flounder',
    'venezuelan',
    'beef-ribs',
    'oysters',
    'mashed-potatoes',
    'crawfish',
    'fathers-day',
    'memorial-day',
    'gelatin',
    'pot-pie',
    'polynesian',
    'mothers-day',
    'dehydrator',
    'libyan',
    'catfish',
    'amish-mennonite',
    'pennsylvania-dutch',
    'pumpkin',
    'rosh-hashana',
    'a1-sauce',
    'honduran',
    'grapes',
    'tempeh',
    'ice-cream',
    'nepalese',
    'nigerian',
    'georgian',
    'super-bowl',
    'hidden-valley-ranch',
    'gumbo',
    'goose',
    'labor-day',
    'smoker',
    'whole-turkey',
    'mahi-mahi',
    'sourdough',
    'reynolds-wrap',
    'chinese-new-year',
    'meatloaf',
    'costa-rican',
    'quiche',
    'beijing',
    'bear',
    'mongolian',
    'moose',
    'rabbit',
    'soul',
    'argentine',
    'octopus',
    'pheasant',
    'micro-melanesia',
    'perch',
    'whole-duck',
    'oatmeal',
    'jellies',
    'macaroni-and-cheese'
]

tag2id_dict = {
    t:i for i, t in enumerate(tags)
}
id2tag_dict = {
    i:t for i, t in enumerate(tags)
}

