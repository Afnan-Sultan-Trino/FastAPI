# ontology_loader.py
#
# Loads Food_Nutritional_Knowledge_Graph.ttl and builds Python dictionaries
# for use in FastAPI. Uses rdflib instead of owlready2 because rdflib
# natively supports Turtle (.ttl); owlready2 does not.

from rdflib import Graph, Namespace, RDF, RDFS
import os

# ---- 1. Load the ontology -------------------------------------------------
TTL_PATH = os.path.join(os.path.dirname(__file__), "Food_Nutritional_Knowledge_Graph.ttl")

g = Graph()
g.parse(TTL_PATH, format="turtle")

NS = Namespace("http://www.semanticweb.org/as/ontologies/2026/6/Food_Nutrition_Graph/")

DATA_PROPS = [
    "hasProteinContent_g_per_100g", "hasIronContent_mg_per_100g",
    "hasVitaminC_mg_per_100g", "hasVitaminD_mcg_per_100g",
    "hasVitaminA_mcg_per_100g", "hasCalcium_mg_per_100g",
    "hasZincContent_mg_per_100g", "hasCarbohydrate_g_per_100g",
    "hasFatContent_g_per_100g", "hasServingSize_g",
]

# ---- 2. Output containers --------------------------------------------------
FOOD_NUTRIENTS = {}   # food name -> {property_name: value}
FOOD_CATEGORY = {}    # food name -> immediate class name (e.g. "Fish")
RDA_VALUES = {}       # nutrient name -> {"male": x, "female": y}
SYMPTOM_MAP = {}      # nutrient name -> [symptom names]
TREATMENT_MAP = {}    # nutrient name -> [food names]


def local_name(uri):
    """Get the short name after the last '/' or '#' in a URI."""
    return str(uri).split("/")[-1].split("#")[-1]


# ---- 3. Find all Food individuals and their properties ---------------------
food_class = NS.Food
for food_uri in g.subjects(RDF.type, food_class) or []:
    pass  # placeholder - real query below handles subclasses too

# Foods are typed with a SUBCLASS of :Food (e.g. :Fish, :Grain), not :Food
# itself, so we first collect all subclasses of :Food, then find their
# instances.
food_subclasses = {food_class}
changed = True
while changed:
    changed = False
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        if o in food_subclasses and s not in food_subclasses:
            food_subclasses.add(s)
            changed = True

food_individuals = set()
for cls in food_subclasses:
    for s in g.subjects(RDF.type, cls):
        food_individuals.add(s)

for food_uri in food_individuals:
    food_name = local_name(food_uri)

    # immediate category = the most specific asserted class that isn't Food itself
    categories = [local_name(o) for o in g.objects(food_uri, RDF.type)
                  if o != food_class and "NamedIndividual" not in str(o)]
    FOOD_CATEGORY[food_name] = categories[0] if categories else "Food"

    nutrient_dict = {}
    for prop in DATA_PROPS:
        prop_uri = NS[prop]
        for val in g.objects(food_uri, prop_uri):
            nutrient_dict[prop] = float(val)
            break  # only one value expected per property
    if nutrient_dict:
        FOOD_NUTRIENTS[food_name] = nutrient_dict

# ---- 4. Find all Nutrient individuals: RDA, causesSymptom, treatedBy -------
nutrient_class = NS.Nutrient
nutrient_subclasses = {nutrient_class}
changed = True
while changed:
    changed = False
    for s, p, o in g.triples((None, RDFS.subClassOf, None)):
        if o in nutrient_subclasses and s not in nutrient_subclasses:
            nutrient_subclasses.add(s)
            changed = True

nutrient_individuals = set()
for cls in nutrient_subclasses:
    for s in g.subjects(RDF.type, cls):
        nutrient_individuals.add(s)

for nutrient_uri in nutrient_individuals:
    n_name = local_name(nutrient_uri)

    rda_male = list(g.objects(nutrient_uri, NS.hasRDA_male_daily))
    rda_female = list(g.objects(nutrient_uri, NS.hasRDA_female_daily))
    if rda_male or rda_female:
        RDA_VALUES[n_name] = {
            "male": float(rda_male[0]) if rda_male else None,
            "female": float(rda_female[0]) if rda_female else None,
        }

    symptoms = [local_name(o) for o in g.objects(nutrient_uri, NS.causesSymptom)]
    if symptoms:
        SYMPTOM_MAP[n_name] = symptoms

    treatments = [local_name(o) for o in g.objects(nutrient_uri, NS.treatedBy)]
    if treatments:
        TREATMENT_MAP[n_name] = treatments

# ---- 4b. enhancesAbsorption is Food -> Nutrient, so we scan all foods -----
ENHANCEMENT_MAP = {}  # nutrient name -> [food names that enhance its absorption]
RESTRICTED_MAP = {}   # food name -> [religion names that commonly avoid this food]
for food_uri in food_individuals:
    food_name = local_name(food_uri)
    for nutrient_uri in g.objects(food_uri, NS.enhancesAbsorption):
        n_name = local_name(nutrient_uri)
        ENHANCEMENT_MAP.setdefault(n_name, []).append(food_name)
    religions = [local_name(o) for o in g.objects(food_uri, NS.restrictedByReligion)]
    if religions:
        RESTRICTED_MAP[food_name] = religions

NATURAL_UNIT_LABELS = {}  # food name -> human-friendly countable unit label (string), if defined
for food_uri in food_individuals:
    food_name = local_name(food_uri)
    labels = list(g.objects(food_uri, NS.hasNaturalUnitLabel))
    if labels:
        NATURAL_UNIT_LABELS[food_name] = str(labels[0])


# ---- 4c. Reproduce the OWL inference in plain Python -----------------------
# IMPORTANT: rdflib does NOT run an OWL reasoner (no HermiT/Pellet). The
# ontology defines :isGoodSourceFor as owl:inverseOf :treatedBy, and
# :helpsWithSymptom as a property chain (:isGoodSourceFor :causesSymptom),
# but those inferred triples only materialize inside a real DL reasoner
# (e.g. when you click "Start reasoner" in Protege, or run HermiT via
# owlready2). Since our FastAPI backend loads the ttl with plain rdflib,
# we recreate the SAME two inference steps here in Python -- this keeps
# the backend lightweight (no Java/reasoner dependency at request time)
# while staying logically identical to what the ontology asserts.

# Step 1 (mirrors "isGoodSourceFor owl:inverseOf treatedBy"):
# invert TREATMENT_MAP (nutrient -> foods) into food -> nutrients
GOOD_SOURCE_MAP = {}  # food name -> [nutrient names this food is a good source of]
for nutrient_name, foods in TREATMENT_MAP.items():
    for food_name in foods:
        GOOD_SOURCE_MAP.setdefault(food_name, []).append(nutrient_name)

# Step 2 (mirrors "helpsWithSymptom = isGoodSourceFor . causesSymptom"):
# chain GOOD_SOURCE_MAP through SYMPTOM_MAP
HELPS_WITH_SYMPTOM_MAP = {}  # food name -> [symptom names this food helps with]
for food_name, nutrients in GOOD_SOURCE_MAP.items():
    symptoms_for_food = []
    for nutrient_name in nutrients:
        symptoms_for_food.extend(SYMPTOM_MAP.get(nutrient_name, []))
    if symptoms_for_food:
        # dict.fromkeys() dedupes while preserving order (a food can reach
        # the same symptom via more than one nutrient)
        HELPS_WITH_SYMPTOM_MAP[food_name] = list(dict.fromkeys(symptoms_for_food))


# ---- 4d. Reverse map: symptom -> nutrients that can cause it --------------
# Same "invert then reuse" pattern as GOOD_SOURCE_MAP above. Lets
# deficiency.py cross-check a person's self-reported symptoms against the
# intake-based deficiency calculation (does the evidence agree?).
SYMPTOM_TO_NUTRIENTS = {}  # symptom name -> [nutrient names that can cause it]
for nutrient_name, symptoms in SYMPTOM_MAP.items():
    for symptom_name in symptoms:
        SYMPTOM_TO_NUTRIENTS.setdefault(symptom_name, []).append(nutrient_name)

# ---- 5. Debug output when run directly -------------------------------------
if __name__ == "__main__":
    print("Total foods loaded:", len(FOOD_NUTRIENTS))
    print("Total nutrients with RDA:", len(RDA_VALUES))
    print("Sample food (Ilish):", FOOD_NUTRIENTS.get("Ilish", {}))
    print("Sample category (Ilish):", FOOD_CATEGORY.get("Ilish"))
    print("Sample RDA (Iron):", RDA_VALUES.get("Iron", {}))
    print("Sample symptoms (Iron):", SYMPTOM_MAP.get("Iron", []))
    print("Sample treatment (Iron):", TREATMENT_MAP.get("Iron", []))
    print("Sample enhancer (Iron):", ENHANCEMENT_MAP.get("Iron", []))
    print("Foods with religious restrictions:", RESTRICTED_MAP)
    print("PalongShak good source for:", GOOD_SOURCE_MAP.get("PalongShak", []))
    print("PalongShak helps with symptoms:", HELPS_WITH_SYMPTOM_MAP.get("PalongShak", []))
    print("Nutrients causing 'Fatigue':", SYMPTOM_TO_NUTRIENTS.get("Fatigue", []))

    # ---- Coverage check against the 22-food survey list (data_importer.py /
    # main.py's BANGLA_TO_ONTOLOGY). Run this after adding the new foods to
    # confirm the .ttl actually has data-property values for all of them --
    # any food missing here will silently contribute 0 intake in
    # deficiency.py, with no error raised anywhere.
    EXPECTED_ONTOLOGY_FOODS = [
        "Ilish", "Rui", "GorurMangsho", "MurgirMangsho", "Dim", "Dudh", "Doi",
        "DudhCha", "LalShak", "PalongShak", "Begun", "Alu", "Gajar", "MishtiKumra",
        "Aam", "Kamalalebu", "Amloki", "Pepe", "Chola", "MasoorDal", "Atta", "Bhat",
    ]
    missing = [f for f in EXPECTED_ONTOLOGY_FOODS if f not in FOOD_NUTRIENTS]
    if missing:
        print(f"\n⚠️  {len(missing)} food(s) expected by the survey but NOT found in the ontology:", missing)
        print("   These will silently show as 0 intake in deficiency.py until added to the .ttl.")
    else:
        print("\n✅ All 22 survey foods are present in FOOD_NUTRIENTS.")