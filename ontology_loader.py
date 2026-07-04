# ontology_loader.py
#
# Loads Food_Nutritional_Knowledge_Graph.ttl and builds Python dictionaries
# for use in FastAPI. Uses rdflib instead of owlready2 because rdflib
# natively supports Turtle (.ttl); owlready2 does not.

from rdflib import Graph, Namespace, RDF, RDFS
import os
from pathlib import Path 

# ---- 1. Load the ontology -------------------------------------------------
TTL_PATH = Path(__file__).parent / "Food_Nutritional_Knowledge_Graph.ttl"

g = Graph()
with open(TTL_PATH, "rb") as f:
    g.parse(f, format="turtle")

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
for food_uri in food_individuals:
    food_name = local_name(food_uri)
    for nutrient_uri in g.objects(food_uri, NS.enhancesAbsorption):
        n_name = local_name(nutrient_uri)
        ENHANCEMENT_MAP.setdefault(n_name, []).append(food_name)


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