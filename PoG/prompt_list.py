# No change required
subobjective_prompt = """Please break down the process of answering the question into as few subobjectives as possible based on semantic analysis.
Here is an example: 
Q: Which of the countries in the Caribbean has the smallest country calling code?
Output: ['Search the countries in the Caribbean', 'Search the country calling code for each Caribbean country', 'Compare the country calling codes to find the smallest one']

Now you need to directly output subobjectives of the following question in list format without other information or notes. 
Q: """


# In use, and contains Freebase specific relations
# Needs to be changed
extract_relation_prompt = """Please provide as few highly relevant relations as possible to the question and its subobjectives from the following relations (separated by semicolons).
Here is an example:
Q: Name the president of the country whose main spoken language was Brahui in 1980?
Subobjectives: ['Identify the countries where the main spoken language is Brahui', 'Find the president of each country', 'Determine the president from 1980']
Topic Entity: Brahui Language
Relations: yago:officialLanguage; schema:knowsLanguage; schema:inLanguage; schema:contentLocation; schema:location; yago:capital; yago:leader; yago:neighbors; schema:birthPlace; schema:address; schema:about
The output is: 
['yago:officialLanguage', 'schema:knowsLanguage', 'yago:leader']

Now you need to directly output relations highly related to the following question and its subobjectives in list format without other information or notes.
Q: """

# In use, and contains Freebase specific relations
# Needs to be changed
answer_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), you are asked to answer the question with these triplets and your knowledge.

Here are five examples:
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Knowledge Triplets: Taste cannot be controlled by law., schema:author, Thomas Jefferson
The output is:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "Based on the given knowledge triplets, it's not sufficient to answer the entire question. The triplets only provide information about the person who said "Taste cannot be controlled by law," which is Thomas Jefferson. To answer the second part of the question, it's necessary to have additional knowledge about where Thomas Jefferson's dead."
}

Q: The artist nominated for The Long Winter lived where?
Knowledge Triplets: The Long Winter, schema:author, Laura Ingalls Wilder
Laura Ingalls Wilder, schema:birthPlace, Pepin County
Pepin County, schema:location, Wisconsin
The output is:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "De Smet"
    },
    "R": "Based on the given knowledge triplets, the author of The Long Winter, Laura Ingalls Wilder, lived in Pepin County, Wisconsin. Therefore, the answer to the question is Pepin County"
}

Q: Who is the coach of the team owned by Steve Bisciotti?
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, Baltimore Ravens
Steve Bisciotti, schema:owns, Baltimore Ravens
Steve Bisciotti, schema:founder, Allegis Group
The output is:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "Based on the given knowledge triplets, the coach of the team owned by Steve Bisciotti is not explicitly mentioned. However, it can be inferred that the team owned by Steve Bisciotti is the Baltimore Ravens, a professional sports team. Therefore, additional knowledge about the current coach of the Baltimore Ravens can be used to answer the question."
}

Q: Rift Valley Province is located in a nation that uses which form of currency?
Knowledge Triplets: Rift Valley Province, schema:location, Kenya
Kenya, yago:officialLanguage, Swahili
Kenya, schema:currency, Kenyan Shilling
The output is:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "Kenyan shilling"
    },
    "R": "Based on the given knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency. Therefore, the answer to the question is Kenyan shilling"
}

Q: The country with the National Anthem of Bolivia borders which nations?
Knowledge Triplets: Narendra Modi, yago:leader, India
Narendra Modi, schema:birthPlace, Vadnagar
Vadnagar, schema:location, India
The output is:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "Based on the given knowledge triplets, Narendra Modi is the leader of India. However, the triplets do not explicitly mention the countries that border India. To answer the question, additional knowledge about India’s neighboring countries is required."
}

Now you need to directly output the results of the following question in JSON format (must include "A" and "R") without other information or notes.
Q: """

# In use, and contains Freebase specific relations
# Needs to be changed
prune_entity_prompt = """
Which entities in the following list ([] in Triples) can be used to answer question? Please provide the minimum possible number of entities, and strictly adhering to the constraints mentioned in the question. 
Here is an example:
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Triplets: Tobin Armbrust, schema:producer ['The Resident', 'So Undercover', 'Let Me In', 'Begin Again', 'The Quiet Ones', 'A Walk Among the Tombstones']
Output: ['So Undercover']

Now you need to directly output the entities from [] in Triplets for the following question in list format without other information or notes.
Q: """

# In use, and contains Freebase specific relations
# Needs to be changed
update_mem_prompt = """Based on the provided information (which may have missing parts and require further retrieval) and your own knowledge, output the currently known information required to achieve the subobjectives.
Here is an example:
Q: Find the person who said "Taste cannot be controlled by law", what did this person die from?
Subobjectives: ['Search the person who said "Taste cannot be controlled by law"', 'Search the cause of death for that person']
Memory: 
Knowledge Triplets: Taste cannot be controlled by law., schema:author, [Thomas Jefferson]
Output: {
    "1": "Thomas Jefferson said 'Taste cannot be controlled by law'.",
    "2": "It is not mentioned, and I also don't know."
}

Now you need to directly output the results of the following question in JSON format without other information or notes. 
Q: """

# In use, and contains Freebase specific relations
# Needs to be changed
answer_depth_prompt = """Please answer the question based on the memory, related knowledge triplets and your knowledge.

Here are five example:
Q: Find the person who said \"Taste cannot be controlled by law\", what did this person die from?
Memory: {
    "1": "The triplet provides the information that Thomas Jefferson said this sentence.",
    "2": "No triplet provides this information."
}
Knowledge Triplets: Taste cannot be controlled by law., schema:author, [Thomas Jefferson]
Output:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "The person who said "Taste cannot be controlled by law," is Thomas Jefferson. It is still uncertain to answer the entire question"
}

Q: The artist nominated for The Long Winter lived where?
Memory: {
    "1": "The triplets provide the information that the author of The Long Winter is Laura Ingalls Wilder.",
    "2": "The triplets provide this information that Laura Ingalls Wilder lived in De Smet."
}
Knowledge Triplets: The Long Winter, schema:author, [Laura Ingalls Wilder]
Laura Ingalls Wilder, schema:birthPlace, [Pepin County]
Pepin County, schema:location, [Wisconsin]
Output:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "Pepin County"
    },
    "R": "The author of The Long Winter is Laura Ingalls Wilder, and Laura Ingalls Wilder lived in Pepin County."
}

Q: Who is the coach of the team owned by Steve Bisciotti?
Memory: {
    "1": "The triplets provide the information that Steve Bisciotti owns Baltimore Ravens.",
    "2": "No triplets provide the information."
}
Knowledge Triplets: Steve Bisciotti, sports.professional_sports_team.owner_s, [Baltimore Ravens]
Steve Bisciotti, schema:owns, [Baltimore Ravens]
Steve Bisciotti, schema:founder, [Allegis Group]
Output:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "The team owned by Steve Bisciotti is Baltimore Ravens based on knowledge triplets. The coach of the team owned by Steve Bisciotti is not explicitly mentioned."
}

Q: Rift Valley Province is located in a nation that uses which form of currency?
Memory: {
    "1": "The triplets provide the information that Rift Valley Province is located in Kenya.",
    "2": "The triplets provide the information that Kenya uses the Kenyan shilling as its currency."
}
Knowledge Triplets: Rift Valley Province, schema:location, [Kenya]
Kenya, yago:officialLanguage, [Swahili]
Kenya, schema:currency, [Kenyan Shilling]
Output:
{
    "A": {
        "Sufficient": "Yes",
        "Answer": "Kenyan shilling"
    },
    "R": "Based on knowledge triplets, Rift Valley Province is located in Kenya, which uses the Kenyan shilling as its currency."
}

Q: The country with the leader as Narendra Modi borders which nations?
Memory: {
    "1": "The triplets provide the information that Narendra Modi is the leader of India.",
    "2": "No triplets provide the information."
}
Knowledge Triplets: Narendra Modi, yago:leader, India
Narendra Modi, schema:birthPlace, Vadnagar
Vadnagar, schema:location, India
Output:
{
    "A": {
        "Sufficient": "No",
        "Answer": "Null"
    },
    "R": "Based on the given knowledge triplets, Narendra Modi is the leader of India. However, the triplets do not explicitly mention the countries that border India. To answer the question, additional knowledge about India’s neighboring countries is required."
}

Now you need to directly output the results of the following question in JSON format (must include "A" and "R") without other information or notes. If the triplets explicitly contains the answer to the question, prioritize the fact of the triplet over memory.
Q: """

# In use, and contains Freebase specific relations
# Needs to be changed
judge_reverse = """Based on the current set of entities to be retrieved and the known information including memory and triplets, is it necessary to add additional entities for answering question?
Here are two examples:
Q: Which of the countries next to the Caribbean Sea has the smallest country calling code?
Entities set to be retrieved: ['Anguilla', 'Saint Lucia']
Memory: Caribbean contains Antilles and Saint Lucia
Knowledge Triplets: Caribbean See, schema:neighbors, ['Antilles', 'Saint Lucia']
Output: 
{
    "Add": "Yes",
    "Reason": "The entities set ignores other countries in Caribbean."
}

Q: The artist nominated for The Long Winter lived where?
Entities set to be retrieved: ['Laura Ingalls Wilder']
Memory: "The author of The Long Winter is Laura Ingalls Wilder."
Knowledge Triplets: The Long Winter, schema:author, [Laura Ingalls Wilder]
Output: 
{
    "Add": "No",
    "Reason": "Now you need to search where Laura Ingalls Wilder lived."
}

Now you need to directly output the results of the following question in the JSON format (must include "Add" and "Reason") without other information or notes.
Q: """

# In use, and contains Freebase specific relations
# Needs to be changed 
add_ent_prompt = """Please select the fewest necessary entities to be retrieved for answering the Q from Candidate Entities, based on the current known information (Memory), the reason for additional retrieval, and your own knowledge.
Here is an example:
Q: Which of the countries in the Caribbean has the smallest country calling code?
Reason: The entities set ignores other countries in the Caribbean.
Candidate Entities: ['Saint Marie', 'Saint Martin (Island)', 'Viceroy Anguilla', 'Lesser Antilles', 'Barbados', 'British Virgin Islands', 'Leeward Islands', 'British West Indies', 'Caribbean', 'Saint Thomas', 'Bronte International University', 'Collectivity of Saint Martin', 'Southern Caribbean', 'University of Medicine and Health Sciences', 'Soufrière Hills', 'Lucayan Archipelago', 'University of the West Indies', 'Aureus University School of Medicine', 'North America', 'Netherlands Antilles', 'Puerto Rico', 'Chances Peak', 'Clarendon Parish', 'Saint Kitts and Nevis', 'Saint Lucia', 'Americas', 'Caribbean special municipalities of the Netherlands', 'Sandy Hill', 'School of Business and Computer Science, Trincity', 'School of Business and Computer Science, San Fernando', 'School of Business and Computer Science, Champs Fleurs', 'School of Business and Computer Science, Port of Spain', 'Bridgetown', 'St. Martinus University School of Medicine, main campus', 'Higher Institute of Medical Sciences. main campus', 'Grace University, main campus', 'Anguilla']
Memory: {
    "1": "The countries in the Caribbean are Antilles and Latin America.",
    "2": "The country calling codes for Antilles and Latin America are not mentioned.",
    "3": "It is not mentioned which country has the smallest country calling code."
}
Output: ['Barbados', 'Saint Lucia', 'Anguilla']

Now you need to directly output the results for the following Q in the list format without other information or notes.
Q: """


cot_prompt = """Please answer the question according to your knowledge step by step. Here is an example:
Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
The output is:
{
    "A": {
        "Known": "Yes",
        "Answer": "Washington, D.C."
    },
    "R": "First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C."
}

Please directly output the answer in JSON format (must include "A" and "R") without other information or notes.
"""
