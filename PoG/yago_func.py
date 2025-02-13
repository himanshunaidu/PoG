from SPARQLWrapper import SPARQLWrapper, JSON
from utils import *
import random
# from yago_func import *
from prompt_list import *
from yago_utils.constants import PREFIXES, INVALID_PROPERTIES
import json
import time
import openai
import re
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
SPARQLPATH = "http://localhost:8080/bigdata/sparql"  #your own IP and port

def get_prefix_string() -> str:
    """
    Returns the prefixes as a substring of the SPARQL format.
    """
    prefix_list = [f"PREFIX {key}: <{value}>" for key, value in PREFIXES.items()]
    prefix_string = "\n".join(prefix_list)
    return prefix_string

PREFIX_STRING = get_prefix_string()

# This can be done because the prefixes are unique.
PREFIX_VALUES = {value: key for key, value in PREFIXES.items()}

def get_invalid_properties() -> list[str]:
    """
    Returns the invalid properties in URL format.
    """
    invalid_properties = []
    for property in INVALID_PROPERTIES:
        if ":" in property:
            prefix, property = property.split(":")
            if prefix in PREFIXES:
                invalid_properties.append(f"{PREFIXES[prefix]}{property}")
    return invalid_properties

# pre-defined sparqls
sparql_head_relations = """\n%s\n SELECT DISTINCT ?relation\nWHERE {\n %s ?relation ?x .\n}"""
sparql_tail_relations = """\n%s\nSELECT DISTINCT ?relation\nWHERE {\n  ?x ?relation %s .\n}"""
sparql_tail_entities_extract = """%s\nSELECT ?tailEntity\nWHERE {\n%s %s ?tailEntity .\n}""" 
sparql_head_entities_extract = """%s\nSELECT ?tailEntity\nWHERE {\n?tailEntity %s %s  .\n}"""
# Gets the possible labels and IDs (e.g. Wikidata) of the entity. Priority is given to the label.
# We ignore relations such as schema:sameAs
# EDIT: Ignored owl:sameAs as well.
sparql_id = """
%s\n
SELECT DISTINCT ?tailEntity\n
WHERE {\n
    %s rdfs:label ?tailEntity .\n
    FILTER (lang(?tailEntity) = 'en')\n
}
"""

# def check_end_word(s):
#     words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
#     return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    """
    MARK: This function gets all invalid properties (deemed by us) for Yago.
    """
    return relation in INVALID_PROPERTIES


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    # print(results["results"]["bindings"])
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    """
    TODO: Find the equivalent prefix in Yago.

    MARK: The potential problem with this is that different relations may have different prefixes in Yago.
    There might be a function that we have written that replaces all types of prefixes.

    Also, this means that we may be better off NOT removing the prefixes in the first place.
    Instead, we will have to change the other functions that use this function to handle the prefixes.
    """
    # Instead of removing the entire prefix url, we just replace with the prefix key.
    replaced_relations = []
    for relation in relations:
        for prefix, value in PREFIXES.items():
            if relation['relation']['value'].startswith(value):
                replaced_relations.append(relation['relation']['value'].replace(value, f"{prefix}:"))
                break
    return replaced_relations

def replace_entities_prefix(entities):
    """
    Replaces the prefix value (URL) with the prefix key.

    MARK: The potential problem with this is that different entities may have different prefixes in Yago.
    There might be a function that we have written that replaces all types of prefixes.

    Also, this means that we may be better off NOT removing the prefixes in the first place.
    Instead, we will have to change the other functions that use this function to handle the prefixes.
    """
    # Instead of removing the entire prefix url, we just replace with the prefix key.
    replaced_entities = []
    for entity in entities:
        for prefix, value in PREFIXES.items():
            if entity['tailEntity']['value'].startswith(value):
                replaced_entities.append(entity['tailEntity']['value'].replace(value, f"{prefix}:"))
                break
    return replaced_entities


def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (PREFIX_STRING, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return entity_id
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']
    


def select_relations(string, entity_id, head_relations, tail_relations):
    last_brace_l = string.rfind('[')
    last_brace_r = string.rfind(']')
    
    if last_brace_l < last_brace_r:
        string = string[last_brace_l:last_brace_r+1]

    relations=[]
    rel_list = eval(string.strip())
    for relation in rel_list:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": True})
        elif relation in tail_relations:
            relations.append({"entity": entity_id, "relation": relation, "head": False})
    
    if not relations:
        return False, "No relations found"
    return True, relations



def construct_relation_prune_prompt(question, sub_questions, entity_name, total_relations, args):
    return extract_relation_prompt + question + '\nSubobjectives: ' + str(sub_questions) + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations)


def relation_search_prune(entity_id, sub_questions, entity_name, pre_relations, pre_head, question, args):
    sparql_relations_extract_head = sparql_head_relations % (PREFIX_STRING, entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    
    sparql_relations_extract_tail= sparql_tail_relations % (PREFIX_STRING, entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
    else:
        head_relations = list(set(head_relations) - set(pre_relations))

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal
    

    prompt = construct_relation_prune_prompt(question, sub_questions, entity_name, total_relations, args)
    result, token_num = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
    flag, retrieve_relations = select_relations(result, entity_id, head_relations, tail_relations) 

    if flag:
        return retrieve_relations, token_num
    else:
        return [], token_num # format error or too small max_length
    
    
def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (PREFIX_STRING, entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (PREFIX_STRING, relation, entity)
        entities = execurte_sparql(head_entities_extract)


    entity_ids = replace_entities_prefix(entities)
    return entity_ids


def provide_triple(entity_candidates_id, relation):
    entity_candidates = []
    for entity_id in entity_candidates_id:
        # if entity_id.startswith("m."):
        entity_candidates.append(id2entity_name_or_type(entity_id))
        # else:
        #     entity_candidates.append(entity_id)

    if len(entity_candidates) <= 1:
        return entity_candidates, entity_candidates_id


    ent_id_dict = dict(sorted(zip(entity_candidates, entity_candidates_id)))
    entity_candidates, entity_candidates_id = list(ent_id_dict.keys()), list(ent_id_dict.values())
    return entity_candidates, entity_candidates_id

    
def update_history(entity_candidates, ent_rel, entity_candidates_id, total_candidates, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]

    candidates_relation = [ent_rel['relation']] * len(entity_candidates)
    topic_entities = [ent_rel['entity']] * len(entity_candidates)
    head_num = [ent_rel['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, question_string, subquestions, cluster_chain_of_entities, depth, call_num, all_t, start_time, args):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    call_num += 1
    answer, token_num = generate_answer(question, subquestions, cluster_chain_of_entities, args)

    for kk in token_num.keys():
        all_t[kk] += token_num[kk]

    save_2_jsonl(question, question_string, answer, cluster_chain_of_entities, call_num, all_t, start_time, file_name=args.dataset+'_'+args.LLM_type)

def half_stop_no_write(question, question_string, subquestions, cluster_chain_of_entities, depth, call_num, all_t, start_time, args):
    print("No new knowledge added during search depth %d, stop searching." % depth)
    call_num += 1
    answer, token_num = generate_answer(question, subquestions, cluster_chain_of_entities, args)

    for kk in token_num.keys():
        all_t[kk] += token_num[kk]
    return answer, call_num, all_t


def generate_answer(question, subquestions, cluster_chain_of_entities, args): 
    prompt = answer_prompt + question 
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt
    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    return result, token_num


def if_topic_non_retrieve(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def is_all_digits(lst):
    for s in lst:
        if not s.isdigit():
            return False
    return True


def entity_condition_prune(question, total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, ent_rel_ent_dict, entid_name, name_entid, args, model):
    """
    the function refines the initial set of candidates, using both heuristic checks and an LLM-based decision, 
    to focus on the entity that most likely answers the question correctly.
    """
    cur_call_time = 0
    cur_token = {'total': 0, 'input': 0, 'output': 0}

    new_ent_rel_ent_dict = {}
    no_prune = ['time', 'number', 'date']
    filter_entities_id, filter_tops, filter_relations, filter_candidates, filter_head = [], [], [], [], []
    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                if is_all_digits(e_list) or rela in no_prune or len(e_list) <= 1:
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    select_ent = sorted_e_list
                else:
                    if len(e_list) > 10: #all(entid_name[item].startswith('m.') for item in e_list) and
                        e_list = random.sample(e_list, 10)

                    if len(e_list) > 70:
                        sorted_e_list = [entid_name[e_id] for e_id in e_list]
                        topn_entities, topn_scores = retrieve_top_docs(question, sorted_e_list, model, 70)
                        e_list = [name_entid[e_n] for e_n in topn_entities]
                        print('sentence:', topn_entities)

                    prompt = prune_entity_prompt + question +'\nTriples: '
                    sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                    prompt += entid_name[topic_e] + ' ' + rela + ' ' + str(sorted_e_list)

                    cur_call_time += 1
                    result, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
                    for kk in token_num.keys():
                        cur_token[kk] += token_num[kk]

                    last_brace_l = result.rfind('[')
                    last_brace_r = result.rfind(']')
                    
                    if last_brace_l < last_brace_r:
                        result = result[last_brace_l:last_brace_r+1]
                    
                    try:
                        result = eval(result.strip())
                    except:
                        result = result.strip().strip("[").strip("]").split(', ')
                        result = [x.strip("'") for x in result]

                    select_ent = sorted(result)
                    select_ent = [x for x in select_ent if x in sorted_e_list]

                if len(select_ent) == 0 or all(x == '' for x in select_ent):
                    continue

                if topic_e not in new_ent_rel_ent_dict.keys():
                    new_ent_rel_ent_dict[topic_e] = {}
                if h_t not in new_ent_rel_ent_dict[topic_e].keys():
                    new_ent_rel_ent_dict[topic_e][h_t] = {}
                if rela not in new_ent_rel_ent_dict[topic_e][h_t].keys():
                    new_ent_rel_ent_dict[topic_e][h_t][rela] = []
                
                for ent in select_ent:
                    if ent in sorted_e_list:
                        new_ent_rel_ent_dict[topic_e][h_t][rela].append(name_entid[ent])
                        filter_tops.append(entid_name[topic_e])
                        filter_relations.append(rela)
                        filter_candidates.append(ent)
                        filter_entities_id.append(name_entid[ent])
                        if h_t == 'head':
                            filter_head.append(True)
                        else:
                            filter_head.append(False)


    if len(filter_entities_id) == 0:
        return False, [], [], [], [], new_ent_rel_ent_dict, cur_call_time, cur_token


    cluster_chain_of_entities = [[(filter_tops[i], filter_relations[i], filter_candidates[i]) for i in range(len(filter_candidates))]]
    return True, cluster_chain_of_entities, filter_entities_id, filter_relations, filter_head, new_ent_rel_ent_dict, cur_call_time, cur_token

def add_pre_info(add_ent_list, depth_ent_rel_ent_dict, new_ent_rel_ent_dict, entid_name, name_entid, args):
    add_entities_id = sorted(add_ent_list)
    add_relations, add_head = [], []
    topic_ent = set()

    for cur_ent in add_entities_id:
        flag = 0
        for depth, ent_rel_ent_dict in depth_ent_rel_ent_dict.items():
            for topic_e, h_t_dict in ent_rel_ent_dict.items():
                for h_t, r_e_dict in h_t_dict.items():
                    for rela, e_list in r_e_dict.items():
                        if cur_ent in e_list:
                            if topic_e not in new_ent_rel_ent_dict.keys():
                                new_ent_rel_ent_dict[topic_e] = {}
                            if h_t not in new_ent_rel_ent_dict[topic_e].keys():
                                new_ent_rel_ent_dict[topic_e][h_t] = {}
                            if rela not in new_ent_rel_ent_dict[topic_e][h_t].keys():
                                new_ent_rel_ent_dict[topic_e][h_t][rela] = []
                            if cur_ent not in new_ent_rel_ent_dict[topic_e][h_t][rela]:
                                new_ent_rel_ent_dict[topic_e][h_t][rela].append(cur_ent)
                            
                            if not flag:
                                add_relations.append(rela)
                                if h_t == 'head':
                                    add_head.append(True)
                                else:
                                    add_head.append(False)
                                flag = 1


        if not flag:
            print('none pre relation')
            print(cur_ent)
            flag = 1
            add_head.append(-1)
            add_relations.append('')
            if cur_ent not in new_ent_rel_ent_dict.keys():
                new_ent_rel_ent_dict[cur_ent] = {}

    return add_entities_id, add_relations, add_head, new_ent_rel_ent_dict

def update_memory(question, subquestions, ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args):
    with open(q_mem_f_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()
    prompt = update_mem_prompt + question + '\nSubobjectives: '+str(subquestions)+'\nMemory: ' + his_mem

    chain_prompt = ''
    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                chain_prompt += entid_name[topic_e] + ' ' + rela + ' ' + str(sorted_e_list) + '\n'

    prompt += "\nKnowledge Triplets:\n" + chain_prompt

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False, False)
    
    mem = extract_memory(response)
    print(mem)
    with open(q_mem_f_path+'/mem', 'w', encoding='utf-8') as f:
        f.write(mem)
    return token_num


def reasoning(question, subquestions, ent_rel_ent_dict, entid_name, cluster_chain_of_entities, q_mem_f_path, args):
    with open(q_mem_f_path+'/mem', 'r', encoding='utf-8') as f:
        his_mem = f.read()

    prompt = answer_depth_prompt + question + '\nMemory: ' + his_mem

    chain_prompt = ''

    for topic_e, h_t_dict in sorted(ent_rel_ent_dict.items()):
        for h_t, r_e_dict in sorted(h_t_dict.items()):
            for rela, e_list in sorted(r_e_dict.items()):
                sorted_e_list = [entid_name[e_id] for e_id in sorted(e_list)]
                chain_prompt += entid_name[topic_e] + ', ' + rela + ', ' + str(sorted_e_list) + '\n'

    prompt += "\nKnowledge Triplets:\n" + chain_prompt

    response, token_num = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, False)
    
    answer, reason, sufficient = extract_reason_and_anwer(response)
    return response, answer, sufficient, token_num

    

if __name__ == "__main__":
    entity = 'yago:Belgium'
    print(entity)
    print(id2entity_name_or_type(entity))

    print(INVALID_PROPERTIES)
    print([relation for relation in ["rdf:type", "rdfs:label", "schema:sameAs"] if not abandon_rels(relation)])