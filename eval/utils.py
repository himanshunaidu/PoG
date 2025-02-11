import json

def read_output(file_path, question_string):
    answered_dict = {}
    file_path = '../PoG/'+file_path + '.jsonl'
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            answered_dict[data[question_string]] = data

    answered_list = list(answered_dict.values())
    return answered_list

def prepare_dataset_for_eval(dataset_name, output_file):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    elif dataset_name == 'dynamickgqa': # MARK: Added case for 'dynamickgqa'
        with open('../data/dynamickgqa_test_output.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa}.")
        exit(-1)

    output_datas = read_output(output_file, question_string)
    return datas, question_string, output_datas


def align(dataset_name, question_string, data, ground_truth_datas, aname_dict, alias_dict, add_ans_alias_dict):
    answer_list= []
    origin_data = [j for j in ground_truth_datas if j[question_string] == data[question_string]][0]
    if dataset_name == 'cwq':
        if 'answers' in origin_data:
            answers = origin_data["answers"]
        else:
            answers = origin_data["answer"]
        # for answer in answers:
        #     alias = answer['aliases']
        #     ans = answer['answer']
        #     alias.append(ans)
        #     answer_list.extend(alias)
        answer_list.append(answers)

    elif dataset_name == 'webqsp':
        answers = origin_data["Parses"]
        for answer in answers:
            for name in answer['Answers']:
                if name['EntityName'] == None:
                    answer_list.append(name['AnswerArgument'])
                else:
                    answer_list.append(name['EntityName'])

    elif dataset_name == 'grailqa':
        answers = origin_data["answer"]
        for answer in answers:
            if "entity_name" in answer:
                answer_list.append(answer['entity_name'])
            else:
                answer_list.append(answer['answer_argument'])

    elif dataset_name == 'simpleqa':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'qald':
        answers = origin_data["answer"]
        for answer in answers:
            answer_list.append(answers[answer])
        
    elif dataset_name == 'webquestions':
        answer_list = origin_data["answers"]

    elif dataset_name == 'trex' or dataset_name == 'zeroshotre':
        answers = origin_data["answer"]
        answer_list.append(answers)

    elif dataset_name == 'creak':
        answer = origin_data['label']
        answer_list.append(answer)

    elif dataset_name == 'dynamickgqa':
        answer = origin_data["answer_readable"]
        # for answer in answers:
        answer_list.append(answer)

    return list(set(answer_list)), origin_data
    

def exact_match(response, answers):
    clean_result = response.strip().replace(" ","").lower()
    for answer in answers:
        clean_answer = answer.strip().replace(" ","").lower()
        if clean_result == clean_answer or clean_result in clean_answer or clean_answer in clean_result:
            return True
    return False
