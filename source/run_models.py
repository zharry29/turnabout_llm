import json
import os
import asyncio
import re
import argparse
import traceback
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    # General args
    parser.add_argument('-m', '--model', type=str, help='model name')
    parser.add_argument('-p', '--prompt', type=str, help='prompt name')
    parser.add_argument('--context', type=str, help='full, sum')
    parser.add_argument('--case', type=str, default="ALL", help='If ALL, run all cases; if a case number like 3-4-1, run that case; if a case number followed by a "+" like 3-4-1+, run that case and all cases after it.')
    parser.add_argument('--no_description', action='store_true')
    parser.add_argument('--data', type=str, default='aceattorney', help='dataset name, aceattorney or danganronpa')

    # Evaluation args
    parser.add_argument('-a', '--all', action='store_true', help='Evaluate all existing models')
    return parser

# OS operations

def get_output_dir(MODEL, PROMPT, CONTEXT, CASE, NO_DESCRIPTION, DATA):
    output_dir = f'../output/{MODEL.split("/")[-1]}_prompt_{PROMPT}'
    if CONTEXT is not None:
        output_dir += f"_context_{CONTEXT}"
    if NO_DESCRIPTION:
        output_dir += "_desc_none"    
    if CASE != "ALL":
        output_dir += f"_case_{CASE}"
    if DATA == 'danganronpa':
        output_dir += f"_data_{DATA}"
    return output_dir

def get_fnames(data_dir, output_dir, CASE, eval=False, verbose=True):
    """Return list of .json files"""
    all_fnames = sorted([
        fname for fname 
        in os.listdir(data_dir) 
        if fname.endswith('.json')
        # if not fname.startswith(('7-', '8-'))  # skip test split
    ])
    fnames = []
    if CASE == "ALL":
        fnames = all_fnames
    else:
        for i, fname in enumerate(all_fnames):
            if fname.startswith(CASE.strip('+')):
                if CASE.endswith('+'):
                    fnames = all_fnames[i:]
                else:
                    fnames = [fname]
                break

    if verbose: print(f"<get_fnames> Found {len(fnames)} total cases")

    if not eval:
        fnames_to_check = fnames.copy()
        for fname in fnames_to_check:
            if os.path.exists(os.path.join(output_dir, fname.split('.')[0] + '.jsonl')):
                # print(f"Skipping existing {fname.split('.')[0] + '.jsonl'}")
                fnames.remove(fname)

    if verbose:print(f"<get_fnames> Running {len(fnames)} new cases")
    return fnames

# Prompt builders

def build_prompt_prefix_suffix(prompt_arg):
    with open("prompts/" + prompt_arg + ".json", 'r') as file:
        # parse json
        data = json.load(file)
        prompt_prefix = data['prefix']
        prompt_suffix = data['suffix']

    # Load cot examples
    if "base_one_shot" in prompt_arg: # base one shot
        with open("prompts/example_one_shot.txt", "r") as file:
            example_one_shot = file.read()
        prompt_prefix = prompt_prefix.format(example_one_shot=example_one_shot)

    elif "cot_one_shot" in prompt_arg:
        with open("prompts/exampe_one_shot_cot.txt", "r") as file:
            example_one_shot_cot = file.read()
        prompt_prefix = prompt_prefix.format(example_one_shot_cot=example_one_shot_cot)

    # elif "few_shot" in prompt_arg:
    #     with open("prompts/example_few_shot.txt", "r") as file:
    #         example_few_shot = file.read()
    #     prompt_prefix = prompt_prefix.format(example_few_shot=example_few_shot)

    return prompt_prefix, prompt_suffix

def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        characters = []
        evidences = []
        prev_context = re.sub(r'\n+', ' ', data.get("previousContext", ""))  # Remove newlines

        for character in data.get('characters', {}):
            characters.append(character)
        for evidence in data.get('evidences', {}):
            evidences.append(evidence)
        turns = []
        for turn in data.get("turns", []):
            if turn["noPresent"]:
                continue
            testimonies = []
            for testimony in turn['testimonies']:
                testimonies.append(testimony)
            turn_dict = {
                'characters': characters,
                'evidences': evidences,
                'testimonies': testimonies,
                'newContext': re.sub(r'\n+', ' ', turn['newContext']),
                'summarizedContext': turn.get('summarizedContext', "")
            }
            turns.append(turn_dict)
        return turns, prev_context

def truncate_context(context, MODEL):
    context_size = len(context)  # Count characters, not tokens

    # Truncate context for specific models
    max_context_size = -1
    if "deepseek" in MODEL or "deepseek-chat" in MODEL:  # Roughly 66000 tokens for deepseek
        max_context_size = 230000
    # elif "70b" in MODEL:  # Roughtly 20000 tokens for 70b
    #     max_context_size = 80000
    if max_context_size != -1:
        start_idx = context_size - max_context_size
        context = "..." + context[start_idx:]

    return context

def build_prompt(
    turns, 
    prev_context, 
    PROMPT_PREFIX, 
    PROMPT_SUFFIX, 
    CONTEXT, 
    NO_DESCRIPTION, 
    MODEL
):
    prompts = []
    context_sofar = ""
    for turn in turns:
        context_is_added = False
        new_context = turn['newContext']  
        new_context = re.sub(r'\n+', ' ', new_context)  # Remove newlines
        context_sofar += new_context
        if CONTEXT is None:
            prompt = ""
        else:
            prompt = "Story:\n"
            full_context = "" 
            if CONTEXT == "full":
                full_context += prev_context + "\n" + context_sofar + "\n"
            elif CONTEXT == "sum":
                full_context += turn['summarizedContext'] + "\n"

            full_context = truncate_context(full_context, MODEL)

            prompt += full_context

        character_counter = 0
        prompt += "Characters:\n"
        for character in turn['characters']:
            prompt += f"Character {character_counter}\n"
            prompt += f"Name: {character['name']}\n"
            if not NO_DESCRIPTION:
                prompt += f"Description: {character['description1']}\n"
            character_counter += 1

        # Format evidences
        evidence_counter = 0
        evidences = []
        for evidence in turn['evidences']:
            evidence_string = f"Evidence {evidence_counter}\n"
            evidence_string += f"Name: {evidence['name']}\n"
            if not NO_DESCRIPTION:
                evidence_string += f"Description: "
                descriptions = []
                for key in evidence.keys():
                    if 'description' in key:
                        descriptions.append(evidence[key])
                evidence_string += " ".join(descriptions) + "\n"
            evidences.append(evidence_string)
            evidence_counter += 1
        
        # Format testimonies
        testimony_counter = 0
        testimonies = []
        for testimony in turn['testimonies']:
            testimony_string = f"Testimony {testimony_counter}\n"
            testimony_string += f"Testimony: {testimony['testimony']}\n"
            testimony_string += f"Person: {testimony['person']}\n"
            # Provide context if needed
            if "source" in testimony and \
                testimony["source"].get("is_self_contained", "yes") == "no" and \
                CONTEXT is None:
                context_span = testimony["source"]["context_span"]
                if not context_is_added and not NO_DESCRIPTION:
                    for i, evidence_string in enumerate(evidences):
                        evidence_spans = testimony["source"]["evidence_span"]
                        if isinstance(evidence_spans, str):
                            evidence_spans = [evidence_spans]
                        for evidence_span in evidence_spans:
                            if evidence_span in evidence_string: # Find evidence
                                evidences[i] += f"{context_span}\n"  # Add context span
                    context_is_added = True
            testimony_counter += 1
            testimonies.append(testimony_string)
        
        # Build rest of the prompt
        prompt += f"Evidences:\n{''.join(evidences)}\nTestimonies:\n{''.join(testimonies)}\n"
        prompts.append(PROMPT_PREFIX + prompt + PROMPT_SUFFIX)
    return prompts

# Model runners

def get_json_answer(multiline_string):
    lines = multiline_string.splitlines()
    target = lines[-1]
    try:
        json_answer = json.loads(target)
        cot = "\n".join(lines[:-1])
    except json.JSONDecodeError:
        try:
            target = lines[-2]
            json_answer = json.loads(target)
            cot = "\n".join(lines[:-2])
        except json.JSONDecodeError:
            # print(f"Error parsing JSON: {multiline_string[:-1]}")
            json_answer = {}
            cot = ""
    return json_answer, cot

def run_model(prompts, client, client_name):
    has_error = False
    answer_jsons = []
    cots = []
    for prompt in prompts:
        #print(prompt)
        try:
            cot = ""
            if type(client).__name__ == "Kani":  # Use kani api
                async def run_async_model():
                    response = await client.chat_round_str(prompt, temperature=0.6)
                    #print(response)
                    return response

                full_answer = asyncio.run(run_async_model())

            elif type(client).__name__ == "OpenAI":  # Use openai api
                response = client.chat.completions.create(
                    model=client_name,
                    messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                    stream=False
                )
                full_answer = response.choices[0].message.content

                # Get COT
                try: 
                    cot = response.choices[0].message.reasoning_content
                    print(f"<run_model> COT returned for {client_name}")
                    full_answer = cot + "\n\n" + full_answer
                except Exception as e:
                    print(f"<run_model> When trying to get COT for {client_name}: {e}")
                    print(f"<run_model> No COT for {client_name}")

            else:
                raise ValueError(f"<run_model> Unknown client: {client}")

            answer_json, parsed_cot = get_json_answer(full_answer)

            if answer_json == {}:
                has_error = True

            if cot == "":  # Only when model does not return its COT field
                cot = parsed_cot
                
        except Exception as e:  # Handle errors, such as rate limit, context window, etc.
            print(f"<run_model> {traceback.format_exc()}")
            answer_json, cot = {}, ""
            has_error = True

        answer_jsons.append(answer_json)
        cots.append(cot)

    return answer_jsons, cots, has_error

def load_model(model, config_path="models.json"):
    with open(config_path, 'r') as file:
        config = json.load(file)
    model = config.get(model, model)
    if "/" in model:  # a huggingface model
        from kani import Kani
        from kani.engines.huggingface import HuggingEngine
        import torch

        torch.cuda.empty_cache()
        engine = HuggingEngine(
            model_id = model, 
            use_auth_token=True, 
            model_load_kwargs={"device_map": "auto"}
        )
        client = Kani(engine, system_prompt="")

        name = model.split("/")[-1]

    else:  # an api model
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv("../.env")

        model_key = model

        if any(m_name in model for m_name in ["gpt", "o3", "o4"]):
            model_key = "openai"

        elif "deepseek" in model:  # deepseek-reasoner (R1), deepseek-chat (V3)
            model_key = "deepseek"

        auth = {
            "deepseek": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": "https://api.deepseek.com",
                "name": model
            },
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY"),
                "name": model
            }
        }

        if "base_url" in auth[model_key]:
            client = OpenAI(
                api_key=auth[model_key]["api_key"],
                base_url=auth[model_key]["base_url"]
            )
        else:
            client = OpenAI(
                api_key=auth[model_key]["api_key"],
            )

        name = auth[model_key]["name"]

    return client, name

def create_batch(fnames, MODEL, PROMPT, CONTEXT, NO_DESCRIPTION, data_dir):
    max_token_key = "max_tokens" if "gpt" in MODEL else "max_completion_tokens"
    max_token_val = 1000 if "gpt" in MODEL else 7000
    batch = []
    skip_count = 0
    for fname in fnames:
        turns, prev_context = parse_json(os.path.join(data_dir, fname))
        if turns == []:
            skip_count += 1
            continue
        PROMPT_PREFIX, PROMPT_SUFFIX = build_prompt_prefix_suffix(PROMPT)
        prompts = build_prompt(turns, prev_context, PROMPT_PREFIX, PROMPT_SUFFIX, CONTEXT, NO_DESCRIPTION, MODEL)
        # print(prompts)
        for i, prompt in enumerate(prompts):
            request = {
                "custom_id": f"{fname.split('.')[0]}_{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."}, 
                        {"role": "user", "content": prompt}
                    ],
                    max_token_key: max_token_val
                }
            }
            batch.append(request)
    
    assert len(batch) > 0, "Must have at least 1 batch item"
    print(f"Skipped {skip_count} cases")

    return batch

def submit_batch_job(jsonl_path, client, output_dir):
    batch_input_file = client.files.create(
        file=open(jsonl_path, "rb"),
        purpose="batch"
    )
    batch_input_file_id = batch_input_file.id

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "turnabout llm"
        }
    )
    batch_job_id = batch_job.id
    with open(os.path.join(output_dir, "batch_api_metadata.json"), "w") as f:
        f.write(json.dumps({
            "batch_job_id": batch_job_id, 
            "batch_file_id": batch_input_file_id
        }, indent=4))

    return batch_job_id

def run_batch_job(fnames, MODEL, PROMPT, CONTEXT, NO_DESCRIPTION, client, output_dir, data_dir):
    # Create batch
    batch = create_batch(fnames, MODEL, PROMPT, CONTEXT, NO_DESCRIPTION, data_dir)

    jsonl_path = os.path.join(output_dir, "batchinput.jsonl")
    # If exists, run the incomplete batch job instead
    if os.path.exists(jsonl_path):
        # Create duplicate file
        dup_count = 1
        new_jsonl_path = jsonl_path
        while os.path.exists(new_jsonl_path):
            new_jsonl_path = os.path.join(output_dir, f"batchinput_{dup_count}.jsonl")
            dup_count += 1
        print(f"<run_batch_job> Creating duplicate batch input file: {os.path.basename(new_jsonl_path)}")

        # Filter incomplete input
        output_data = []
        output_files = sorted([
            os.path.join(output_dir, output_path) 
            for output_path in os.listdir(output_dir) 
            if output_path.startswith("batchoutput")
        ])  # Guaranteed to be mutually exclusive
        for output_file in output_files:
            with open(output_file, "r") as f:
                output_data += [json.loads(line) for line in f]
        output_ids = sorted(set([item["custom_id"] for item in output_data]))
        new_input_data = []
        for input_item in batch:
            if input_item["custom_id"] not in output_ids:
                new_input_data.append(input_item)
        print(f"<run_batch_job> Found {len(new_input_data)} incomplete cases")

        # Create new batch input file
        with open(new_jsonl_path, "w") as f:
            for request in new_input_data:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
        jsonl_path = new_jsonl_path

    else:
        with open(jsonl_path, "w") as f:
            for request in batch:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")
    
    batch_job_id = submit_batch_job(jsonl_path, client, output_dir)
    return batch_job_id

# Main loop

def run_job(fnames, MODEL, PROMPT, CONTEXT, NO_DESCRIPTION, client, client_name, output_dir, data_dir):
    error_count = 0
    skip_count = 0
    for fname in fnames:
        # Parse and build prompt
        if error_count > 5:
            print(f"<run_job> Terminating due to {error_count}+ json parsing errors")
            break

        print(fname)
        turns, context = parse_json(os.path.join(data_dir, fname))
        if turns == []:  # Skip cases with no turns
            skip_count += 1
            continue
        PROMPT_PREFIX, PROMPT_SUFFIX = build_prompt_prefix_suffix(PROMPT)
        prompts = build_prompt(turns, context, PROMPT_PREFIX, PROMPT_SUFFIX, CONTEXT, NO_DESCRIPTION, MODEL)

        # Answer
        answer_jsons, cots, has_error = run_model(prompts, client, client_name)
        if has_error:
            error_count += 1
            print(f"<run_job> Error when running the model for {fname}")
            continue  # Skip cases with errors
        for answer_json in answer_jsons:
            print(answer_json)

        # Log
        with open(os.path.join(output_dir, fname.split('.')[0] + '.jsonl'), 'w') as file:
            for answer_json in answer_jsons:
                file.write(json.dumps(answer_json) + "\n")
        with open(os.path.join(output_dir, fname.split('.')[0] + '_outputs.json'), 'w') as file:
            json_response = []
            for idx, (answer_json, cot) in enumerate(zip(answer_jsons, cots)):
                json_response.append({ 
                    "idx": idx,
                    "prompt": prompts[idx],
                    "response_json": answer_json,
                    "cot": cot
                })
            file.write(json.dumps(json_response, indent=2))
    
    print(f"Skipped {skip_count} cases")

if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    MODEL = args.model
    PROMPT = args.prompt
    CASE = args.case if args.case else "ALL"
    CONTEXT = args.context
    NO_DESCRIPTION = args.no_description
    DATA = args.data

    if DATA == 'aceattorney':
        data_dir = '../data/aceattorney_data/final'
    elif DATA == 'danganronpa':
        data_dir = '../data/danganronpa_data/final'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Make output dir
    output_dir = get_output_dir(MODEL, PROMPT, CONTEXT, CASE, NO_DESCRIPTION, DATA)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as file:
        json.dump({
            'model': MODEL,
            'prompt': PROMPT,
            'context': "none" if CONTEXT is None else CONTEXT,
            'case': CASE if CASE != "ALL" else "all",
            'no_description': NO_DESCRIPTION,
            'data': DATA,
            'timestamp': timestamp
        }, file, indent=2)
    # Load model
    client, client_name = load_model(MODEL)

    # Collect cases
    fnames = get_fnames(data_dir, output_dir, CASE)

    # Run cases
    if any(name in MODEL for name in ["o3", "o4", "gpt"]):
        run_batch_job(fnames, MODEL, PROMPT, CONTEXT, NO_DESCRIPTION, client, output_dir, data_dir)
    else:
        run_job(fnames, MODEL, PROMPT, CONTEXT, NO_DESCRIPTION, client, client_name, output_dir, data_dir)
