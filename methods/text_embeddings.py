import torch
from compel import Compel


def get_concept_ind(tokenizer, prompts, concepts, template="{}"):
    concept_inds = []
    for prompt in prompts:
        input_ids = tokenizer(prompt)["input_ids"]
        tokens = [tokenizer.decode(tk) for tk in input_ids]
        concept_ind = [tokens.index(template.format(c)) for c in concepts]
        concept_inds.append(concept_ind)
    return concept_inds


def init_label_token_embeds(tokenizer, text_encoder, label_id_to_cat, template="<{}>"):
    placeholder_tokens = []
    for c in label_id_to_cat:
        placeholder_tokens.append(template.format(c))

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    
    if num_added_tokens == 0: return
    # assert num_added_tokens == len(label_id_to_cat), "some labels were not added to the tokenizer, or the tokenizer already contains them"

    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)
    label_id_to_token_ids = [tokenizer.encode(c, add_special_tokens=False) for c in label_id_to_cat]
    
    text_encoder.resize_token_embeddings(len(tokenizer))

    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for label_id, token_id in enumerate(placeholder_token_ids):
            label_token_ids = label_id_to_token_ids[label_id]
            token_embeds[token_id] = token_embeds[label_token_ids].mean(dim=0)

            
@torch.no_grad()
def get_text_embeddings_simple(tokenizer, text_encoder, prompt):
    prompts = [prompt]
    input_ids = tokenizer(
        prompts,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device)
    text_embeds = text_encoder(input_ids).last_hidden_state # last hidden state
    return text_embeds


@torch.no_grad()
def get_text_embeddings(
    tokenizer, text_encoder, labels, label_id_to_cat,
    max_length=None, truncation=True
):
    init_label_token_embeds(tokenizer, text_encoder, [l.replace(" ", "-") for l in label_id_to_cat])
    concepts = [l.replace(" ", "-") for l in labels]
    label_token_template = "<{}>"
    prompts = ["a photo of " + ", ".join(label_token_template.format(c) for c in concepts)]
    concept_ind = get_concept_ind(tokenizer, prompts, concepts, template=label_token_template)[0]
    input_ids = tokenizer(
        prompts,
        # padding="max_length",
        # max_length=model.tokenizer.model_max_length if max_length is None else max_length,
        # truncation=truncation,
        return_tensors="pt",
    ).input_ids.to(text_encoder.device)
    text_embeds = text_encoder(input_ids).last_hidden_state # last hidden state
    return text_embeds, concept_ind, prompts[0]


@torch.no_grad()
def get_contextualized_text_embeddings(tokenizer, text_encoder, labels, max_length=None, truncation=True):
    template = "a good photo of some {}"
    prompts = [template.format(c) for c in labels]
    text_input = tokenizer(
        prompts,
        padding=True,
        # max_length=model.tokenizer.model_max_length if max_length is None else max_length,
        # truncation=truncation,
        return_tensors="pt",
    )
    max_length = text_input.input_ids.shape[-1]
    embeddings = text_encoder(text_input["input_ids"].to(text_encoder.device)).last_hidden_state # (num_concepts, max_len, embed_size)
    sos = embeddings[:, 0].mean(dim=0, keepdim=True) # average of <sos> embeds, (1, embed_size)
    concept_embeds = []
    eos = []
    for i in range(len(labels)):
        eos_idx = (text_input["input_ids"][i] == tokenizer.eos_token_id).long().argmax(dim=-1).item()
        some_idx = (text_input["input_ids"][i] == 836).long().argmax(dim=-1).item() # index of the "some" token
        concept_embeds.append(embeddings[i, (some_idx+1):eos_idx].mean(dim=0, keepdim=True)) # average over tokens of this label, (1, embed_dim,)
        eos.append(embeddings[i, eos_idx]) # (embed_dim,)
    concept_embeds = torch.cat(concept_embeds, dim=0) # (num_concepts, embed_dim)
    eos = torch.stack(eos, dim=0).mean(dim=0, keepdim=True) # average of <eos> embeds, (1, 768)
    embeddings = torch.cat([sos, concept_embeds, eos], dim=0)
    concept_ind = torch.arange(1, len(labels)+1).tolist() # index of labels in input_ids
    return embeddings.unsqueeze(0), concept_ind, " ".join(labels)


@torch.no_grad()
def get_txt_embeddings(tokenizer, text_encoder, labels, label_id_to_cat, cat_to_label_id, use_compel=False):
    # for labels that are broken into multiple tokens, average those embeddings (after passing the input_ids to the text encoder)
    def get_concept_tokens(concepts, cat_to_label_id, label_id_to_token_ids):
        concept_tokens = []
        for c in concepts:
            concept_id = cat_to_label_id[c]
            concept_tokens.append(label_id_to_token_ids[concept_id])
        return concept_tokens
    def get_indices(concept_tokens, input_ids):
        # each concept can be tokenized into multiple token
        # this function returns, for each concept, a list of the indicies of its token(s) in input_ids
        # for instance: [[10000, 2000], [3000]], [0, 11, 10000, 20000, 34, 3000, 42690]
        # returns [[1, 2], [4]]
        concept_indices = []
        for tokens in concept_tokens:
            concept_index = []
            for i in range(len(input_ids) - len(tokens) + 1):
                if input_ids[i:i + len(tokens)].tolist() == tokens:
                    concept_index.extend(list(range(i,i+len(tokens))))
            concept_indices.append(concept_index)
        return concept_indices
    def relabel(label, template):
        return template.format(label)
    def construct_prompt(labels, template="{}"):
        if len(labels) > 1:
            new_labels = [relabel(l, template) for l in labels[:-1]]
            prompt = "A photo of " + ", ".join(new_labels) + ", and " + relabel(labels[-1], template)
        else:
            prompt = "A photo of " + relabel(labels[0], template)
        # prompt = prompt + ", high quality, 8k, photo"
        return prompt

    assert not isinstance(labels[0], list), "does not support batch mode yet"
    label_id_to_token_ids = [tokenizer.encode(c, add_special_tokens=False) for c in label_id_to_cat]

    prompt = construct_prompt(labels)
    input_ids = tokenizer([prompt], return_tensors="pt").input_ids[0]

    if use_compel:
        prompt_compel = construct_prompt(labels, template="{}++")
        compel_proc = Compel(tokenizer=tokenizer, text_encoder=text_encoder, device=text_encoder.device)
        embeds = compel_proc(prompt_compel)
    else:
        embeds = text_encoder(input_ids[None].to(text_encoder.device)).last_hidden_state

    embeds = embeds.to(text_encoder.dtype)

    concept_tokens = get_concept_tokens(labels, cat_to_label_id, label_id_to_token_ids)
    concept_indices = get_indices(concept_tokens, input_ids)
    concept_ind = [c[0] for c in concept_indices]

    return embeds, concept_ind, concept_indices, prompt


@torch.no_grad()
def get_label_embeddings(tokenizer, text_encoder, labels, label_id_to_cat, cat_to_label_id, use_compel=False):
    """For each label in labels, construct a sentence with that label and get its embedding
    
    Returns:
        embeds: (num_labels, seq_len, embed_dim)
        concept_indices: list of list of indices of the concept tokens in the input_ids,
            e.g [[4,5],[4],[4,5,6]]
    """
    def get_concept_tokens(concepts, cat_to_label_id, label_id_to_token_ids):
        """Get the tokens of each concept in concepts
        """
        concept_tokens = []
        for c in concepts:
            concept_id = cat_to_label_id[c]
            concept_tokens.append(label_id_to_token_ids[concept_id])
        return concept_tokens
    def relabel(label, template):
        return template.format(label)
    def get_indices(concept_tokens, input_ids):
        """For each sequence in input_ids, return the indices of the concept tokens in that sequence
        concept_tokens: list (for each sequence) of list of concept tokens (each sequence only has one concept)
        """
        assert len(concept_tokens) == len(input_ids)
        concept_indices = []
        for c_tokens, seq in zip(concept_tokens, input_ids):
            concept_index = []
            for tk in c_tokens:
                concept_index.append((seq == tk).nonzero(as_tuple=True)[0].view(1))
            concept_indices.append(torch.cat(concept_index))
        return concept_indices
    def construct_prompts(labels, prompt_template, label_template="{}"):
        """Construct a prompt for each label in labels
        """
        labels = [relabel(l, label_template) for l in labels]
        prompts = []
        for l in labels:
            prompts.append(prompt_template.format(l))
        return prompts

    label_id_to_token_ids = [tokenizer.encode(c, add_special_tokens=False) for c in label_id_to_cat]
    prompt_template = "a photo of a {}"
    prompts = construct_prompts(labels, prompt_template)
    input_ids = tokenizer(prompts, padding=True, return_tensors="pt").input_ids

    if use_compel:
        prompts_compel = construct_prompts(labels, prompt_template, label_template="{}++")
        compel_proc = Compel(tokenizer=tokenizer, text_encoder=text_encoder, device=text_encoder.device)
        embeds = compel_proc(prompts_compel)
    else:
        embeds = text_encoder(input_ids.to(text_encoder.device))[0]
    embeds = embeds.to(text_encoder.dtype)

    concept_tokens = get_concept_tokens(labels, cat_to_label_id, label_id_to_token_ids)
    concept_indices = get_indices(concept_tokens, input_ids)
    concept_ind = [c[0].item() for c in concept_indices]

    return embeds, concept_ind, concept_indices, prompts