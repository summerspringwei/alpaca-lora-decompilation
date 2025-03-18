
from typing import Dict, List
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import torch.nn.functional as F
import torch

def get_reward_discrete(output: Dict) -> float:
    reward = -1
    if not output["target_execution_success"]:
        reward = 0
    elif isinstance(output["predict_execution_success"], list) and any(
            output["predict_execution_success"]):
        reward = 1
    elif isinstance(output["predict_compile_success"], list) and any(
            output["predict_compile_success"]):
        reward = 0.8
    else:
        reward = -1

    return reward


def get_reward_length(output: Dict, tokenizer: LlamaTokenizer, input_max_length: int) -> float:
    # length = 
    tokenized_question = tokenizer(output["output"], truncation=True)
    length = len(tokenized_question["input_ids"])
    reward = -1
    if not output["target_execution_success"]:
        reward = 0
    elif isinstance(output["predict_execution_success"], list) and any(
            output["predict_execution_success"]):
        reward = 1
    elif isinstance(output["predict_compile_success"], list) and any(
            output["predict_compile_success"]):
        reward = 0.8
    else:
        reward = -1
    reward = torch.tanh(torch.tensor(reward * length / input_max_length, dtype=torch.float32))
    return reward



def compute_logit_distance(logits1, logits2, method="cosine"):
    """
    Computes the distance between two logits tensors.

    Args:
        logits1 (torch.Tensor): The first logits tensor (shape: [batch_size, seq_len, vocab_size]).
        logits2 (torch.Tensor): The second logits tensor (shape: [batch_size, seq_len, vocab_size]).
        method (str, optional): The distance metric to use. Options: "cosine", "euclidean". Defaults to "cosine".

    Returns:
        torch.Tensor: The distance tensor (shape: [batch_size, seq_len]).
        OR
        str: An error message if the input shapes are incompatible or if an invalid method is provided.
    """
    if logits1.shape != logits2.shape:
        return "Error: Logits tensors must have the same shape."

    if method == "cosine":
        # Reshape to (batch_size * seq_len, vocab_size) for cosine similarity calculation
        logits1_flat = logits1.view(-1, logits1.shape[-1])
        logits2_flat = logits2.view(-1, logits2.shape[-1])
        # Calculate cosine similarity
        similarity = F.cosine_similarity(logits1_flat, logits2_flat)
        # Convert similarity to distance (1 - similarity)
        distance = 1 - similarity
        # Reshape back to (batch_size, seq_len)
        return distance
    elif method == "euclidean":
        # Calculate euclidean distance
        distance = torch.cdist(logits1, logits2, p=2) # p=2 for Euclidean distance
        return distance.squeeze(-1) # Remove the last dimension as it is 1 after cdist for single vectors
    else:
        return "Error: Invalid distance method. Choose 'cosine' or 'euclidean'."



def get_logits_distance(validation_results: List[Dict], tokenizer: LlamaTokenizer, model: LlamaForCausalLM) -> Dict:
    predict_irs = [record["predict"][0] for record in validation_results]
    target_irs = [record["output"] for record in validation_results]
    combined_irs = predict_irs + target_irs
    inputs = tokenizer(combined_irs, return_tensors="pt", padding=True).to(model.device)
    # Get the logits
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    batch, seq_len, vocab_size = logits.size()
    predict_logits = logits[:len(predict_irs), :, :]
    target_logits = logits[len(predict_irs):, :, :]
    distance = compute_logit_distance( predict_logits, target_logits, method="cosine")
    validation_results = [dict(record, distance=distance[i].item()) for i, record in enumerate(validation_results)]

    return validation_results



def get_distance_reward(output: Dict) -> float:
    reward = -1
    if not output["target_execution_success"]:
        reward = 0
    elif isinstance(output["predict_execution_success"], list) and any(
            output["predict_execution_success"]):
        reward = 1
    elif isinstance(output["predict_compile_success"], list) and any(
            output["predict_compile_success"]):
        reward = 0.2
    else:
        reward = -output["distance"]
    reward = torch.tanh(torch.tensor(reward, dtype=torch.float32))
    return reward

